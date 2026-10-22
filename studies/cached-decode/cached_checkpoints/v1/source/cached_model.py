"""Stage-owned KV cache. No other stage's layer tensors or KV are loaded."""
from .pretrained_model import Partition


class CachedPartition(Partition):
    def __init__(self, directory, stage):
        super().__init__(directory, stage)
        # Cache indices are local; rotary positions remain global sequence positions.
        for index, layer in enumerate(self.layers):
            layer.self_attn.layer_idx = index
        self.reset()

    def reset(self):
        from transformers.cache_utils import DynamicCache
        self.cache = DynamicCache()
        self.position = 0

    def __call__(self, value):
        import torch
        if self.stage == 0:
            if value.dtype != torch.int64 or value.ndim != 2 or value.shape[0] != 1 or not 1 <= value.shape[1] <= 64:
                raise ValueError('invalid token tensor')
            if value.min() < 0 or value.max() >= 49152:
                raise ValueError('invalid token ID')
            hidden = self.embedding(value)
        else:
            if value.dtype != torch.float32 or value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 576 or not 1 <= value.shape[1] <= 64 or not torch.isfinite(value).all():
                raise ValueError('invalid activation tensor')
            hidden = value
        length = hidden.shape[1]
        if self.position + length > 64 or (self.position and length != 1):
            raise ValueError('invalid cached sequence length')
        positions = torch.arange(self.position, self.position + length)
        ids = positions.unsqueeze(0)
        keys = torch.arange(self.position + length)
        mask = torch.zeros((length, self.position + length), dtype=torch.float32)
        mask.masked_fill_(keys[None, :] > positions[:, None], torch.finfo(torch.float32).min)
        rotary = self.rotary(hidden, ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=mask[None, None], position_ids=ids,
                           past_key_value=self.cache, output_attentions=False, use_cache=True,
                           cache_position=positions, position_embeddings=rotary)[0]
        self.position += length
        if len(self.cache.key_cache) != 15 or any(t.shape[-2] != self.position for t in self.cache.key_cache):
            raise ValueError('cache ownership/length invariant failed')
        if self.stage == 1:
            hidden = self.head(self.norm(hidden[:, -1:, :]))[:, 0, :]
        return hidden

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.cache.key_cache + self.cache.value_cache)
