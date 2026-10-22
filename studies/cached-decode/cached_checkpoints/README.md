# Source checkpoints

v1 and v2 are immutable source snapshots preceding any model execution. Their hash manifests include all Python source/tests plus the cached-decode protocol. Source directories retain only new cached-decode files; inherited dependencies remain unchanged in the tree and are identified by the manifests. Earlier transport/pretrained result directories are not evidence for this milestone.

Public allowlist is additive to the existing public repository and deliberately excludes model files, environments, private material and inherited result copies.
