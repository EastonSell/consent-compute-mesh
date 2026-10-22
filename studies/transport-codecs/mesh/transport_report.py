"""Post-run descriptive tables only; does not execute models or select cases."""
import csv
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    result=json.loads((ROOT/'results/transport_comparison.json').read_text())
    rows=[];prefixes=[]
    for codec in ['fp32','fp16','int8']:
        samples=[s for s in result['samples'] if s['codec']==codec]
        for link in [0,1]:
            selected=[s for s in samples if s['link_index']==link]
            ratios=[]
            for sample in selected:
                baseline=next(s for s in result['samples'] if s['codec']=='fp32' and s['link_index']==link and s['case']==sample['case'] and s['repeat']==sample['repeat'])
                ratios.append(sample['activation_frame_bytes']/baseline['activation_frame_bytes'])
            rows.append({'codec':codec,'link_index':link,'paired_runs':len(selected),'distinct_prefixes':len({s['case'] for s in selected}),
                'median_split_ms':statistics.median(s['split_seconds'] for s in selected)*1000,
                'median_reference_ms':statistics.median(s['reference_seconds'] for s in selected)*1000,
                'median_activation_encode_ms':statistics.median(s['worker_encode_seconds'][0] for s in selected)*1000,
                'median_activation_decode_ms':statistics.median(s['worker_decode_seconds'][1] for s in selected)*1000,
                'median_observed_emulation_ms':statistics.median(s['emulated_wait_observed_seconds'] for s in selected)*1000,
                'median_frame_fraction_of_matched_fp32':statistics.median(ratios),
                'max_logit_abs_error':max(s['logit_max_abs_error'] for s in selected),
                'max_kl_reference_to_codec_nats':max(s['kl_reference_to_codec'] for s in selected),
                'unique_prefix_next_token_agreements':sum(next(s for s in selected if s['case']==case)['next_token_agreement'] for case in {s['case'] for s in selected})})
        for case in dict.fromkeys(s['case'] for s in samples):
            selected=[s for s in samples if s['case']==case]
            # Verify repeat/link-invariance rather than pretending repeats add accuracy evidence.
            signatures={(s['reference_next_token'],s['codec_next_token'],s['logit_max_abs_error'],s['kl_reference_to_codec']) for s in selected}
            if len(signatures)!=1:raise ValueError('unexpected numerical outcome variation across timing repeats')
            sample=selected[0]
            prefixes.append({key:sample[key] for key in ['case','codec','token_count','reference_next_token','codec_next_token','next_token_agreement','reference_top1_top2_logit_margin','logit_max_abs_error','logit_rms_error','kl_reference_to_codec','activation_frame_bytes']})
    for name,records in [('transport_summary.csv',rows),('transport_prefix_metrics.csv',prefixes)]:
        with (ROOT/'results'/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('Generated6descriptive condition rows and36prefix/codec numerical rows; no model execution.')


if __name__=='__main__':main()
