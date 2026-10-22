"""Render a calculated summary of 12unique prefixes, never72independent samples."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=json.loads((ROOT/'results/transport_comparison.json').read_text())
    diagnostic=json.loads((ROOT/'results/transport_posthoc_diagnostic.json').read_text())
    rows=[r for r in primary['samples'] if r['link_index']==0 and r['repeat']==0]
    codecs=['fp32','fp16','int8'];fractions=[];agreements=[]
    for codec in codecs:
        chosen=[r for r in rows if r['codec']==codec]
        assert len(chosen)==len({r['case'] for r in chosen})==12
        fractions.append(100*statistics.median(r['activation_frame_bytes']/next(b['activation_frame_bytes'] for b in rows if b['case']==r['case'] and b['codec']=='fp32') for r in chosen))
        agreements.append(sum(r['next_token_agreement'] for r in chosen))
    int8=[r for r in diagnostic['rows'] if r['codec']=='int8']
    zeros=[100*r['zero_fraction_after_decode'] for r in int8];l2=[100*r['activation_relative_l2_error'] for r in int8]
    fp16error=max(r['logit_max_abs_error'] for r in rows if r['codec']=='fp16')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'transport-fixed12-v1','figure.facecolor':'#f7f9fc','axes.facecolor':'#f7f9fc','text.color':'#17273a','axes.labelcolor':'#17273a','xtick.color':'#425169','ytick.color':'#17273a','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,8.5),gridspec_kw={'width_ratios':[1.15,1]})
    fig.subplots_adjust(left=.16,right=.95,top=.73,bottom=.43,wspace=.30)
    fig.text(.07,.945,'Smaller activations are not always faithful activations',fontsize=19,weight='bold')
    fig.text(.07,.897,'12 fixed prefixes • SmolLM2-135M • same FP32 compute • only boundary encoding changes',fontsize=11)
    fig.text(.07,.847,'FP16 kept all 12 next-token choices; this whole-tensor INT8 codec kept none.',fontsize=13,weight='bold',color='#964226')
    colors=['#607b98','#087e8b','#c05a3b'];positions=[2,1,0]
    axes[0].barh(positions,fractions,color=colors,height=.55)
    axes[0].set_yticks(positions,['FP32','FP16','INT8\nwhole tensor'],fontsize=13)
    axes[0].set_xlim(0,117);axes[0].set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
    axes[0].set_title('Full-frame bytes',loc='left',fontsize=14,weight='bold',pad=14)
    axes[0].set_xlabel('Median fraction of matched FP32 frame',fontsize=10)
    for position,value in zip(positions,fractions):axes[0].text(value+2,position,f'{value:.1f}%',va='center',fontsize=12,weight='bold')
    axes[1].barh(positions,agreements,color=colors,height=.55)
    axes[1].set_yticks(positions,[]);axes[1].set_xlim(0,14);axes[1].set_xticks([0,3,6,9,12]);axes[1].set_title('Reference next-token matches',loc='left',fontsize=13,weight='bold',pad=14)
    axes[1].set_xlabel('Out of 12 unique prefixes, not quality scores',fontsize=10)
    for position,value in zip(positions,agreements):axes[1].text(value+.25,position,f'{value}/12',va='center',fontsize=12,weight='bold')
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines['left'].set_visible(False)
    fig.text(.07,.337,'POST-HOC RAW-ARRAY AUDIT',fontsize=10,weight='bold',color='#964226')
    fig.text(.07,.287,f'INT8 zeroed {min(zeros):.2f}–{max(zeros):.2f}% of activation coordinates.',fontsize=16,weight='bold')
    fig.text(.07,.238,f'Global relative reconstruction L2 error was only {min(l2):.2f}–{max(l2):.2f}%: that did not preserve outputs.',fontsize=12)
    fig.text(.07,.185,f'FP16 maximum absolute logit drift: {fp16error:.5f}. Agreement on 12 prefixes is not a quality guarantee.',fontsize=10.5)
    fig.text(.07,.115,'Source: retained 216 paired timing runs, with 12 unique numerical observations per codec.',fontsize=10)
    fig.text(.07,.078,'This is one model/split/codec setting, not a finding that INT8 generally fails. No physical-network claim.',fontsize=10)
    fig.text(.07,.037,'Data, hashes and reproduction: TRANSPORT_RESULTS.md • python -m mesh.transport_figure',fontsize=9,color='#526277')
    (ROOT/'figures').mkdir(exist_ok=True)
    for suffix in ['svg','png']:
        kwargs={'metadata':{'Date':None}} if suffix=='svg' else {}
        fig.savefig(ROOT/f'figures/transport_codec_summary.{suffix}',dpi=180,bbox_inches='tight',**kwargs)
    plt.close(fig)
    print('Saved calculated SVG/PNG using12unique prefixes and separately labeled post-hoc evidence.')


if __name__=='__main__':main()
