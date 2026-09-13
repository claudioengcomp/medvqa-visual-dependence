"""Regenerates Figures 2-5 of the paper from the aggregated figure data.

Input:  data/fig_data.json  (override with the FIG_DATA env var)
Output: figures/            (override with the FIG_OUT env var)

Usage:
    python src/analysis/make_figs.py
"""
import json,sys,os,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.lines import Line2D
FIG_DATA = os.environ.get('FIG_DATA', 'data/fig_data.json')
OUT = os.environ.get('FIG_OUT', 'figures')
FD = json.load(open(FIG_DATA)); C = FD['combos']; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'pdf.fonttype':42})
COL={'M3AE':'#2A62B8','QWEN2VL':'#D06C14','SMOLVLM':'#249A4C','MEDVLTHINKER':'#A63496','M0':'#7A8794'}
def fam(b): return next(k for k in COL if b.startswith(k))
def style(b):
    f=fam(b); c=COL[f]
    if f=='M3AE': return dict(marker='s',mfc=c,mec=c)
    if f=='MEDVLTHINKER': return dict(marker='D',mfc=c,mec=c)
    if f=='M0': return dict(marker='o',mfc=c,mec=c)
    return dict(marker='o',mfc=(c if 'LORA' in b else 'white'),mec=c)
NAME={'M3AE':'M3AE','QWEN2VL_LORA':'Qwen2-VL LoRA','QWEN2VL_DORA':'Qwen2-VL DoRA','SMOLVLM_LORA':'SmolVLM2 LoRA','SMOLVLM_DORA':'SmolVLM2 DoRA','MEDVLTHINKER_ZEROSHOT':'MedVLThinker (zero-shot)','M0_question_only':'Question-only (M0)'}
def short(b):
    for k,v in NAME.items():
        if b.startswith(k): return v
DSN={'KVASIR':'Kvasir-VQA-x1','SLAKE':'SLAKE','VQARAD':'VQA-RAD'}; DSO=['SLAKE','VQARAD','KVASIR']
ORDER=['M3AE','QWEN2VL_LORA','QWEN2VL_DORA','SMOLVLM_LORA','SMOLVLM_DORA','MEDVLTHINKER_ZEROSHOT','M0_question_only']
def combos(ds): 
    out=[]
    for o in ORDER:
        for b in C:
            if b.startswith(o) and b.endswith(ds): out.append(b)
    return out
# ---------- Fig 1: Acc x VDD, C1 and C2
fig,axes=plt.subplots(2,3,figsize=(6.9,2.25),sharex=True,sharey=True)
for j,ds in enumerate(DSO):
    for i,cond in enumerate(['C1','C2']):
        ax=axes[i,j]
        for b in combos(ds):
            d=C[b]; v=d[cond]
            if not v: continue
            x,y=d['acc0'][0]*100,v[0]*100
            if d['seeds']>1: ax.errorbar(x,y,xerr=d['acc0'][1]*100,yerr=v[1]*100,fmt='none',ecolor=COL[fam(b)],elinewidth=0.7,capsize=0,alpha=.8)
            ax.plot(x,y,ms=5,mew=1.2,ls='none',**style(b))
        ax.set_xlim(20,95); ax.set_ylim(-3,52); ax.set_yticks([0,20,40]); ax.grid(True,lw=0.3,alpha=.5)
        if i==0: ax.set_title(DSN[ds],fontsize=8.5)
        if j==0: ax.set_ylabel(f'VDD({cond})\n[pp]',fontsize=7.5)
        if i==1: ax.set_xlabel('Acc(C0) [%]')
h=[Line2D([],[],ls='none',ms=5,mew=1.2,**style(k+'_X'),label=NAME[k]) for k in ['M3AE','QWEN2VL_LORA','QWEN2VL_DORA','SMOLVLM_LORA','SMOLVLM_DORA','MEDVLTHINKER_ZEROSHOT','M0_question_only']]
fig.legend(handles=h,loc='lower center',ncol=7,frameon=False,fontsize=7,bbox_to_anchor=(0.5,-0.04),handletextpad=0.3,columnspacing=1.0)
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(f'{OUT}/fig1_scatter.pdf',bbox_inches='tight'); plt.close(fig)
# ---------- Fig 2: floor
fig,axes=plt.subplots(2,1,figsize=(3.4,3.45))
for ax,ds in zip(axes,['SLAKE','VQARAD']):
    rows=[b for b in combos(ds) if not b.startswith('M0') and C[b]['C1closed']]
    floor=C[f'M0_question_only_{ds}']['acc0_closed'][0]*100
    for k,b in enumerate(rows):
        a0=C[b]['acc0_closed'][0]*100; a1=a0-C[b]['C1closed'][0]*100; y=len(rows)-k
        ax.annotate('',xy=(a1,y),xytext=(a0,y),arrowprops=dict(arrowstyle='->',color=COL[fam(b)],lw=1.1))
        ax.plot(a0,y,ms=5,mew=1.2,ls='none',**style(b))
        ax.text(a1-1,y,f'{a1:.1f}',ha='right',va='center',fontsize=6.5,color=COL[fam(b)])
    ax.axvline(floor,color='#444',ls='--',lw=0.9); ax.text(floor,len(rows)+0.7,f'question-only {floor:.1f}%',ha='center',fontsize=6.5,color='#444')
    ax.set_yticks(range(1,len(rows)+1)); ax.set_yticklabels([short(b) for b in rows][::-1],fontsize=7)
    ax.set_xlim(22,92); ax.set_ylim(0.4,len(rows)+1.2); ax.set_xlabel('accuracy on closed questions [%]'); ax.set_title(DSN[ds],fontsize=8.5); ax.grid(True,axis='x',lw=0.3,alpha=.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_floor.pdf',bbox_inches='tight'); plt.close(fig)
# ---------- Fig 3: (a) dumbbell blank vs donor, (b) VDD vs f
fig,(ax,bx)=plt.subplots(1,2,figsize=(6.9,2.3),gridspec_kw={'width_ratios':[1.25,1]})
ylabels=[];y=-1.4
for ds in ['KVASIR','SLAKE','VQARAD']:
    cond='C6' if ds=='VQARAD' else 'C5'
    rows=[b for b in combos(ds) if C[b]['seeds']==3]
    ax.text(-2,y,f"{DSN[ds]} · {cond}",fontsize=7,fontweight='bold',va='center',ha='left',clip_on=False)
    for b in rows:
        a=C[b]['C1at'+cond]; d=C[b][cond+'e']
        if not a or not d: continue
        y-=1; ax.plot([a[0]*100,d[0]*100],[y,y],color='#bbb',lw=2.2,zorder=1)
        ax.plot(a[0]*100,y,marker='o',ms=5,mfc='white',mec=COL[fam(b)],mew=1.3,ls='none',zorder=2)
        ax.plot(d[0]*100,y,marker='o',ms=5,mfc=COL[fam(b)],mec=COL[fam(b)],ls='none',zorder=3)
        ylabels.append((y,short(b)))
    y-=1.7
ax.set_yticks([p for p,_ in ylabels]); ax.set_yticklabels([n for _,n in ylabels],fontsize=7); ax.set_xlim(-3,56); ax.set_xlabel('VDD on the swapped questions [pp]'); ax.grid(True,axis='x',lw=0.3,alpha=.5)
ax.legend(handles=[Line2D([],[],marker='o',ms=5,mfc='white',mec='#444',ls='none',label='C1 (blank image)'),Line2D([],[],marker='o',ms=5,mfc='#444',mec='#444',ls='none',label='C5/C6 (donor image)')],loc='lower right',frameon=False,fontsize=6.5)
ax.set_title('(a) image removal vs. restricted swap',fontsize=8,loc='left',pad=10)
MK={'C2':'^','C5':'o','C6':'s'}
for p in FD['fvsvdd']:
    if p['subset']!='all': continue
    bx.plot(p['f']*100,p['vdd']*100,marker=MK[p['cond']],ms=4.5,mfc=(COL[fam(p['model'])] if p['ds']=='KVASIR' else 'white'),mec=COL[fam(p['model'])],mew=1,ls='none',alpha=.9)
for p in FD['fvsvdd']:
    if p['subset']=='all': continue
    bx.plot(p['f']*100,p['vdd']*100,marker='o',ms=4.5,mfc=COL[fam(p['model'])],mec='k',mew=0.8,ls='none')
import numpy as np
xs=np.linspace(0,75,10)
for k,lab in [(0.6,'0.6'),(0.8,'0.8')]: bx.plot(xs,k*xs,color='#999',lw=0.7,ls=':'); bx.text(26,k*26+1.5,f'VDD = {lab}·f',fontsize=6,color='#666',ha='left')
bx.set_xlim(20,78); bx.set_ylim(-3,56); bx.set_xlabel('f: swapped questions whose answer changes [%]'); bx.set_ylabel('VDD on the swapped questions [pp]'); bx.grid(True,lw=0.3,alpha=.5)
bx.legend(handles=[Line2D([],[],marker='^',ms=4.5,mfc='#444',mec='#444',ls='none',label='C2'),Line2D([],[],marker='o',ms=4.5,mfc='#444',mec='#444',ls='none',label='C5'),Line2D([],[],marker='s',ms=4.5,mfc='#444',mec='#444',ls='none',label='C6'),Line2D([],[],marker='o',ms=4.5,mfc='white',mec='#444',ls='none',label='SLAKE (hollow)'),Line2D([],[],marker='o',ms=4.5,mfc='#444',mec='k',ls='none',label='Kvasir C5 same/cross proc.')],loc='lower right',frameon=False,fontsize=6)
bx.set_title('(b) swap cost scales with changed answers',fontsize=8,loc='left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_swap.pdf',bbox_inches='tight'); plt.close(fig)
# ---------- Fig 4: CCR stacked
fig,axes=plt.subplots(2,1,figsize=(3.4,3.2))
for ax,(ds,cond) in zip(axes,[('KVASIR','C5'),('SLAKE','C6')]):
    rows=[b for b in combos(ds) if b in FD['ccr'] and cond in FD['ccr'][b] and not b.startswith('M0')]
    for k,b in enumerate(rows):
        d=FD['ccr'][b][cond]; y=len(rows)-k; x=0
        for key,col in [('followed_donor','#2A62B8'),('kept_original','#D06C14'),('neither','#C9CFD6')]:
            w=d[key]*100; ax.barh(y,w,left=x,height=0.62,color=col,edgecolor='white',lw=0.6)
            if w>9: ax.text(x+w/2,y,f'{w:.0f}',ha='center',va='center',fontsize=6.5,color=('white' if key!='neither' else '#222'))
            x+=w
    ax.set_yticks(range(1,len(rows)+1)); ax.set_yticklabels([short(b) for b in rows][::-1],fontsize=7); ax.set_xlim(0,100); ax.set_xlabel('informative swapped questions [%]',fontsize=7)
    ax.set_title(f"{DSN[ds]} · {cond} (n = {FD['ccr'][rows[0]][cond]['n']//C[rows[0]]['seeds']:,} per seed)",fontsize=8,loc='left')
fig.legend(handles=[plt.Rectangle((0,0),1,1,color='#2A62B8',label="answer of the donor image (CCR)"),plt.Rectangle((0,0),1,1,color='#D06C14',label='original answer kept'),plt.Rectangle((0,0),1,1,color='#C9CFD6',label='other answer')],loc='lower center',ncol=2,frameon=False,fontsize=6.5,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(f'{OUT}/fig4_ccr.pdf',bbox_inches='tight'); plt.close(fig)
print('figs done', os.listdir(OUT))
