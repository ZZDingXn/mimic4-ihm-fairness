"""Reformat the existing baseline subgroup plot without recalculating estimates.

Contract: a quantitative 2x2 grid contrasting discrimination (top row) with
mean risk bias (bottom row), stratified by insurance/ethnicity (columns).
Field mapping and units match baseline_figures.py: auroc and its stored 95% CI;
prediction_minus_observed and difference_lower/upper_95 multiplied by 100 (pp).
All nine target groups and both models are retained. Source aggregates and all
old figures remain unchanged. Outputs are a 170-mm vector PDF and 300-dpi PNG.
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
QA = STUDY/'tem/english_revision/figure_qa'
QA.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(STUDY/'tem/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = STUDY/'results/baseline'
metrics = pd.read_csv(SOURCE/'subgroup_metrics.csv')
calibration = pd.read_csv(SOURCE/'calibration_in_large.csv')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,
                     'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':0.7,'legend.frameon':False})
fig, axes = plt.subplots(2,2,figsize=(170/25.4,150/25.4))
fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.89,wspace=.32,hspace=.44)
colors={'lr':'#4f7cac','cw_lstm':'#c75b39'}
for col,attribute in enumerate(['insurance','ethnicity']):
    groups=metrics.loc[metrics.attribute.eq(attribute),'group'].drop_duplicates().tolist()
    for row in range(2):
        ax=axes[row,col]
        for offset,model in zip([-.12,.12],['lr','cw_lstm']):
            frame=metrics if row==0 else calibration
            sub=frame[(frame.attribute==attribute)&(frame.model==model)].set_index('group').loc[groups]
            fields=['auroc','auroc_lower_95','auroc_upper_95'] if row==0 else ['prediction_minus_observed','difference_lower_95','difference_upper_95']
            val,lo,hi=[sub[f].to_numpy(float)*(1 if row==0 else 100) for f in fields]
            ax.errorbar(np.arange(len(groups))+offset,val,yerr=np.vstack([val-lo,hi-val]),fmt='o',
                        markersize=4,capsize=2.5,linewidth=.9,color=colors[model],label='LR' if model=='lr' else 'LSTM')
        ax.set_xticks(np.arange(len(groups)),groups,rotation=25,ha='right',rotation_mode='anchor')
        ax.set_ylabel('AUROC' if row==0 else 'Mean risk bias (pp)')
        if row==0:ax.set_ylim(.65,1.01);ax.set_title(attribute.title(),pad=8)
        else:ax.set_ylim(-4.5,6.5);ax.axhline(0,color='.3',linewidth=.7)
        ax.grid(axis='y',alpha=.18)
        ax.annotate('abcd'[row*2+col],xy=(0,1),xycoords='axes fraction',xytext=(-25,12),textcoords='offset points',weight='bold',fontsize=11)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,1),ncol=2)
fig.canvas.draw()
(STUDY/'figures/revised').mkdir(exist_ok=True)
fig.savefig(STUDY/'figures/revised/baseline_subgroups_en.pdf')
fig.savefig(STUDY/'figures/revised/baseline_subgroups_en.svg')
fig.savefig(STUDY/'figures/revised/baseline_subgroups_en.png',dpi=300)
plt.close(fig)
