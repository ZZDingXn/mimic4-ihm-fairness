"""Redraw existing aggregate figures; no training, predictions or bootstrap."""
from pathlib import Path
import json
import os
import pandas as pd

STUDY=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(STUDY/'tem/matplotlib')
from .baseline_figures import plot_cohort_flow, plot_performance, plot_subgroups
from .plots import write_publication_plots

def main():
    results=STUDY/'results';base=results/'baseline';out=STUDY/'figures'
    out.mkdir(parents=True,exist_ok=True)
    for language in ('en','zh'):
        plot_cohort_flow(pd.read_csv(base/'cohort_flow.csv'),language,out/f'cohort_flow_{language}')
        plot_performance(json.loads((base/'overall_metrics.json').read_text()),language,out/f'baseline_performance_{language}')
        plot_subgroups(pd.read_csv(base/'subgroup_metrics.csv'),pd.read_csv(base/'calibration_in_large.csv'),language,out/f'baseline_subgroups_{language}')
    write_publication_plots(*[pd.read_csv(results/name) for name in ('policy_metrics.csv','fairness_gaps.csv','calibration_bins.csv','bootstrap_confidence_intervals.csv')],out,out,STUDY/'tem')
    import matplotlib.pyplot as plt
    curves=pd.read_csv(results/'training_curves.csv')
    for lang in ('en','zh'):
        zh=lang=='zh'
        with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['SimHei','DejaVu Sans'] if zh else ['DejaVu Sans'],'axes.unicode_minus':False}):
            fig,axes=plt.subplots(1,2,figsize=(8.5,4),constrained_layout=True)
            for ax,(attr,title) in zip(axes,[('insurance_group','保险' if zh else 'Insurance'),('ethnicity_group','族裔' if zh else 'Ethnicity')]):
                df=curves[curves.target_attribute.eq(attr)]
                ax.plot(df.epoch,df.train_loss,label='加权训练BCE' if zh else 'Weighted train BCE')
                ax.plot(df.epoch,df.validation_loss,label='未加权验证BCE' if zh else 'Unweighted validation BCE')
                epoch=int(df.selected_epoch.iloc[0])
                ax.axvline(epoch,ls='--',color='0.4',label=('所选轮次 ' if zh else 'Selected epoch ')+str(epoch))
                ax.set(title=title,xlabel='训练轮次' if zh else 'Epoch',ylabel='BCE');ax.grid(alpha=.2);ax.legend(fontsize=10)
            for ext in ('png','pdf'):fig.savefig(out/f'training_curves_{lang}.{ext}',dpi=200,bbox_inches='tight')
            plt.close(fig)

if __name__=='__main__':main()
