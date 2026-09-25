"""Plot measured evidence gains separately from frozen-head pose outcomes."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root;out=root/'measurement_integrity_v30';out.mkdir(exist_ok=True)
    strict=json.loads((root/'dense_measurement_v29/strict_metrics.json').read_text())['metrics']['heldout']['augmented']
    pose=json.loads((root/'completion_mass_v31/comparison.json').read_text())['nonsym_augmented_rotation']['arms']
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for i,(label,key) in enumerate([('Precision','precision'),('Recall','recall')]):
        for j,(kind,color) in enumerate([('original','#65768a'),('dense','#168b95')]):
            value=100*strict[kind][key]
            bar=axes[0].bar(i+(j-.5)*.32,value,.30,label=kind if i==0 else None,color=color)
            axes[0].bar_label(bar,fmt='%.1f',padding=3,fontsize=9)
    axes[0].set(xticks=[0,1],xticklabels=['Precision','Recall'],ylim=(0,105),ylabel='RGB-visible valid-depth pixels (%)',title='Selector: more measured pixels retained')
    axes[0].legend(frameon=False,loc='upper left')
    kinds=['original','original_balanced','dense','dense_balanced'];labels=['Original','Original\n+ mass cap','Dense','Dense\n+ mass cap']
    bars=axes[1].bar(range(4),[pose[k]['rotation'] for k in kinds],color=['#65768a','#92a2b7','#168b95','#78bcc2'])
    axes[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[1].axhline(10,color='#555555',ls='--',lw=1,label='Initial rotation error')
    axes[1].set(xticks=range(4),xticklabels=labels,ylim=(0,11.2),ylabel='Rotation error (degrees; lower is better)',title='Pose: no consistent downstream gain')
    axes[1].legend(frameon=False,loc='lower left')
    fig.suptitle('Controlled added-occlusion holdout; fixed JEPA and pose weights',fontsize=12)
    fig.text(.5,.01,'Pixel labels do not certify CAD depth agreement. Pose: 48 non-symmetric +/-10 degree cases. Not native validation.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(out/'measurement_vs_pose.png',dpi=180);fig.savefig(out/'measurement_vs_pose.pdf');plt.close(fig)
    with (out/'deployable_interventions.csv').open('w') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['arm','rotation_deg','center_error_d','rgb_visible_measured_weight_fraction','rgb_invisible_measured_weight_fraction'])
        for kind in kinds:
            p=pose[kind];writer.writerow([kind,p['rotation'],p['center_d'],p['true_measurement_fraction_of_all_weight'],p['false_measurement_fraction_of_all_weight']])

if __name__=='__main__':main()
