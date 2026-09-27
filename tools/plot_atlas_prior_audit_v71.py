"""Show oracle-only decoder distortion separately from model prediction."""
import json,argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r70=json.loads((a.root/'atlas_prior_quality_v70/outcome.json').read_text())
r71=json.loads((a.root/'local_surface_projection_v71/outcome.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,angle in zip(axes,(10,60)):
 for j,(records,name,label,color) in enumerate(((r70,'perfect_prior','Existing atlas','#939ba8'),(r70,'perfect_prior_tight','Tighter soft prior','#dd9950'),(r71,'perfect_prior','Local CAD projection','#4686b6'))):
  vals=[records[f'{angle}/True/{region}/{name}']['canonical_xyz_mm'] for region in ('real','proxy')]
  ax.bar([x+(j-1)*.25 for x in (0,1)],vals,width=.24,label=label,color=color)
 ax.set_xticks([0,1],['Artificially hidden real','CAD proxy'])
 ax.set_title(f'{angle}° initial error, heavy occlusion');ax.set_ylabel('Final XYZ error (mm)')
 ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend(fontsize=8)
fig.suptitle('Oracle-only decoder audit: exact canonical coordinates supplied\nNOT learned model accuracy; original targets retained',fontsize=12)
fig.savefig(a.root/'local_surface_projection_v71/oracle_decoder_audit.png',dpi=180)
fig.savefig(a.root/'local_surface_projection_v71/oracle_decoder_audit.pdf')
