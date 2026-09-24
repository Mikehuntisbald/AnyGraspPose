"""Paired fixed40 normal audit report and valid-normal angle histograms."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    records={n:json.loads((a.root/n/'summary.json').read_text()) for n in ['source_mlp','update1000']}
    assert all(r['completed'] and r['rows']==8000 for r in records.values())
    for key in ['reference_sha256','normal_contract','expected_physical_sequences']:
        assert records['source_mlp']['identity'][key]==records['update1000']['identity'][key]
    lines=['# Frozen normal degeneracy / orientation audit','',
        'Same fixed40 sequences, same crops and target stencils; no training. Table: controlled heavy occlusion, equal frame/case/sequence mass. Strides1/2 pooled within each frame.','',
        'Degenerate: nonfinite prediction, tangent length <=1e-4 object diameters, or tangent sine <=0.05. Target eligibility is unchanged. Unit angle and sign metrics exclude these predictions; their failure rate is reported separately. Negative dot means angle >90 degrees, not proof of a pure sign reversal. Unoriented angle is acos(abs(dot)); near-opposite means angle >=150 degrees.','',
        '| Model | Target | Degenerate % | Unit angle deg | Negative dot % | Unoriented angle deg | Near opposite % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,r in records.items():
        for source in ['real','proxy']:
            m=r['tables']['heavy_pooled'][source]['both_all']['sequence_mean']
            fields=['degenerate_pct','unit_angle_deg','negative_dot_pct','unoriented_angle_deg','near_opposite_pct']
            lines.append('| '+name+' | '+source+' | '+' | '.join(f'{m[k]:.4f}' for k in fields)+' |')
    lines+=['','## Within-patch vs cross-patch unit angle','',
            '| Model | Target | Within patch deg | Cross patch deg |','|---|---|---:|---:|']
    for name,r in records.items():
        for source in ['real','proxy']:
            d=r['tables']['heavy_pooled'][source]
            lines.append(f"| {name} | {source} | {d['both_within_patch']['sequence_mean']['unit_angle_deg']:.4f} | {d['both_cross_patch']['sequence_mean']['unit_angle_deg']:.4f} |")
    lines += ['', 'Threshold components overlap; do not add their percentages. Detailed source JSON also reports each stencil scale, nonfinite/zero/small/collinear fractions, legacy-normal attenuation, raw counts and pixel-pooled metrics. Histograms below are pooled valid stencils, not equal-sequence averages.','',
              '![Valid unit-normal angle distributions](angle_histograms.png)']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,source in zip(axes,['real','proxy']):
        for name,r in records.items():
            counts=r['tables']['heavy_pooled'][source]['both_all']['counts']
            hist=[100*counts[f'angle_bin_{i}']/counts['valid_unit'] for i in range(12)]
            ax.plot([7.5+15*i for i in range(12)],hist,marker='o',label=name)
        ax.set_title('Heavy occlusion / '+source);ax.set_xlabel('Oriented unit-normal error (degrees)')
        ax.set_ylabel('Valid stencils (%)');ax.set_xlim(0,180);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.root/'angle_histograms.png',dpi=180);fig.savefig(a.root/'angle_histograms.pdf');plt.close(fig)
    receipt=dict(completed=True,physical_sequences=40,rows_per_checkpoint=8000,optimizer_updates=0,
        checkpoints={n:r['identity']['checkpoint_sha256'] for n,r in records.items()},paired_protocol_verified=True,
        normal_contract=records['source_mlp']['identity']['normal_contract'])
    (a.root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
