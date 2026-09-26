"""Reduce already completed fixed40 trials; no training or fresh data access."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(root):
    rows={}
    for rank in range(8):
        p=root/f'rank{rank}';manifest=json.loads((p/'manifest.json').read_text())
        assert manifest['completed']
        assert hashlib.file_digest((p/'frames.jsonl').open('rb'),'sha256').hexdigest()==manifest['frames_sha256']
        with (p/'frames.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['history']!='off' or r['phase']!='occlusion' or not r['case'].startswith('heavy'):continue
                key=(r['physical_sequence'],r['case'],r['relative_frame'])
                assert key not in rows
                rows[key]={name:r[name] for name in ('geometry_focus_real','geometry_canonical_real','geometry_canonical_proxy')}
    assert len({key[0] for key in rows})==40
    return rows


def reduce(rows):
    grouped={}
    for (sequence,case,frame),regions in rows.items():
        for region,values in regions.items():
            if values is None:continue
            for metric,value in values.items():
                grouped.setdefault(region,{}).setdefault(metric,{}).setdefault(sequence,{}).setdefault(case,[]).append(value)
    return {region:{metric:statistics.mean(statistics.mean(statistics.mean(v) for v in cases.values()) for cases in sequences.values())
                    for metric,sequences in metrics.items()} for region,metrics in grouped.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    data={arm:read(a.root/arm) for arm in ('source','corrupted','clean')}
    for arm,rows in data.items():
        assert rows.keys()==data['source'].keys()
        for key,regions in rows.items():
            for name,r in regions.items():
                other=data['source'][key][name]
                assert (r is None)==(other is None)
                if r is not None:assert r['pixels']==other['pixels']
    metrics={arm:reduce(rows) for arm,rows in data.items()}
    result=dict(completed=True,physical_sequences=40,paired_target_pixels=True,heavy_records_per_model=len(data['source']),
                metrics=metrics,reduction='frames within case, equal heavy-case mass within sequence, equal sequence mass',
                conclusion='No generalized heavy-geometry improvement; do not promote or extend V40 solely on short probe',
                history_disabled=True,goal_complete=False)
    (a.root/'paired_outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V40 fixed40 outcome','','|Checkpoint|Real sensor XYZ mm|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---:|---:|---:|---:|---:|']
    for arm,m in metrics.items():
        r,c,p=m['geometry_focus_real'],m['geometry_canonical_real'],m['geometry_canonical_proxy']
        lines.append(f"|{arm}|{r['xyz_mm']:.3f}|{c['xyz_mm']:.3f}|{r['depth_mm']:.3f}|{p['xyz_mm']:.3f}|{p['depth_mm']:.3f}|")
    lines+=['','Both terminal models completed100 updates. No geometry improvement on this full protocol; source remains the reference. These are reconstruction metrics under shared baseline-conditioned crops, not native pose scores. History is disabled.','',
            'The separate base-pose audit is descriptive: canonical real XYZ is7.94mm in the <=15deg bin and76.45mm above45deg. Populations differ. Geodesic angles are not symmetry reduced; this does not by itself prove unavailable CAD surface is the cause.']
    (a.root/'PAIRED_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
