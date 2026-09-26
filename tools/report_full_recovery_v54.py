"""Full paired validation accounting, sequence-balanced metrics and strata."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path


class Means:
    def __init__(self):self.groups={}
    def add(self,key,sequence,stream,value):
        assert math.isfinite(value)
        bucket=self.groups.setdefault(key,{}).setdefault(sequence,{}).setdefault(stream,[0.,0])
        bucket[0]+=value;bucket[1]+=1
    def result(self):
        return {key:dict(mean=statistics.mean(statistics.mean(s/n for s,n in streams.values()) for streams in seq.values()),
                         physical_sequences=len(seq),streams=sum(len(x) for x in seq.values()),records=sum(n for x in seq.values() for _,n in x.values()))
                for key,seq in self.groups.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    means=Means();strata=Means();coverage=collections.Counter();keys=set();seen_streams=set();regions=collections.Counter();refs=[]
    for rank in range(8):
        root=a.root/f'rank{rank}';manifest=json.loads((root/'manifest.json').read_text());refs.append(manifest)
        assert manifest['completed'] and not manifest['smoke']
        assert hashlib.file_digest((root/'frames.jsonl').open('rb'),'sha256').hexdigest()==manifest['frames_sha256']
        assert not (set(manifest['streams'])&seen_streams);seen_streams.update(manifest['streams'])
        for line in (root/'frames.jsonl').open():
            r=json.loads(line);key=(r['stream'],r['frame'],r['case']);assert key not in keys;keys.add(key)
            coverage[r['case']+'/'+r['status']]+=1
            if r['status']!='evaluated':continue
            if r['gt_surface_crop_pixels']==0:coverage[r['case']+'/no_gt_surface_in_crop']+=1
            if r['visible_crop_coverage'] is not None and r['visible_crop_coverage']<.5:coverage[r['case']+'/less_than_half_visible_object_in_crop']+=1
            sequence=r['physical_sequence'];stream=r['stream'];case=r['case']
            for name in ('real','proxy','visible'):
                left,right=(r['metrics'][arm][name] for arm in ('v52','v53'))
                assert (left is None)==(right is None)
                if left:assert left['pixels']==right['pixels'] and left['normal_stencils']==right['normal_stencils']
                regions[case+'/'+name+'/records_with_targets']+=int(left is not None)
                if left:regions[case+'/'+name+'/target_pixels']+=left['pixels']
            angle=r['base_rotation_deg'];rot='le15' if angle<=15 else ('15to45' if angle<=45 else 'gt45')
            v=r['native_visibility'];vis='unknown' if v is None else ('lt20pct' if v<.2 else ('20to50pct' if v<.5 else 'ge50pct'))
            keep=r['remaining_originally_visible_fraction'];ret='unknown' if keep is None else ('le20pct' if keep<=.2 else ('20to50pct' if keep<=.5 else 'gt50pct'))
            for arm,parts in r['metrics'].items():
                for region,metrics in parts.items():
                    if metrics is None:continue
                    for metric,value in metrics.items():
                        if value is None or metric.endswith('pixels') or metric=='normal_stencils':continue
                        key='/'.join((case,arm,region,metric));means.add(key,sequence,stream,value)
                        if metric not in ('canonical_xyz_mm','depth_mm','camera_xyz_mm','camera_normal_deg','identity_reprojection_px','joint_xyz10mm_depth5mm'):continue
                        for group,binname in [('object',str(r['object_id'])),('base_rotation',rot),('natural_visibility',vis),('remaining_visible',ret)]:
                            strata.add('/'.join((group,binname,key)),sequence,stream,value)
    assert len(seen_streams)==320 and len(keys)==23200*3
    assert all(sum(n for k,n in coverage.items() if k.startswith(case+'/') and k.split('/')[1] in ('evaluated','unavailable_initialization','missing_reference_pose'))==23200 for case in ('natural','light','heavy'))
    for m in refs[1:]:
        for k in ('checkpoints','index_sha256','split_hash','mesh_hash','reference_sha256','initializers_sha256','occluder_bank_sha256'):assert m[k]==refs[0][k]
    reduced=means.result();result=dict(completed=True,split='val',streams=320,physical_sequences=40,native_frames=23200,paired_conditions=len(keys),
        coverage=dict(coverage),target_support=dict(regions),metrics=reduced,strata=strata.result(),checkpoints=refs[0]['checkpoints'],
        reduction='Frame means within stream, equal streams within physical sequence, equal physical sequences. Quantiles are framewise then averaged, not pooled pixel percentiles.',
        scope='Full s0 validation population, shared previous-baseline pose/crop, conditional geometry recovery. No GT student pose, no GT input mask, no pose solver, no training or official test.',
        source_receipts=[dict(rank=m['rank'],frames_sha256=m['frames_sha256'],seconds=m['seconds']) for m in refs],default_model_changed=False)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    val=lambda c,arm,region,metric:reduced.get('/'.join((c,arm,region,metric)),{}).get('mean')
    fmt=lambda x:'—' if x is None else f'{x:.3f}'
    lines=['# V54 full validation: V52 versus V53 recovery','','All320 camera streams /40 physical sequences /23200 frames accounted; natural/light25%/heavy80% input conditions,69600 paired conditions. Synthetic fractions refer to the estimated CAD silhouette; actual retained visibility is reported.','',
        'The student uses the same previous sealed LIP pose/crop (PoseCNN at initialization) for both weights. GT supplies scoring only. This is full-population conditional reconstruction, not closed-loop tracking, pose evaluation, or a controlled10-degree initialization experiment.','',
        '|Condition|Evaluated native frames|Unavailable initialization|Missing reference|No GT surface in evaluated crop|','|---|---:|---:|---:|---:|']
    for case in ('natural','light','heavy'):
        lines.append('|'+case+'|'+'|'.join(str(coverage[case+'/'+k]) for k in ('evaluated','unavailable_initialization','missing_reference_pose','no_gt_surface_in_crop'))+'|')
    lines+=['','## Missing-region recovery','','|Condition/target|CAD identity XYZ mm V52→V53|Depth mm V52→V53|Camera XYZ mm V52→V53|Camera normal deg V52→V53|','|---|---:|---:|---:|---:|']
    for case in ('natural','light','heavy'):
        for region in ('real','proxy'):
            lines.append('|'+case+'/'+region+'|'+'|'.join(fmt(val(case,'v52',region,k))+' → '+fmt(val(case,'v53',region,k)) for k in ('canonical_xyz_mm','depth_mm','camera_xyz_mm','camera_normal_deg'))+'|')
    lines+=['','Real means originally visible then synthetically hidden: original sensor depth. Proxy means naturally hidden: rendered CAD depth. Unmasked visible regions are scored separately against sensor depth. No predicted-confidence masks filter the errors. Empty target regions are counted, not assigned zero error. Means are conditional on legal initialization and nonempty targets; coverage above is part of the result.','',
        '## Actual correspondences and valid recovery','','|Condition/target|Identity projection error px V52→V53|All-target valid XYZ<10mm AND depth<5mm fraction V52→V53|','|---|---:|---:|']
    for case in ('natural','light','heavy'):
        for region in ('real','proxy'):
            lines.append('|'+case+'/'+region+'|'+'|'.join(fmt(val(case,'v52',region,k))+' → '+fmt(val(case,'v53',region,k)) for k in ('identity_reprojection_px','joint_xyz10mm_depth5mm'))+'|')
    lines+=['','Per-object, original-visibility, retained-visibility and base-rotation strata, visible-region preservation and validity calibration are in `outcome.json`. Rotation bins use unreduced GT-relative base angles for scoring; their populations differ and cannot establish causality. No default model was promoted.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
