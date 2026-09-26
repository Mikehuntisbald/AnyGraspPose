"""Audit the frozen representation and distinguish prior mixing from learning."""
import argparse,json,statistics
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    checkpoints={k:torch.load(root/'runs/seed42'/f'{k}.pt',map_location='cpu',weights_only=False) for k in ('initial','last')}
    initial,final=checkpoints['initial'],checkpoints['last']
    assert final['step']==100
    frozen=[n for n in initial['model'] if not n.startswith('cad_transport.')]
    assert all(torch.equal(initial['model'][n],final['model'][n]) for n in frozen)
    assert all(n.startswith('cad_transport.') for g in final['optimizer']['param_groups'] for n in g['names'])
    parent=torch.load(final['config']['geometry_transport_training']['source_checkpoint'],map_location='cpu',weights_only=False)
    changed=[]
    for name,tensor in initial['model'].items():
        if not torch.equal(tensor,parent['model'][name]):changed.append(name)
    if final['config']['cad_transport'].get('reference_conditioned'):
        assert changed==['cad_transport.head.0.weight']
        old=parent['model'][changed[0]];new=initial['model'][changed[0]]
        assert torch.equal(old,new[:,:16]) and not new[:,16:].any()
        assert final['model'][changed[0]][:,16:].norm()>0
    else:
        assert changed==['cad_transport.head.2.bias']
        before=parent['model'][changed[0]];after=initial['model'][changed[0]]
        assert torch.equal(before[:6],after[:6]) and after[6]==0
    assert json.loads((root/'runs/seed42/resume2.json').read_text())['complete_state_verified']
    table={};data={}
    for step in (0,100):
        rows=[]
        for rank in range(8):
            folder=root/'probe'/f'step{step}'/f'rank{rank}'
            receipt=json.loads((folder/'receipt.json').read_text());assert receipt['completed'] and receipt['lookup_audit']
            rows += [json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        data[step]={r['seed']:r for r in rows};table[str(step)]={}
        for kind in ('real','proxy','real_zero_flow','proxy_zero_flow'):
            values={}
            for row in rows:
                v=row['metrics'][kind]
                if not row['heavy'] or v is None:continue
                physical='/'.join(row['stream'].split('|')[0].split('/')[:2])
                for k,x in v.items():
                    if isinstance(x,(int,float)):values.setdefault(k,{}).setdefault(physical,[]).append(x)
            table[str(step)][kind]={k:statistics.mean(statistics.mean(x) for x in seqs.values()) for k,seqs in values.items()}
    assert data[0].keys()==data[100].keys()
    for seed,row in data[0].items():
        other=data[100][seed]
        assert all(row[k]==other[k] for k in ('seed','stream','heavy','natural','window'))
        for kind in ('real','proxy'):
            x,y=row['metrics'][kind],other['metrics'][kind]
            assert (x is None)==(y is None)
            if x:assert x['pixels']==y['pixels']
    flow_pass=all(table['100'][k]['flow_epe'] < .9*table['100'][k]['zero_flow_epe'] for k in ('real','proxy'))
    geometry_pass=all(table['100'][k][metric] < table['0'][k][metric] for k in ('real','proxy') for metric in ('xyz_mm','depth_mm'))
    report=dict(completed=True,step=100,paired_records=len(data[0]),reduction='equal physical-sequence mass; heavy records within sequence',
                frozen_model_tensors_verified=len(frozen),initialization_changes=changed,gate_bias_override=0.,
                strict_resume_verified=True,only_transport_optimizer=True,metrics=table,flow_gate_passed=flow_pass,
                geometry_gate_passed=geometry_pass,goal_complete=False,default_model_changed=False,automatic_continuation=False)
    (root/'outcome.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Frozen-backbone correspondence decoder trial','','|Step|Real XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---:|---:|---:|---:|']
    for step in ('0','100'):
        real,proxy=table[step]['real'],table[step]['proxy']
        lines.append(f"|{step}|{real['xyz_mm']:.3f}|{real['depth_mm']:.3f}|{proxy['xyz_mm']:.3f}|{proxy['depth_mm']:.3f}|")
    lines += ['',f'Flow gate passed: {flow_pass}; geometry gate passed: {geometry_pass}. No automatic continuation.',
              f'All {len(frozen)} non-transport model tensors, including encoder/DPT/EMA/pose, remain exact.',
              'Step0 already includes the stronger CAD mixing prior; its improvement over V34 is not a training gain.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
