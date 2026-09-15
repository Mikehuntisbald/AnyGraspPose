"""Fit a proposal-conditioned reliability head; audit before any integration."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve,average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.models.update_quality import UpdateQualityHead
from lip.engine.stream_checkpoint import sha,source_hash


def load_cache(root):
    report=json.loads((root/'collection.json').read_text());parts={k:[] for k in ('observation','delta','harmful','advantage')};partitions=[];physical=[];count=0
    failed=0
    for rank in range(report.get('world_size',8)):
        folder=root/f'rank{rank}';done=json.loads((folder/'completed.json').read_text());assert done['completed'] and done['teacher_sha256']==report['teacher_sha256']
        assert not done.get('smoke',False),'A smoke subset cannot be used as full training data'
        failed+=done.get('unusable_frames',0)
        for row in map(json.loads,(folder/'progress.jsonl').read_text().splitlines()):
            if 'file' not in row:continue
            path=folder/row['file'];assert sha(path)==row['sha256'];data=torch.load(path,map_location='cpu',weights_only=False)
            for key in parts:parts[key].append(data[key])
            partitions+=data['partition'];physical+=data['physical_sequence'];count+=len(data['observation'])
    expected=report['expected_observations'] if 'expected_observations' in report else sum(report['clips'].values())*report['frames_per_clip']
    assert count+failed==expected
    report['unusable_frames']=failed
    tensors={k:torch.cat(v) for k,v in parts.items()};partition=np.array(partitions);physical=np.array(physical)
    for name,allowed in report['physical_sequences'].items():assert set(physical[partition==name])<=set(allowed)
    return report,tensors,partition


@torch.no_grad()
def evaluate(model,data,index,original,prevalence):
    logits=[];advantages=[]
    for ids in index.split(1024):
        b=len(ids);obs=data['observation'][ids].float();delta=data['delta'][ids];c=delta.shape[1]
        out=model(obs[:,None].expand(-1,c,-1).flatten(0,1),delta.flatten(0,1))
        logits.append(out['harm_logit'].reshape(b,c).cpu());advantages.append(out['advantage'].reshape(b,c).cpu())
    p=torch.sigmoid(torch.cat(logits)).numpy();pred=torch.cat(advantages).numpy();truth=data['harmful'][index].cpu().numpy();adv=data['advantage'][index].cpu().numpy()
    y=truth[:,original];prob=p[:,original];fpr,tpr,threshold=roc_curve(y,prob)
    eligible=fpr<=.05;recall=float(tpr[eligible].max())
    pair_correct=[]
    for i in range(adv.shape[1]):
        for j in range(i+1,adv.shape[1]):
            valid=np.abs(adv[:,i]-adv[:,j])>.005
            d=pred[valid,i]-pred[valid,j];target=adv[valid,i]-adv[valid,j]
            pair_correct.extend(np.where(d==0,.5,(d*target>0).astype(float)).tolist())
    result=dict(observations=len(index),original_harm_prevalence=float(y.mean()),original_harm_auc=float(roc_auc_score(y,prob)),
        original_harm_ap=float(average_precision_score(y,prob)),original_brier=float(np.mean((prob-y)**2)),constant_brier=float(np.mean((prevalence-y)**2)),
        recall_at_fpr05=recall,candidate_pair_accuracy=float(np.mean(pair_correct)),candidate_pairs=len(pair_correct),
        original_advantage_mae=float(np.mean(np.abs(pred[:,original]-adv[:,original]))))
    result['gate_passed']=bool(result['original_harm_auc']>=.75 and result['original_brier']<.9*result['constant_brier'] and result['recall_at_fpr05']>=.20 and result['candidate_pair_accuracy']>=.65)
    return result,p,pred


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--harm-mode',choices=('all','original','weighted'),default='all');p.add_argument('--candidate-weight',type=float,default=.25);a=p.parse_args()
    if a.candidate_weight<0:raise ValueError('Candidate weight must be nonnegative')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);torch.cuda.set_device(0)
    report,data,partition=load_cache(a.cache);indices={p:torch.tensor(np.flatnonzero(partition==p),device='cuda') for p in ('train','calibration','audit')}
    data={k:v.cuda() for k,v in data.items()};original=report['original_candidate'];train=indices['train'];model=UpdateQualityHead().cuda()
    with torch.no_grad():
        obs=data['observation'][train].float();model.observation_mean.copy_(obs.mean(0));model.observation_std.copy_(obs.std(0).clamp_min(.01));del obs
        model.action_std.copy_(data['delta'][train].flatten(0,1).std(0).clamp_min(.001))
    prevalence=float(data['harmful'][train,original].float().mean());optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.steps,eta_min=.00005);best=float('inf');record=[]
    receipt=dict(teacher_sha256=report['teacher_sha256'],source_sha256=source_hash(),cache_manifest_sha256=sha(a.cache/'collection.json'),seed=42,max_steps=a.steps,
        selection='Minimum original-proposal Brier on calibration; audit partition evaluated once after selecting the checkpoint',
        gate=dict(original_harm_auc_min=.75,brier_ratio_max=.9,recall_at_fpr05_min=.20,candidate_pair_accuracy_min=.65),main_model_integration=False,
        harm_training_mode=a.harm_mode,candidate_harm_weight=a.candidate_weight if a.harm_mode=='weighted' else None,
        harm_consumer_scope='Original actor proposal; other-candidate harm probabilities uncalibrated' if a.harm_mode=='original' else 'Proposal-conditioned harm; original actor proposal is the calibration consumer',
        supervision='GT-derived ADD-S/d advantage and harmful update labels; no FP, no hand labels; all samples from official train split')
    (a.out/'experiment.json').write_text(json.dumps(receipt,indent=2))
    with (a.out/'training.jsonl').open('w') as log:
        for step in range(1,a.steps+1):
            model.train();ids=train[torch.randint(len(train),(1024,),device='cuda')];action=torch.randint(data['delta'].shape[1],(len(ids),),device='cuda')
            out=model(data['observation'][ids],data['delta'][ids,action]);harm=data['harmful'][ids,action].float();adv=data['advantage'][ids,action]
            candidate_bce=torch.nn.functional.binary_cross_entropy_with_logits(out['harm_logit'],harm)
            if a.harm_mode=='all':bce=candidate_bce
            else:
                original_out=model(data['observation'][ids],data['delta'][ids,original])
                bce=torch.nn.functional.binary_cross_entropy_with_logits(original_out['harm_logit'],data['harmful'][ids,original].float())
                if a.harm_mode=='weighted':bce=bce+a.candidate_weight*candidate_bce
            regression=torch.nn.functional.smooth_l1_loss(out['advantage']/.05,adv.clamp(-.2,.2)/.05)
            loss=bce+.5*regression;optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
            row=dict(step=step,loss=float(loss),bce=float(bce),regression=float(regression));log.write(json.dumps(row)+'\n');log.flush()
            if step%100==0 or step==a.steps:
                model.eval();metrics,_,_=evaluate(model,data,indices['calibration'],original,prevalence);record.append(dict(step=step,**metrics))
                (a.out/'calibration_history.json').write_text(json.dumps(record,indent=2))
                if metrics['original_brier']<best:
                    best=metrics['original_brier'];torch.save(dict(model=model.state_dict(),step=step,experiment=receipt,calibration=metrics),a.out/'best.pt')
                print(json.dumps(dict(step=step,calibration=metrics)),flush=True)
    best=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best['model']);model.eval()
    audit,predictions,advantages=evaluate(model,data,indices['audit'],original,prevalence)
    np.savez_compressed(a.out/'audit_predictions.npz',harm_probability=predictions,predicted_advantage=advantages,
        harmful=data['harmful'][indices['audit']].cpu().numpy(),advantage=data['advantage'][indices['audit']].cpu().numpy())
    result=dict(completed=True,selected_step=best['step'],checkpoint_sha256=sha(a.out/'best.pt'),calibration=best['calibration'],audit=audit,
        integration_eligible=bool(best['calibration']['gate_passed'] and audit['gate_passed'] and report['unusable_frames']==0),main_model_integration=False,
        unusable_frames=report['unusable_frames'],audit_previously_inspected=report.get('audit_previously_inspected',False),
        scope='Quality-head development audit on physically disjoint partitions inside official train; this is not a tracking accuracy result or an untouched external test')
    (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
