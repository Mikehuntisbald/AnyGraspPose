"""Wait for terminal experiments, verify every arm, plot and archive the evidence."""
import argparse,hashlib,io,json,sys,tarfile,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from analyze_spatial_alignment_factorial import ARMS,CONTRASTS,write_csv


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--wait',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];r=root/'runs/factorial';out=root/'runs/completion';out.mkdir(exist_ok=False)
    state=dict(phase='waiting',started=time.time())
    def save():
        state['updated']=time.time();tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
    try:
        save()
        while True:
            status=json.loads((r/'status.json').read_text())
            if status['phase']=='failed':raise RuntimeError(status.get('error'))
            if status['phase']=='completed':break
            if not a.wait:raise RuntimeError('Experiment is not complete')
            time.sleep(20);save()
        state['phase']='verifying';save();e=json.loads((r/'experiment.json').read_text());assert source_hash()==e['source_sha256']
        analysis=json.loads((r/'analysis/report.json').read_text());assert analysis['completed'] and not analysis['candidate_promoted']
        tests={};cpu_ids=set()
        for name in ('tests','extended_tests','cuda_tests'):
            tree=ET.parse(root/'runs'/(name+'.xml'));suites=list(tree.getroot().iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
            assert counts['errors']==counts['failures']==0;tests[name]=counts
            if name!='cuda_tests':cpu_ids|={(c.get('classname'),c.get('name')) for c in tree.getroot().iter('testcase') if c.find('skipped') is None}
        assert len(cpu_ids)==65
        equivalent=json.loads((r/'equivalence.json').read_text());assert equivalent['passed'] and equivalent['source_sha256']==source_hash()
        initial_states=[];training={};timing={}
        for name in ARMS:
            receipt=json.loads((r/name/'training_receipt.json').read_text());assert receipt['completed'] and receipt['steps']==1000
            assert receipt['checkpoint_sha256']==sha(r/name/'train/last.pt')==analysis['checkpoints'][name]
            training[name]=receipt
            m=json.loads((r/'evaluation'/name/'scored/manifest.json').read_text())
            assert m['frames']==23200 and len(m['streams'])==320 and m['missing_pose_frames']==578
            assert m['fp_calls']==m['gt_pose_reads']==m['gt_mask_reads']==m['hand_annotation_reads']==m['gt_resets']==0
            assert sha(r/'evaluation'/name/'scored/predictions.jsonl')==m['predictions_sha256']
        for name in ('residual',)+ARMS:
            t=json.loads((r/'benchmark'/name/'receipt.json').read_text());assert t['completed'] and t['metrics']['steady_after8']['frames']==480
            timing[name]=t['metrics']['steady_after8']
        assert json.loads((r/'evaluation/parent_reproduction.json').read_text())['poses_and_scores_exact']
        plots=out/'figures';plots.mkdir();colors=['#777777','#2277aa','#aa7722','#229966','#9955aa','#dd6655']
        methods=list(analysis['checkpoints']);fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        for ax,pop,metric,title in zip(axes,('all','visibility_lt_03','bad_initial_first8'),('adds_005','adds_005','rotation_deg'),
            ('All: strict ADD-S (%)','Visibility < 0.3: strict ADD-S (%)','Bad init first 8: rotation (deg)')):
            values=[analysis['populations'][pop][metric]['values'][n] for n in methods]
            ax.bar(methods,values,color=colors);ax.tick_params(axis='x',rotation=45);ax.set_title(title)
            for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
        fig.savefig(plots/'accuracy.png',dpi=170);fig.savefig(plots/'accuracy.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        effects=analysis['populations']['bad_initial_first8']['rotation_deg']['effects'];labels=list(CONTRASTS)
        for ax,ci,title in zip(axes,('ci95','ci99'),('Unadjusted 95% CI','99% CI: primary five-contrast family')):
            point=np.array([effects[n]['delta'] for n in labels]);lo=np.array([effects[n][ci][0] for n in labels]);hi=np.array([effects[n][ci][1] for n in labels])
            ax.errorbar(point,np.arange(5),xerr=np.stack([point-lo,hi-point]),fmt='o',capsize=3);ax.axvline(0,color='gray');ax.set(yticks=np.arange(5),yticklabels=labels,xlabel='Rotation error difference (deg); negative is favorable',title=title)
        fig.savefig(plots/'factorial_intervals.png',dpi=170);fig.savefig(plots/'factorial_intervals.pdf');plt.close(fig)
        curves=[];fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for name,color in zip(ARMS,colors[1:5]):
            rank=[list(map(json.loads,(r/name/f'train/rank{k}.jsonl').read_text().splitlines())) for k in range(2)]
            for begin in range(0,1000,25):
                rows=[row for rr in rank for row in rr[begin:begin+25]]
                metrics=np.array([v['metrics'] for v in rows])
                row=dict(method=name,step=begin+25,pose_loss=float((metrics[:,0]+.5*metrics[:,1]+metrics[:,2]).mean()),
                    ce=float(np.mean([v['alignment_ce'] for v in rows])) if 'alignment_ce' in rows[0] else None,
                    top1=float(np.mean([v['alignment_top1'] for v in rows])) if 'alignment_top1' in rows[0] else None,
                    visible_keys_per_microbatch=float(np.mean([v['alignment_visible_keys'] for v in rows])) if 'alignment_visible_keys' in rows[0] else None,
                    supported_queries_per_microbatch=float(np.mean([v['alignment_supported_queries'] for v in rows])) if 'alignment_supported_queries' in rows[0] else None,
                    seconds_per_step=float(np.mean([v['seconds'] for v in rows])))
                curves.append(row)
            mine=[v for v in curves if v['method']==name]
            axes[0].plot([v['step'] for v in mine],[v['pose_loss'] for v in mine],label=name,color=color)
            if mine[0]['ce'] is not None:
                axes[1].plot([v['step'] for v in mine],[v['ce'] for v in mine],label=name,color=color)
                axes[2].plot([v['step'] for v in mine],[v['top1'] for v in mine],label=name,color=color)
        for ax,title in zip(axes,('Original pose loss (excludes auxiliary)','Spatial CE (training support)','Attention target hit rate (training support)')):
            ax.set(title=title,xlabel='Stage step');ax.legend(fontsize=8)
        fig.savefig(plots/'training.png',dpi=170);fig.savefig(plots/'training.pdf');plt.close(fig);write_csv(out/'training_windows.csv',curves)
        text='# 显式空间监督 × 父 latent：完整 2×2 结果\n\n四组各完成 1,000 步、64,000 次相同片段采样，及 320 流 / 23,200 帧真实初始化、零 FP native s0 val。以下为最终固定步数，一次训练种子；没有自动晋级或新 official test。\n\n'
        text+='| 方法 | 总体严格 ADD-S (%) | 严重遮挡严格 (%) | 坏初值前八帧旋转 (°) | 启动中心 (mm) |\n|---|---:|---:|---:|---:|\n'
        for n in methods:
            values=[analysis['populations'][p][m]['values'][n] for p,m in [('all','adds_005'),('visibility_lt_03','adds_005'),('bad_initial_first8','rotation_deg'),('bad_initial_first8','center_mm')]]
            text+='| '+n+' | '+' | '.join(f'{v:.6f}' for v in values)+' |\n'
        text+='\nS1：空间监督权重 0.05；L1：新分支读取父 latent。L0 只切断这条直接输入，不移除父模型或 dense 特征已有的几何。\n\n| 启动旋转的预设对比 | 差值 (°) | 95% CI | 99% CI |\n|---|---:|---|---|\n'
        for n in CONTRASTS:
            v=effects[n];text+=f"| {n} | {v['delta']:+.6f} | {v['ci95']} | {v['ci99']} |\n"
        supported=[n for n in CONTRASTS if n!='interaction' and effects[n]['ci99'][1]<0]
        text+='\n按预设五对比族 99% 区间，支持启动旋转降低的条件效应：'+(', '.join(supported) if supported else '无')+'。这不自动代表总体、中心、遮挡和延迟同时改善；interaction 必须结合条件效应解释。\n'
        text+='\n训练 CE/命中率使用各自闭环产生的可见对应点，支持集合会变化；不能拿训练损失直接当作跨模型验证精度。完整 val 才是共同帧上的部署对照。\n\n| 方法 | steady P50 (ms) | P95 (ms) |\n|---|---:|---:|\n'
        for n,t in timing.items():text+=f"| {n} | {t['p50_ms']:.3f} | {t['p95_ms']:.3f} |\n"
        text+='\n65 项不同的 CPU 检查、4 项 CUDA 检查、真实零起点、单卡显存、双卡 DDP 保存恢复均通过。父权重全量预测精确复现。正式相机/mesh/单位不变；对称对象仅排除辅助对应监督，仍参与位姿训练和全部评测。\n\n![精度](figures/accuracy.png)\n\n![配对区间](figures/factorial_intervals.png)\n\n![训练信号](figures/training.png)\n'
        (out/'RESULT.md').write_text(text)
        receipt=dict(completed=True,source_sha256=source_hash(),experiment_sha256=sha(r/'experiment.json'),analysis_sha256=sha(r/'analysis/report.json'),tests=tests,cpu_unique=len(cpu_ids),
            training=training,timing=timing,parent_reproduction_exact=True,candidate_promoted=False,official_test_launched=False)
        (out/'verification.json').write_text(json.dumps(receipt,indent=2));state['phase']='packaging';save()
        files={}
        for sub in ('src','tools','tests','configs'):
            for path in sorted((root/sub).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':files[sub+'/'+str(path.relative_to(root/sub))]=path
        for folder in (r,out,root/'runs/target_audit'):
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix not in ('.pt','.pth','.pyc') and path.name not in ('status.json','status.tmp'):files[str(path.relative_to(root))]=path
        for path in sorted((root/'runs').glob('*')):
            if path.is_file() and path.suffix in ('.log','.xml','.json'):files[str(path.relative_to(root))]=path
        files['runs/factorial/status.json']=r/'status.json';files['pyproject.toml']=root/'pyproject.toml'
        for name in ('parent','training_manifest','train_initializers','val_initializers','models_info'):
            assert sha(e[name])==e[name+'_sha256']
            if name!='parent':files['bound_inputs/'+name+'.json']=Path(e[name])
        for name in ('manifest.json','metrics.json','predictions.jsonl'):
            files['fp_reference/'+name]=Path('/mnt/why/dexycb_lip/fp_val_20260915/runs/full_v2/scored')/name
        archive=root/'runs/completed_evidence.tar.gz';assert not archive.exists();hashes={}
        with tarfile.open(archive,'w:gz') as tar:
            for name,path in files.items():
                raw=path.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest();member=tarfile.TarInfo(name);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
            raw=json.dumps(dict(completed=True,files=hashes,weights='Remain on server, SHA-bound; no candidate promoted'),indent=2).encode()
            member=tarfile.TarInfo('SHA256.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
        archive_receipt=dict(completed=True,archive=str(archive),sha256=sha(archive),files=len(hashes),bytes=archive.stat().st_size)
        (out/'archive.json').write_text(json.dumps(archive_receipt,indent=2));state.update(phase='completed',completed=time.time());save();print(json.dumps(archive_receipt),flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
