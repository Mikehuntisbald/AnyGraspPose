"""Measure complete 8-GPU training steps at an unchanged effective batch size."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import yaml


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--init-from', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--trials', default='2:4:1,4:2:2,8:1:2,8:1:4')
    args = parser.parse_args()
    if not 8 <= args.steps <= 350:
        parser.error('Use 8 to 350 bounded steps')
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    args.out.mkdir(parents=True, exist_ok=True)
    original = yaml.safe_load(Path(args.config).read_text())
    assert original['world_size'] == 8
    effective = original['effective_sequences_per_step']
    reports = []
    for trial in args.trials.split(','):
        batch, accum, workers = map(int, trial.split(':'))
        if batch * accum * 8 != effective:
            raise ValueError('This tuning must preserve the effective batch')
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError('GPU jobs already present; no trial started: ' + occupied)
        name = f'b{batch}_a{accum}_w{workers}'
        folder = args.out / name
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir()
        c = dict(original, batch_sequences_per_gpu=batch, grad_accum_steps=accum,
                 num_workers=workers, prefetch_factor=2 if workers > 1 else 1)
        cfg = folder / 'config.yaml'
        cfg.write_text(yaml.safe_dump(c, sort_keys=False))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=8', '-m', 'lip.train_stream', '--config', str(cfg),
                   '--init-from', args.init_from, '--data-root', args.data_root,
                   '--output', str(folder / 'train'), '--max-steps', str(args.steps), '--preflight']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                   PYTHONPATH=str(project / 'src'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        sensor = []
        with (folder / 'console.log').open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            started = time.time()
            while proc.poll() is None:
                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                                '--format=csv,noheader,nounits'], text=True)
                step = 0
                progress = folder / 'train/rank0.jsonl'
                if progress.exists():
                    lines = progress.read_text().splitlines()
                    if lines:
                        try:
                            step = json.loads(lines[-1])['new_stage_step']
                        except json.JSONDecodeError:
                            pass
                sensor.append(dict(time=time.time(), step=step, gpus=[list(map(int, row.split(',')))
                                                                    for row in gpu.strip().splitlines()]))
                time.sleep(1)
        (folder / 'gpu_samples.json').write_text(json.dumps(sensor, indent=2))
        report = dict(name=name, command=command, returncode=proc.returncode,
                      wall_seconds=time.time()-started, config=c)
        if proc.returncode == 0:
            ranks = [list(map(json.loads, (folder / f'train/rank{rank}.jsonl').read_text().splitlines()))
                     for rank in range(8)]
            assert all(len(rows) == args.steps for rows in ranks)
            assert all(r['actual_supervised_frames_global'] == original['nominal_supervised_updates_per_step']
                       and r['effective_sequences'] == effective and r['kv_has_training_graph']
                       for rows in ranks for r in rows)
            # Slowest rank controls completion of each synchronized optimizer step.
            seconds = [max(rows[i]['seconds'] for rows in ranks) for i in range(4, args.steps)]
            used = [sample for sample in sensor if sample['step'] >= 4]
            report.update(mean_step_seconds=statistics.mean(seconds),
                          supervised_frames_per_second=original['nominal_supervised_updates_per_step']/statistics.mean(seconds),
                          peak_allocated_bytes=max(r['peak_allocated'] for rows in ranks for r in rows),
                          gpu_utilization_mean=statistics.mean(g[1] for sample in used for g in sample['gpus']) if used else None,
                          losses=[statistics.mean(rows[i]['loss'] for rows in ranks) for i in range(args.steps)])
        reports.append(report)
        (args.out / 'trials.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ('command','config','losses')}), flush=True)
        if proc.returncode:
            raise RuntimeError('Bounded trial failed; see ' + str(folder / 'console.log'))
        # torchrun has exited, so all its distributed workers have joined.
    best = min(reports, key=lambda r:r['mean_step_seconds'])
    summary = dict(completed=True, effective_sequences_per_step=effective,
                   unchanged_source=True, selected=best['name'],
                   baseline_seconds=reports[0]['mean_step_seconds'],
                   selected_seconds=best['mean_step_seconds'],
                   speedup=reports[0]['mean_step_seconds']/best['mean_step_seconds'],
                   trials=reports)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='trials'}), flush=True)


if __name__ == '__main__':
    main()
