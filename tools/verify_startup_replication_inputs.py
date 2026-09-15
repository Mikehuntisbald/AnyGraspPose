"""Audit seeded model construction and replay every fixed train sampling draw."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.data.stream_clips import StreamClips
from lip.engine.stream_checkpoint import source_hash, sha
from lip.engine.stream_config import load_stream_config, make_model
from prepare_startup_factorial import ARMS


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--experiment', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    folders = [a.original, *a.experiment]
    experiments = [json.loads((f / 'experiment.json').read_text()) for f in folders]
    original = experiments[0]
    assert json.loads((a.original / 'status.json').read_text())['phase'] == 'completed'
    assert [e['seed'] for e in experiments] == [42, 1000003, 2000003]
    assert all(e['source_sha256'] == source_hash() for e in experiments)
    assert all(e[k] == original[k] for e in experiments for k in
               ('parent_sha256', 'split_hash', 'mesh_hash', 'initial_poses_sha256', 'factors', 'steps'))
    assert sha(original['parent']) == original['parent_sha256']
    parent = torch.load(original['parent'], map_location='cpu', weights_only=False)['model']
    configs = {arm: load_stream_config(original['arms'][arm]['config']) for arm in ARMS}
    ds = StreamClips(original['data_root'], original['index_root'], 8, 48)
    records = []; hidden = []; tuples = []; child_seeds = []
    for number, (folder, e) in enumerate(zip(folders, experiments)):
        raw = Path(e['training_manifest']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e['training_manifest_sha256']
        samples = json.loads(raw); assert len(samples) == 64000
        sampling = e.get('randomness_contract', {}).get('sampling_receipt')
        counter_start = 320000 if number == 0 else sampling['sample_counter_range'][0]
        if number:
            contract = e['randomness_contract']
            assert contract['model_initialization_seed'] == contract['dataloader_generator_seed'] == e['seed']
            assert contract['training_rank_seeds'] == [e['seed'], e['seed'] + 1]
            assert sampling['seed'] == e['seed'] and sampling['sample_counter_range'] == [0, 63999]
            assert sampling['manifest_sha256'] == e['training_manifest_sha256']
        ds.seed = e['seed']; ds.start_sample = counter_start
        for i, item in enumerate(samples):
            assert ds.choose(i) == item, (e['seed'], i, 'sampling replay mismatch')
        initial = None
        for arm in ARMS:
            cfg = load_stream_config(e['arms'][arm]['config'])
            assert cfg['seed'] == e['seed']
            ignored = {'seed', 'preflight_receipt', 'fixed_sampling_manifest_sha256'}
            assert {k:v for k,v in cfg.items() if k not in ignored} == {
                k:v for k,v in configs[arm].items() if k not in ignored}, (e['seed'], arm)
            assert sha(e['arms'][arm]['init']) == e['arms'][arm]['init_sha256']
            state = torch.load(e['arms'][arm]['init'], map_location='cpu', weights_only=False)['model']
            assert all(torch.equal(t, state[k]) for k, t in parent.items())
            if initial is None:
                initial = state
                torch.manual_seed(e['seed']); model = make_model(cfg)
                model.load_state_dict(parent, strict=False)
                assert all(torch.equal(t, state[k]) for k, t in model.state_dict().items())
                hidden.append(state['pose_reference_feedback.readout.0.weight'].clone())
                assert not state['pose_reference_feedback.readout.2.weight'].count_nonzero()
                assert not state['pose_reference_feedback.readout.2.bias'].count_nonzero()
                del model
            else:
                assert initial.keys() == state.keys()
                assert all(torch.equal(t, state[k]) for k, t in initial.items())
            del state
        records.append(dict(seed=e['seed'], counter_range=[counter_start, counter_start+63999],
                            actual_sampler_rng_range=[e['seed']+counter_start, e['seed']+counter_start+63999],
                            manifest_sha256=e['training_manifest_sha256'], replayed_draws=64000,
                            original_parent_tensors_unchanged=True, seeded_reconstruction_exact=True,
                            four_arms_identical_initial_tensors=True))
        tuples.append({(s['stream'],s['start'],s['seed']) for s in samples})
        child_seeds.append({s['seed'] for s in samples})
        del initial
    pairs = []
    for i in range(3):
        for j in range(i+1,3):
            left = records[i]['actual_sampler_rng_range']; right = records[j]['actual_sampler_rng_range']
            assert left[1] < right[0] or right[1] < left[0]
            assert not torch.equal(hidden[i], hidden[j])
            pairs.append(dict(seeds=[records[i]['seed'],records[j]['seed']],
                              exact_sample_tuple_overlap=len(tuples[i]&tuples[j]),
                              incidental_child_seed_overlap=len(child_seeds[i]&child_seeds[j]),
                              new_hidden_weights_differ=True))
    result = dict(completed=True, source_sha256=source_hash(), seeds=records, pairs=pairs,
                  scope='Shared frozen ancestor; all train sampling draws replayed without reading raw images. Independent adaptation seeds, not independent ancestors or disjoint image populations.')
    a.out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
