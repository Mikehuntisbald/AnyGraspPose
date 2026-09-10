"""Independent minimal reader. Split semantics checked against NVlabs/dex-ycb-toolkit.

Official release has 100 sorted sequences per subject. Refuse to infer positions
from an incomplete directory listing. No hand arrays or calibration are accessed.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from lip.geometry.mesh import mesh_metadata

SUBJECTS=['20200709-subject-01','20200813-subject-02','20200820-subject-03','20200903-subject-04',
          '20200908-subject-05','20200918-subject-06','20200928-subject-07','20201002-subject-08',
          '20201015-subject-09','20201022-subject-10']
CAMERAS=['836212060125','839512060362','840412060917','841412060263','932122060857','932122060861','932122061900','932122062010']
CLASSES=['002_master_chef_can','003_cracker_box','004_sugar_box','005_tomato_soup_can','006_mustard_bottle',
         '007_tuna_fish_can','008_pudding_box','009_gelatin_box','010_potted_meat_can','011_banana','019_pitcher_base',
         '021_bleach_cleanser','024_bowl','025_mug','035_power_drill','036_wood_block','037_scissors',
         '040_large_marker','051_large_clamp','052_extra_large_clamp','061_foam_brick']


class CalibrationLoader(yaml.SafeLoader):
    """Official calibration YAML includes tuples; allow this inert type only."""


CalibrationLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:tuple(loader.construct_sequence(node)))


def read_yaml(path):
    return yaml.load(Path(path).read_text(),Loader=CalibrationLoader)


def split_of(subject_index, sequence_index, setup='s0'):
    if setup=='s0':
        return 'train' if sequence_index%5!=4 else ('val' if subject_index<2 else 'test')
    if setup=='s1':return 'train' if subject_index in (0,1,2,3,4,5,9) else ('val' if subject_index==6 else 'test')
    raise ValueError('Only s0 and s1 are supported')


def target(meta, labels, pose_scale=1.):
    j=int(meta['ycb_grasp_ind']); ids=meta['ycb_ids']
    if not 0<=j<len(ids):raise ValueError('Invalid grasp index')
    a=np.asarray(labels['pose_y'][j], dtype='f4')
    if a.shape!=(3,4) or not np.isfinite(a).all():raise ValueError('Invalid pose shape/nonfinite')
    r=a[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=2e-3) or abs(np.linalg.det(r)-1)>2e-3:raise ValueError('Invalid rotation')
    t=np.eye(4,dtype='f4');t[:3]=a;t[:3,3]*=pose_scale
    if t[2,3]<=0:raise ValueError('Target behind camera')
    return int(ids[j]),j,t


def read_frame(root, stream, frame, depth_scale):
    base=Path(root)/stream['relative_dir']
    rgb=cv2.imread(str(base/f'color_{frame:06d}.jpg'))
    depth=cv2.imread(str(base/f'aligned_depth_to_color_{frame:06d}.png'),cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:raise ValueError(f'Unreadable RGB/depth: {base} frame {frame}')
    if rgb.shape[:2]!=(480,640) or depth.shape!=(480,640):raise ValueError('Unexpected RGB/depth resolution')
    if depth.dtype!=np.uint16:raise ValueError('Expected uint16 aligned depth; audit scale before using another release')
    return cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB).transpose(2,0,1).astype('f4')/255, depth[None].astype('f4')*depth_scale


def subject_listings(root, available_subjects=None, verified_train_subset=False, inventory=None):
    root=Path(root);selected=set(available_subjects) if available_subjects else set(SUBJECTS)
    if not selected.issubset(SUBJECTS):raise ValueError('Unknown subject selection')
    if available_subjects and verified_train_subset:raise ValueError('Choose one subset mode')
    listings={};missing=[]
    for subject in SUBJECTS:
        if subject not in selected:listings[subject]=[];continue
        dirs=sorted(p.name for p in (root/subject).glob('*') if p.is_dir() and (p/'meta.yml').exists())
        canonical=inventory['sequences'][subject] if inventory else dirs
        if verified_train_subset:
            known=['20200709_141754','20200709_141841','20200709_141931','20200709_142022']
            if subject==SUBJECTS[0]:
                if any(s not in known for s in dirs):raise ValueError('Unverified partial sequence')
                canonical=known
            else:canonical=[]
        elif len(canonical)!=100 or canonical!=sorted(canonical) or len(set(canonical))!=100:
            missing.append(dict(subject=subject,found=len(dirs)))
        listings[subject]=canonical
    return listings,missing


def _index_camera(job):
    root,out,subject,seq,camera,meta,record,depth_scale,pose_scale=job
    directory=root/subject/seq/camera;poses=[];frames=[];errors=[]
    for fi in range(int(meta['num_frames'])):
        try:
            for name in (f'color_{fi:06d}.jpg',f'aligned_depth_to_color_{fi:06d}.png'):
                if not (directory/name).is_file():raise ValueError('Missing '+name)
            with np.load(directory/f'labels_{fi:06d}.npz',allow_pickle=False) as labels:
                _,_,t=target(meta,labels,pose_scale)
            read_frame(root,record,fi,depth_scale)
            poses.append(t);frames.append(fi)
        except (OSError,ValueError,KeyError,IndexError,cv2.error) as e:
            errors.append(dict(path=str(directory),frame=fi,error=str(e)))
    if not frames:return None,errors
    np.savez(out/record['pose_cache'],frames=np.array(frames),poses=np.stack(poses))
    record=dict(record,num_frames=len(frames))
    record['pose_cache_sha256']=hashlib.sha256((out/record['pose_cache']).read_bytes()).hexdigest()
    return record,errors


def build_index(root, out, setup='s0', depth_scale=.001, pose_scale=1., mesh_scale=1., fps=30., inventory=None,
                verified_train_subset=False, available_subjects=None, workers=8):
    root,out=Path(root).resolve(),Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if out.is_relative_to(root):raise ValueError('Index/cache must be outside raw data')
    audit=dict(errors=[],missing_subjects=[],streams=0,frames=0,source_root=str(root),cache_schema_version=2,
               depth_scale_to_m=depth_scale,pose_scale_to_m=pose_scale,mesh_scale_to_m=mesh_scale,fps=fps,
               timestamp_source='frame_index/fps; official capture rate 30 Hz',setup=setup)
    inventory_data=json.loads(Path(inventory).read_text()) if inventory else None
    listings,missing=subject_listings(root,available_subjects,verified_train_subset,inventory_data)
    audit['missing_subjects']=missing
    if missing:
        (out/'audit.json').write_text(json.dumps(audit,indent=2))
        raise RuntimeError('Official split requires complete sequence inventory (100 per selected subject); see audit.json')
    meshes={};jobs=[];members={x:set() for x in ('train','val','test')}
    for si,subject in enumerate(SUBJECTS):
        for qi,seq in enumerate(listings[subject]):
            path=root/subject/seq
            if not (path/'meta.yml').exists():
                if verified_train_subset:continue
                audit['errors'].append(dict(path=str(path),error='Missing sequence'));continue
            meta=read_yaml(path/'meta.yml');oid=int(meta['ycb_ids'][int(meta['ycb_grasp_ind'])])
            mesh_path=root/'models'/CLASSES[oid-1]/'textured_simple.obj'
            if oid not in meshes:meshes[oid]=mesh_metadata(mesh_path,out/'meshes',mesh_scale)
            mesh,cache_path=meshes[oid];split=split_of(si,qi,setup)
            members[split].add(subject+'/'+seq)
            for camera in CAMERAS:
                intr=read_yaml(root/'calibration'/'intrinsics'/f'{camera}_640x480.yml')['color']
                k=[[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1]]
                stream_id=f'{subject}/{seq}/{camera}';key=hashlib.sha256(stream_id.encode()).hexdigest()[:20]
                record=dict(subject_id=subject,sequence_id=seq,camera_serial=camera,stream_id=stream_id,relative_dir=stream_id,
                            split=split,object_id=oid,object_index_in_sequence=int(meta['ycb_grasp_ind']),intrinsics=k,
                            mesh_path=str(mesh_path.relative_to(root)),mesh_cache=str(Path(cache_path).relative_to(out)),
                            mesh_center=mesh['center'].tolist(),mesh_diameter=float(mesh['diameter']),pose_cache=key+'.npz')
                jobs.append((root,out,subject,seq,camera,meta,record,depth_scale,pose_scale))
    cv2.setNumThreads(1);records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(record,errors) in enumerate(pool.map(_index_camera,jobs)):
            audit['errors'].extend(errors)
            if record:records.append(record);audit['streams']+=1;audit['frames']+=record['num_frames']
            if (i+1)%32==0:print(json.dumps(dict(indexed_streams=i+1,total_streams=len(jobs),frames=audit['frames'],errors=len(audit['errors']))),flush=True)
    assert all(not(members[a]&members[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    audit['split_disjoint']=True;audit['physical_sequences']={k:len(v) for k,v in members.items()}
    audit['complete']=not audit['errors'] and sum(map(len,members.values()))==1000
    audit['verified_train_subset']=verified_train_subset
    audit['verified_subject_subset']=bool(available_subjects)
    audit['selected_subjects']=[s for s in SUBJECTS if listings[s]]
    audit['object_ids']=sorted(meshes)
    audit['inventory_scope']=('Official README positions 0..3 only' if verified_train_subset else
                              'Complete inventories of selected subjects; official '+setup+' subset' if available_subjects else 'full official inventory')
    blob=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records);(out/'streams.jsonl').write_text(blob)
    audit['split_hash']=hashlib.sha256(blob.encode()).hexdigest()
    audit['mesh_hash']=hashlib.sha256(''.join(str(meshes[k][0]['cache_key']) for k in sorted(meshes)).encode()).hexdigest()
    (out/'audit.json').write_text(json.dumps(audit,indent=2));(out/'sequence_inventory.json').write_text(json.dumps(dict(sequences=listings),indent=2))
    return audit
