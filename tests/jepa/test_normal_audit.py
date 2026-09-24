import torch
from lip.unified.normal_audit import audit_normals


def test_degenerate_sign_and_unit_angle_separate():
    y,x=torch.meshgrid(torch.arange(28),torch.arange(28),indexing='ij')
    target=torch.stack((x*.005,y*.005,x*0.+.2)).float()[None]
    mask=torch.ones(1,1,28,28,dtype=torch.bool)
    for kind in ('correct','opposite','collapsed','tiny','collinear'):
        prediction=target.clone()
        if kind=='opposite':prediction[:,0]*=-1
        if kind=='collapsed':prediction.zero_()
        if kind=='tiny':prediction*=1e-5
        if kind=='collinear':prediction[:,1]=prediction[:,0]
        result=audit_normals(prediction,target,mask)
        for stride in (1,2):
            a=result[f's{stride}_all'];inside=result[f's{stride}_within_patch'];edge=result[f's{stride}_cross_patch']
            assert a['eligible']==inside['eligible']+edge['eligible']
            if kind in ('correct','opposite'):
                assert a['degenerate']==0 and a['valid_unit']==a['eligible']
                assert a['unit_angle_sum']/a['valid_unit']==(0 if kind=='correct' else 180)
                assert a['negative_dot']==(0 if kind=='correct' else a['valid_unit'])
                assert a['unoriented_angle_sum']==0
            else:
                assert a['degenerate']==a['eligible'] and a['valid_unit']==0
            if kind=='collapsed':assert abs(a['legacy_angle_sum']/a['eligible']-90)<1e-5
