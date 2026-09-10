import argparse,json
from lip.data.sampling import merge_sampling
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--length',type=int,default=8);p.add_argument('--world',type=int,default=8);p.add_argument('--test-finalized',action='store_true');a=p.parse_args()
print(json.dumps(merge_sampling(a.index,a.length,a.world,a.test_finalized),indent=2))
