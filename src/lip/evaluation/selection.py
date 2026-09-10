import numpy as np


def fixed_balanced_subset(items,limit,key,seed=42):
    """Predeclared object-balanced quick diagnostic; independent of errors/GT quality."""
    if limit is None or limit>=len(items):return list(items)
    rng=np.random.default_rng(seed);groups={}
    for item in items:groups.setdefault(key(item),[]).append(item)
    for values in groups.values():rng.shuffle(values)
    result=[]
    while len(result)<limit:
        for group in sorted(groups):
            if groups[group]:result.append(groups[group].pop())
            if len(result)==limit:break
    return result
