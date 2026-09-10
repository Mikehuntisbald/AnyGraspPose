import torch


def hat(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)


def exp(v):
    theta = torch.linalg.vector_norm(v, dim=-1)
    k = hat(v)
    a = torch.sinc(theta / torch.pi)
    b = .5 * torch.sinc(theta / (2 * torch.pi)).square()
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def angle(r):
    # atan2 avoids acos' singular derivative at identity. Norm has zero subgradient.
    s = torch.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0],
                     r[..., 1, 0]-r[..., 0, 1]), -1) * .5
    c = (r.diagonal(dim1=-2, dim2=-1).sum(-1)-1) * .5
    return torch.atan2(torch.linalg.vector_norm(s, dim=-1), c)


def log(r):
    # Quaternion extraction using the largest diagonal candidate is stable near pi.
    a, b, c = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    qabs = torch.sqrt(torch.stack((1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c), -1).clamp_min(1e-12))
    cand = torch.stack((
        torch.stack((qabs[..., 0]**2, r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]), -1),
        torch.stack((r[..., 2, 1]-r[..., 1, 2], qabs[..., 1]**2, r[..., 1, 0]+r[..., 0, 1], r[..., 0, 2]+r[..., 2, 0]), -1),
        torch.stack((r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]+r[..., 0, 1], qabs[..., 2]**2, r[..., 2, 1]+r[..., 1, 2]), -1),
        torch.stack((r[..., 1, 0]-r[..., 0, 1], r[..., 2, 0]+r[..., 0, 2], r[..., 2, 1]+r[..., 1, 2], qabs[..., 3]**2), -1)), -2)
    cand = cand / (2*qabs[..., :, None].clamp_min(.1))
    idx = qabs.argmax(-1)[..., None, None].expand(*qabs.shape[:-1], 1, 4)
    q = cand.gather(-2, idx).squeeze(-2)
    q = torch.where(q[..., :1] < 0, -q, q)
    q = torch.nn.functional.normalize(q, dim=-1)
    n = torch.linalg.vector_norm(q[..., 1:], dim=-1)
    half = torch.atan2(n, q[..., 0])
    return q[..., 1:] * (2/torch.sinc(half/torch.pi))[..., None]


def update(base, rotvec, center_norm, diameter):
    out = base.clone()
    out[..., :3, :3] = exp(rotvec) @ base[..., :3, :3]
    out[..., :3, 3] = base[..., :3, 3] + diameter[..., None] * center_norm
    return out


def center_pose(original, center):
    out = original.clone()
    out[..., :3, 3] += (original[..., :3, :3] @ center[..., None]).squeeze(-1)
    return out


def original_pose(centered, center):
    return center_pose(centered, -center)
