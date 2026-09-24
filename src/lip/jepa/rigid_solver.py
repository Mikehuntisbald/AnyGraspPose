"""Independent fixed-observation left-SE(3) LM, with no proposal prior term."""
from dataclasses import dataclass
import torch


def skew(x):
    a, b, c = x.unbind(-1)
    z = torch.zeros_like(a)
    return torch.stack((z, -c, b, c, z, -a, -b, a, z), -1).reshape(*x.shape[:-1], 3, 3)


def se3_exp(delta):
    algebra = delta.new_zeros(4, 4)
    algebra[:3, :3] = skew(delta[3:])
    algebra[:3, 3] = delta[:3]
    return torch.matrix_exp(algebra)


@torch.autocast('cuda', enabled=False)
def project_and_jacobian(pose, points, k):
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    h = camera @ k.T
    z = h[:, 2:]
    uv = h[:, :2] / z
    jp = (k[None, :2] * z[:, :, None] - h[:, :2, None] * k[None, 2:3]) / z[:, :, None].square()
    tangent = torch.cat((torch.eye(3, device=points.device, dtype=points.dtype).expand(len(points), -1, -1), -skew(camera)), -1)
    return uv, jp @ tangent, camera[:, 2]


@dataclass(frozen=True)
class Measurements:
    points_object_m: torch.Tensor
    positions_crop_px: torch.Tensor
    sigma_crop_px: torch.Tensor
    probability: torch.Tensor
    current_raw_valid: torch.Tensor
    completion_only: torch.Tensor


@torch.no_grad()
@torch.autocast('cuda', enabled=False)
def solve_pose(base, measurements, k, diameter, iterations=3, probability_min=.5):
    """Meters, object-to-camera. All measurements stay fixed throughout LM."""
    if not torch.isfinite(base).all() or diameter <= 0:
        raise ValueError('Invalid base pose/diameter')
    dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
    pose, k = base.to(dtype).clone(), k.to(dtype)
    x = measurements.points_object_m.to(dtype)
    uv = measurements.positions_crop_px.to(dtype)
    sigma = measurements.sigma_crop_px.to(dtype)
    probability = measurements.probability.to(dtype)
    valid = measurements.current_raw_valid.bool() & ~measurements.completion_only.bool()
    valid &= (probability >= probability_min) & torch.isfinite(probability)
    valid &= torch.isfinite(x).all(-1) & torch.isfinite(uv).all(-1) & torch.isfinite(sigma) & (sigma > 0)
    valid &= (x @ pose[:3, :3].T + pose[:3, 3])[:, 2] > .001
    report = dict(measurement_valid=False, input_points=len(x), used_points=int(valid.sum()),
        status='insufficient_evidence', information_rank=0, information_condition=None,
        accepted_steps=0, completion_measurements=0, update_convention='left SE3 rho_m phi_rad',
        measurements_fixed=True, proposal_prior_weight=0.)
    if valid.sum() < 6:
        return base.clone(), report
    x, uv, sigma, probability = x[valid], uv[valid], sigma[valid], probability[valid]
    span = uv.amax(0) - uv.amin(0)
    report['image_span_px'] = span.tolist()
    if span.min() < 8:
        report['status'] = 'insufficient_spatial_coverage'
        return base.clone(), report
    scale = pose.new_tensor([diameter] * 3 + [1.] * 3)
    def terms(t):
        prediction, jac, depth = project_and_jacobian(t, x, k)
        residual = (prediction - uv) / sigma[:, None]
        jac = jac / sigma[:, None, None] * scale
        norm = residual.norm(dim=-1)
        robust = (3 / norm.clamp_min(1e-9)).clamp_max(1.)
        weight = probability * robust
        cost = (probability * torch.where(norm <= 3, norm.square(), 6 * norm - 9)).sum()
        h = torch.einsum('nki,n,nkj->ij', jac, weight, jac)
        g = torch.einsum('nki,n,nk->i', jac, weight, residual)
        return cost, h, g, depth
    cost, h, g, depth = terms(pose)
    eigen = torch.linalg.eigvalsh(h)
    rank = int((eigen > eigen.max() * 1e-7).sum())
    condition = float(eigen.max() / eigen.min().clamp_min(1e-20))
    report.update(information_rank=rank, information_condition=condition, initial_objective=float(cost))
    if rank < 6 or condition > 1e7 or not torch.isfinite(h).all():
        report['status'] = 'degenerate_information'
        return base.clone(), report
    trace = [float(cost)]
    damping = 1e-3
    for _ in range(iterations):
        delta = torch.linalg.solve(h + damping * torch.diag(h.diag().clamp_min(1e-5)), -g)
        rotation = delta[3:].norm()
        delta = delta * torch.minimum(.175 / rotation.clamp_min(1e-8), delta.new_tensor(1.))
        candidate = se3_exp(delta * scale) @ pose
        # rho in a left twist includes the translation needed to rotate about a
        # distant object center. Bound actual center motion, not that coupling term.
        center_motion_d = (candidate[:3, 3] - pose[:3, 3]).norm() / diameter
        if center_motion_d > .05:
            delta = delta * (.05 / center_motion_d)
            candidate = se3_exp(delta * scale) @ pose
        nc, nh, ng, nz = terms(candidate)
        if torch.isfinite(nc) and (nz > .001).all() and nc < cost:
            pose, cost, h, g = candidate, nc, nh, ng
            damping = max(1e-8, damping / 3)
            report['accepted_steps'] += 1
            trace.append(float(cost))
        else:
            damping *= 10
    report.update(measurement_valid=True, status='updated' if report['accepted_steps'] else 'base_retained',
                  final_objective=float(cost), objective_trace=trace)
    return pose.to(base.dtype), report
