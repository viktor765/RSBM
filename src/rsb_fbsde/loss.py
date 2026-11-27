# SPDX-License-Identifier: Apache-2.0
# Adapted from rSB-FBSDE/loss.py in WayneDW/Reflected_Schrodinger_Bridge.
# Modified for rsbm: optional exact divergence, the paper's boundary local-time
# term and matched running and boundary quadrature.
import torch


def alternate_loss(
    policy,
    states,
    times,
    importance_controls,
    weights,
    displacements,
    boundary_points,
    boundary_times,
    boundary_weights,
    sigma,
    dt,
    trace_probes=None,
):
    """Algorithm 1 of Deng et al. (2024) for one policy, from cached path points.

    The running integral is a weighted sum over path states, each with its
    policy time and sampling control. The boundary integral is a sum over
    Euler intervals. For each interval the sampler supplies boundary points
    q_e and applied vectors d_e = n_e dL_e: one projected point/vector, or
    separate edge crossings/vectors. Their sum is twice the projection
    correction for reflection, once for projection. Each boundary point is
    evaluated at its interval's quadrature times s_j, with weights v_j.
    For constant diffusion sigma, the loss is

        sum_i w_i [Z·(Z/2 + Z_impt) + div(sigma Z)](x_i, t_i) dt
        + (2 / sigma) sum_k sum_j v_kj sum_e <Z(q_ke, s_kj), d_ke>,

    the bulk integrand and the paper's local-time term <Z, n> dL / sigma with
    weight 2 instead of the paper's 1: only 2 makes the continuous objective
    minimal at the true score (see the paper's appendix). This objective weight is separate
    from the reflection factor 2 already in d. dL gets no extra dt factor.

    Without ``trace_probes`` the divergence is exact, one VJP per dimension.
    With them it is upstream's Hutchinson estimate, one probe per state: with
    identity covariance, E[εᵀ ∂(sigma Z)/∂x ε] = div(sigma Z). Either way the
    VJPs keep create_graph=True: the parameter gradient needs mixed second
    derivatives. The result is a sum over the given paths; the caller scales
    it to the mean per-path loss. Importance controls are detached, as upstream.
    """
    # Give projection records a singleton edge axis, then broadcast each edge
    # over its interval's times.
    if boundary_points.ndim == 2:
        boundary_points = boundary_points[:, None, :]
        displacements = displacements[:, None, :]
    boundary_shape = (*boundary_times.shape, boundary_points.shape[1], states.shape[-1])
    boundary_points = boundary_points[:, None, :, :].expand(boundary_shape)
    displacements = displacements[:, None, :, :].expand(boundary_shape)
    boundary_times = boundary_times[:, :, None].expand(boundary_shape[:-1]).reshape(-1)
    boundary_weights = (
        boundary_weights[:, :, None].expand(boundary_shape[:-1]).reshape(-1)
    )
    boundary_points = boundary_points.reshape(-1, states.shape[-1])
    displacements = displacements.reshape(-1, states.shape[-1])
    z = policy(states, times)
    if trace_probes is None:
        # States are independent, so the gradient of a column sum is per state.
        divergence = sum(
            torch.autograd.grad(sigma * z[:, i].sum(), states, create_graph=True)[0][
                :, i
            ]
            for i in range(z.shape[-1])
        )
    else:
        vjp = torch.autograd.grad(sigma * z, states, trace_probes, create_graph=True)[0]
        divergence = (vjp * trace_probes).sum(-1)
    dt_integrand = (z * (0.5 * z + importance_controls.detach())).sum(-1) + divergence
    loss = (weights * dt_integrand).sum() * dt
    # Interior proposals need no boundary policy evaluation, even an empty one.
    reflected = displacements.ne(0).any(dim=-1)
    if reflected.any():
        z_boundary = policy(boundary_points[reflected], boundary_times[reflected])
        dL_integrand = (z_boundary * displacements[reflected]).sum(-1)
        loss = loss + 2 * (boundary_weights[reflected] * dL_integrand).sum() / sigma
    return loss
