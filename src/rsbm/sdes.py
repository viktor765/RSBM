from typing import NamedTuple

import torch as th

from .reflectors import Reflector


class EulerTrajectory(NamedTuple):
    """States and interval data; boundary fields are None without a boundary."""

    states: th.Tensor  # (batch, steps + 1, ...): start, then each post-step state
    drifts: th.Tensor  # (batch, steps, ...): model drift at each pre-step state
    displacements: th.Tensor | None  # Applied correction; crossing adds an edge axis.
    boundary_points: th.Tensor | None  # Projected proposals or per-edge crossings.


def reflected_step(
    proposal,
    lower,
    upper,
    *,
    project=False,
    previous=None,
    boundary_evaluation="projection",
):
    """One checked mirror step, or projection, with boundary data in native units.

    The correction is n dL: twice the overshoot for reflection, once for
    projection. A mirror step must return to the box without another reflection
    in the same coordinate. Crossing evaluation needs the previous vector state
    and returns (batch, edges, dimensions) boundary points and correction vectors.
    Each edge retains only its own coordinate of the applied correction.
    """
    if boundary_evaluation not in {"projection", "crossing"}:
        raise ValueError("boundary_evaluation must be projection or crossing.")
    if boundary_evaluation == "crossing":
        if previous is None or proposal.ndim != 2 or previous.shape != proposal.shape:
            raise ValueError(
                "Crossing evaluation requires matching previous vector states."
            )
        if (~th.isfinite(previous) | (previous < lower) | (previous > upper)).any():
            raise ValueError(
                "Crossing evaluation requires previous states inside the finite box."
            )
    boundary = proposal.clamp(min=lower, max=upper)
    displacement = (boundary - proposal) * (1 if project else 2)
    state = boundary if project else proposal + displacement
    invalid = (
        ~th.isfinite(proposal) | ~th.isfinite(state) | (state < lower) | (state > upper)
    )
    if project and boundary_evaluation == "crossing":
        # Crossing locations use the auxiliary mirror path even on a projected
        # final step, so that path must obey the same reflection restriction.
        mirror = proposal + 2 * (boundary - proposal)
        invalid = invalid | ~th.isfinite(mirror) | (mirror < lower) | (mirror > upper)
    if invalid.any():
        if not th.isfinite(proposal).all() or not th.isfinite(state).all():
            raise ValueError(
                "Euler proposals and post-step states must be finite; "
                "check the model drift and diffusion."
            )
        raise ValueError(
            "A single Euler reflection must return a point inside the box; "
            "increase sde.euler_steps to reduce the step size."
        )
    if boundary_evaluation == "crossing":
        active = displacement.ne(0)
        increment = proposal - previous
        # Inactive coordinates get s=1 without dividing by zero. Axis -2
        # selects the crossed edge; axis -1 selects a spatial coordinate.
        fractions = th.where(
            active,
            (boundary - previous)
            / th.where(active, increment, th.ones_like(increment)),
            th.ones_like(increment),
        )
        points = previous[:, None, :] + fractions[:, :, None] * increment[:, None, :]
        already_crossed = active[:, None, :] & (
            fractions[:, None, :] <= fractions[:, :, None]
        )
        points = th.where(already_crossed, 2 * boundary[:, None, :] - points, points)
        # Pin each normal coordinate to its edge and remove roundoff outside
        # the box. Inactive records have zero correction and are never evaluated.
        edge_axes = th.eye(proposal.shape[-1], dtype=th.bool, device=proposal.device)
        points = th.where(edge_axes | ~active[:, :, None], boundary[:, None, :], points)
        boundary = points.clamp(min=lower, max=upper)
        displacement = th.diag_embed(displacement)
    return state, displacement, boundary


class SDE:
    def __init__(self, sigma: float, euler_steps: int, noise_on_last_step: bool = True):
        if not (euler_steps >= 1 and float(euler_steps).is_integer()):
            raise ValueError("sde.euler_steps must be a positive integer.")
        if not 0 <= sigma < float("inf"):
            raise ValueError("sde.sigma must be finite and nonnegative.")
        self.sigma = sigma
        self.euler_steps = int(euler_steps)
        self.noise_on_last_step = bool(noise_on_last_step)

    @th.no_grad()
    def forward_euler(
        self,
        model,
        x0: th.Tensor,
        direction: int | th.Tensor,
        generator: th.Generator | None = None,
        return_trajectory: bool = False,
        boundary_evaluation: str = "projection",
    ):
        """Return final states, or a trajectory with projection/crossing records.

        A (batch,) direction tensor runs each sample in its own direction; time
        is direction-relative, so both directions share the Euler clock.
        Boundary evaluation only selects recorded loss data; sampling keeps the
        same Euler proposals and endpoints. Crossing records have an extra edge
        axis and are available for vector states of a reflected SDE.
        """
        if boundary_evaluation not in {"projection", "crossing"}:
            raise ValueError("boundary_evaluation must be projection or crossing.")
        self._validate_initial_state(x0)
        x = x0
        trajectory = [x] if return_trajectory else None
        drifts, displacements, boundary_points = [], [], []
        ts = th.linspace(0, 1, self.euler_steps + 1, device=x0.device, dtype=x0.dtype)
        for i in range(self.euler_steps):
            t = ts[i].repeat(x.shape[0])
            dt = ts[i + 1] - ts[i]
            drift = model(x, t, direction=direction).to(x.dtype)
            proposal = x + drift * dt
            is_last_step = i == self.euler_steps - 1
            if self.noise_on_last_step or not is_last_step:
                noise = th.randn(
                    x.shape, dtype=x.dtype, device=x.device, generator=generator
                )
                proposal = proposal + self.sigma * noise * th.sqrt(dt)
            x, displacement, boundary = self._post_step(
                proposal,
                args={
                    "is_last_step": is_last_step,
                    "previous_state": x,
                    "boundary_evaluation": (
                        boundary_evaluation if return_trajectory else "projection"
                    ),
                },
            )
            if return_trajectory:
                drifts.append(drift)
                if displacement is not None:
                    displacements.append(displacement)
                    boundary_points.append(boundary)
                trajectory.append(x)
        if return_trajectory:
            return EulerTrajectory(
                th.stack(trajectory, dim=1),
                th.stack(drifts, dim=1),
                th.stack(displacements, dim=1) if displacements else None,
                th.stack(boundary_points, dim=1) if boundary_points else None,
            )
        return x

    def _validate_initial_state(self, x: th.Tensor):
        if not th.isfinite(x).all():
            raise ValueError("Euler trajectories must start at finite points.")

    def _post_step(self, x: th.Tensor, args: dict | None = None):
        if not th.isfinite(x).all():
            raise ValueError(
                "Euler proposals must be finite; check the model drift and diffusion."
            )
        return x, None, None


class ReflectedSDE(SDE):
    def __init__(
        self,
        sigma: float,
        euler_steps: int,
        reflector: Reflector,
        noise_on_last_step: bool = True,
        project_last_step: bool = False,
    ):
        super().__init__(sigma, euler_steps, noise_on_last_step=noise_on_last_step)
        self.reflector = reflector
        self.project_last_step = bool(project_last_step)

    def _validate_initial_state(self, x: th.Tensor):
        lower, upper = self._bounds(x)
        if (~th.isfinite(x) | (x < lower) | (x > upper)).any():
            raise ValueError(
                "Reflected Euler trajectories must start inside the finite box."
            )

    def _bounds(self, x: th.Tensor):
        # A coordinate vector also handles flattened boxes and image layouts.
        unit_lower = x.new_zeros((x.shape[-1],))
        normalization = self.reflector.normalization
        return normalization.inv(unit_lower), normalization.inv(th.ones_like(unit_lower))

    def _post_step(self, x: th.Tensor, args: dict | None = None):
        lower, upper = self._bounds(x)
        project = (
            self.project_last_step
            and args is not None
            and args.get("is_last_step", False)
        )
        return reflected_step(
            x,
            lower,
            upper,
            project=project,
            previous=args.get("previous_state") if args else None,
            boundary_evaluation=(
                args.get("boundary_evaluation", "projection") if args else "projection"
            ),
        )
