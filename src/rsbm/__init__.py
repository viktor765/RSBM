import math

from . import bridges, normalizations, reflectors, sdes


def create_sde_and_bridge(
    sde_conf, normalization: normalizations.Normalization
) -> tuple[sdes.SDE, bridges.BrownianBridge]:
    """The reference SDE and its bridge; reflected ones reflect into normalization's box."""
    common_sde_args = {
        "sigma": sde_conf.sigma,
        "euler_steps": sde_conf.euler_steps,
        "noise_on_last_step": sde_conf.noise_on_last_step,
    }
    if sde_conf.type == "brownian":
        sde = sdes.SDE(**common_sde_args)
        bridge = bridges.BrownianBridge(sigma=sde_conf.sigma)
    elif sde_conf.type == "reflected":
        n_reflections = sde_conf.n_reflections
        if isinstance(n_reflections, str) and n_reflections.lower() == "auto":
            if not isinstance(normalization, normalizations.AffineBox):
                raise ValueError("sde.n_reflections=auto requires an AffineBox.")
            widths = [
                upper - lower
                for lower, upper in zip(normalization.lower, normalization.upper)
            ]
            if not widths or not all(math.isfinite(width) for width in widths):
                raise ValueError("sde.n_reflections=auto requires finite box widths.")
            n_reflections = reflectors.recommended_n_reflections(
                sde_conf.sigma / min(widths)
            )
        reflector = reflectors.Reflector(
            n_reflections=n_reflections, normalization=normalization
        )
        sde = sdes.ReflectedSDE(
            reflector=reflector,
            project_last_step=sde_conf.project_last_step,
            **common_sde_args,
        )
        bridge = bridges.ReflectedBrownianBridge(
            sigma=sde_conf.sigma, reflector=reflector
        )
    else:
        raise ValueError(f"Unknown SDE type: {sde_conf.type}")
    return sde, bridge
