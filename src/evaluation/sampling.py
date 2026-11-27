import torch as th

from .directions import DirectionSpec


def sample_direction(
    *,
    sde,
    model: th.nn.Module,
    source: th.Tensor,
    direction: DirectionSpec,
    device: th.device,
    generator: th.Generator,
    return_trajectory: bool = False,
):
    sampled = sde.forward_euler(
        model,
        source.to(device),
        direction=direction.sampling_direction,
        generator=generator,
        return_trajectory=return_trajectory,
    )
    if return_trajectory:
        return type(sampled)(
            *(value.cpu() if value is not None else None for value in sampled)
        )
    return sampled.cpu()
