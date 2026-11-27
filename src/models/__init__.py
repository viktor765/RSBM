import torch.nn as nn
from .unet import UNet
from .dunet import DirectionalUNet
from .litenet import LiteNet, DirectionalLiteNet
from .multinet import MultiNet
from .control_drift import ControlDrift
from .toy_policy import ToyPolicyNet
from .utils import create_eval_copy, update_ema
from .export import (
    WEIGHTS,
    final_model_file_name,
    load_final_model,
    read_final_model_metadata,
    save_final_model,
)


def create_model(conf, sigma: float) -> nn.Module:
    """Build the model, whose output is the SDE drift for diffusion sigma.

    With model.output = drift each network outputs the drift. With control it
    outputs the control Z, and the drift is sigma * Z; the model stays a
    MultiNet of one ControlDrift per direction.
    """
    model = _create_networks(conf)
    if conf.output == 'drift':
        return model
    if conf.output != 'control':
        raise ValueError(f'Unknown model output: {conf.output}')
    if not isinstance(model, MultiNet):
        raise ValueError(
            'model.output=control needs one network per direction '
            '(model.directional: false).'
        )
    if not sigma > 0:
        raise ValueError('model.output=control needs a positive sde.sigma.')
    return MultiNet({
        direction: ControlDrift(network, sigma)
        for direction, network in model.nets.items()
    })


def _create_networks(conf) -> nn.Module:
    if conf.type.lower() == 'unet':
        if conf.directional:
            return DirectionalUNet(**conf.args)
        else:
            return MultiNet({
                +1: UNet(**conf.args),
                -1: UNet(**conf.args),
            })
    elif conf.type.lower() == 'litenet':
        if conf.directional:
            return DirectionalLiteNet(conf.args.input_dim,
                                              conf.args.time_embed_dim,
                                              conf.args.dir_embed_dim)
        else:
            return MultiNet({
                +1: LiteNet(conf.args.input_dim, conf.args.time_embed_dim),
                -1: LiteNet(conf.args.input_dim, conf.args.time_embed_dim),
            })
    elif conf.type.lower() == 'toy_policy':
        if conf.directional:
            raise ValueError('toy_policy uses one network per direction.')
        return MultiNet({
            +1: ToyPolicyNet(**conf.args),
            -1: ToyPolicyNet(**conf.args),
        })
    else:
        raise ValueError(f'Unknown model type: {conf.type}')
