import math
import torch

from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['EMAFunction', 'update_bn']


class EMAFunction(object):
    r"""Use in AveragedModel avg_fn=EMAFunction()"""

    def __init__(self, decay=0.9999, tau=2000):
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # to more

    @torch.no_grad()
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):
        r"""d=[0.0, 0.393] corresponds to num_averaged=[0, 1000]"""
        d = self.decay(num_averaged)
        return averaged_model_parameter * d + (1 - d) * model_parameter


@torch.no_grad()
def update_bn(loader, model, device=None, norm=True):
    r"""
    Updates BatchNorm running_mean, running_var buffers in the model.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for x in loader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        if device is not None:
            if norm:
                x = x.to(device) / 255
            else:
                x = x.to(device)

        model(x)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
