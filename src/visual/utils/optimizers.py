r"""
Optimizer Modules.
Consist of some easy call optimizer by hyp(hyperparameter).
"""

from torch import optim

__all__ = ['select_optimizer']


def select_optimizer(params, optimizer_name: str, kwargs: dict):
    r"""
    Select optimizer.
    Args:
        params: = iterable of parameters to optimize or dicts defining parameter groups.
        optimizer_name: str = optimizer name corresponding to optimizer module in pytorch.
        kwargs: dict = the hyperparameter corresponding to the optimizer.

    Returns:
        optimizer instance
    """
    optimizer_class = eval(f'optim.{optimizer_name}')
    return optimizer_class(params, **kwargs)
