r"""
Lr_schedulers Modules.
Consist of some easy call lr_scheduler by hyp(hyperparameter)
and some Lambda function for LambdaLR.
"""

import torch.optim.lr_scheduler as lr_scheduler

from utils.typeslib import optimizer_

__all__ = ['select_lr_scheduler']


def select_lr_scheduler(optimizer_instance: optimizer_, hyp: dict):
    r"""
    Select lr_scheduler.
    Args:
        optimizer_instance: optimizer_ = the instance of optimizer.
        hyp: dict = hyperparameter dict.

    Returns:
        lr_scheduler instance
    """

    _lr_lambda_dict = {}  # TODO need to add in the feature

    lr_scheduler_name = hyp['lr_scheduler']
    kwargs = hyp['lr_scheduler_kwargs']

    if lr_scheduler_name in ('LambdaLR', 'MultiplicativeLR'):
        lr_lambda_names = kwargs['lr_lambda'] if isinstance(kwargs['lr_lambda'], (list, tuple)) \
            else [kwargs['lr_lambda']]
        kwargs['lr_lambda'] = [_lr_lambda_dict[x] for x in lr_lambda_names]

    lr_scheduler_class = eval(f'lr_scheduler.{lr_scheduler_name}')
    return lr_scheduler_class(optimizer_instance, **kwargs, verbose=hyp['verbose'])
