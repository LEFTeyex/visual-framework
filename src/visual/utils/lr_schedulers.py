r"""
Lr_scheduler Modules.
Consist of some easy call lr_scheduler by hyp(hyperparameter)
and some Lambda function for LambdaLR.
"""

from torch.optim import lr_scheduler

from .typeslib import optimizer_, dict_or_None

__all__ = ['select_lr_scheduler']


def select_lr_scheduler(optimizer_instance: optimizer_,
                        lr_scheduler_name: str,
                        kwargs: dict,
                        lr_lambda_dict: dict_or_None = None):
    # need _lr_lambda_dict to add in the future
    r"""
    Select lr_scheduler.
    Args:
        optimizer_instance: optimizer_ = the instance of optimizer.
        lr_scheduler_name: str = lr_scheduler name corresponding to lr_scheduler module in pytorch.
        kwargs: dict = the hyperparameter corresponding to the lr_scheduler.
        lr_lambda_dict: dict_or_None = the dict of lambda function for LambdaLR or MultiplicativeLR.

    Returns:
        lr_scheduler instance
    """
    _lr_lambda_dict = {}
    if lr_lambda_dict is not None:
        _lr_lambda_dict = lr_lambda_dict

    if lr_scheduler_name in ('LambdaLR', 'MultiplicativeLR'):
        lr_lambda_names = kwargs['lr_lambda'] if isinstance(kwargs['lr_lambda'], (list, tuple)) \
            else [kwargs['lr_lambda']]
        kwargs['lr_lambda'] = [_lr_lambda_dict[x] for x in lr_lambda_names]

    lr_scheduler_class = eval(f'lr_scheduler.{lr_scheduler_name}')
    return lr_scheduler_class(optimizer_instance, **kwargs)
