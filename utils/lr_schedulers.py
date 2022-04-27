r"""
Lr_schedulers Modules.
Consist of some easy call lr_scheduler by hyp(hyperparameter).
"""

import torch.optim.lr_scheduler as lr_sche

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # ignore it


class WarmUpWithScheduler(object):
    # TODO maybe should add warmup_momentum
    r"""
    Add warmup before lr_scheduler.
    Called behind lr_scheduler.

    Args:
        optimizer: Optimizer = Wrapped optimizer.
        lr_scheduler: _LRScheduler = Wrapped lr_scheduler.
        warmup_steps: int = The number of iterations for warmup.
        warmup_start_lr: list or float = The start learning rate of warmup.
        len_loader: int = The length of dataloader.
        warmup_mode: str = Consist of 'linear'.

    Example:
        '>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)                                                   '
        '>>> lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)                        '
        '>>> data_loader = torch.utils.data.DataLoader(...)                                                            '
        '>>> warmup_scheduler = WarmUpWithScheduler(optimizer, lr_scheduler,                                           '
        '>>>                                        warmup_steps=64, warmup_start_lr=0.01, len_loader=len(data_loader))'
        '>>> for epoch in range(10):                                                                                   '
        '>>>     for batch in data_loader:                                                                             '
        '>>>         train(...)                                                                                        '
        '>>>         validate(...)                                                                                     '
        '>>>         warmup_scheduler.step()                                                                           '
    """

    def __init__(self, optimizer, lr_scheduler, warmup_steps: int, warmup_start_lr,
                 len_loader: int = 1, warmup_mode: str = 'linear'):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        # Attach lr_scheduler
        if not isinstance(lr_scheduler, _LRScheduler):
            raise TypeError(f'{type(lr_scheduler).__name__} is not a _LRScheduler')
        self.lr_scheduler = lr_scheduler

        # check whether attribute initial_lr in optimizer.param_group
        for idx, group in enumerate(optimizer.param_groups):
            if 'initial_lr' not in group:
                raise KeyError("param 'initial_lr' is not specified "
                               f"in param_groups[{idx}] when resuming an optimizer")
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

        self.len_loader = len_loader
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode

        if isinstance(warmup_start_lr, list):
            assert len(warmup_start_lr) == len(self.base_lrs), \
                f'The length of warmup_start_lr {len(warmup_start_lr)} ' \
                f'and optimizer.param_group {len(self.base_lrs)} do not correspond'
            self.warmup_start_lrs = warmup_start_lr

        else:
            self.warmup_start_lrs = [warmup_start_lr] * len(self.base_lrs)

        self.last_step = -1
        self.last_epoch = -1
        self._warmup_done = False
        self._last_lr = None
        self._step_count = 0

        self.step()

    def state_dict(self):
        r"""
        It contains an entry for every variable in self.__dict__
        which is not one of the ('optimizer', 'lr_scheduler').

        Returns:
            the state of the scheduler as a dict.
        """
        return {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}

    def load_state_dict(self, state_dict):
        r"""
        Loads the schedulers state.

        Args:
            state_dict: dict = scheduler state. Should be an object returned from a call to state_dict.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        r"""
        Return last computed learning rate by current warmup scheduler.
        """
        return self._last_lr

    def get_warmup_lr(self):
        r"""Return warmup learning rate to upgrade"""
        if self.warmup_mode == 'linear':
            return [warmup_lr + (base_lr - warmup_lr) * (self.last_step / self.warmup_steps)
                    for warmup_lr, base_lr in zip(self.warmup_start_lrs, self.base_lrs)]
        else:
            raise ValueError(f'New the other warmup_mode is not implemented')

    @property
    def _new_epoch(self):
        r"""Return whether is a new epoch started now"""
        return self.last_step % self.len_loader == 0

    def _step(self):
        r"""For warmup and lr_scheduler step once"""
        if self._warmup_done and self._new_epoch:
            self.lr_scheduler.step()

        elif not self._warmup_done and self.last_step <= self.warmup_steps:
            values = self.get_warmup_lr()

            if self.last_step >= self.warmup_steps:
                self._warmup_done = True

            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr

    def step(self, step=None, epoch=None):
        self._step_count += 1

        if step is None and epoch is None:
            self.last_step += 1
            if self._new_epoch:
                self.last_epoch += 1
            self._step()

        elif step is not None and epoch is None:
            self.last_step = step
            self.last_epoch = step // self.len_loader
            self._step()

        elif step is None and epoch is not None:
            self.last_step = epoch * self.len_loader
            self.last_epoch = epoch
            self._step()

        else:  # if step and epoch
            # step is relative to epoch only here
            self.last_step = step + epoch * self.len_loader
            self.last_epoch = epoch
            self._step()

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def _test_warmup_scheduler():
    import torch
    import torch.optim as optim

    params = [torch.tensor(1., requires_grad=True)] * 2
    optimizer = optim.SGD(params, 0.1)
    lr_scheduler = lr_sche.ConstantLR(optimizer, 1, 100)
    warmup_scheduler = WarmUpWithScheduler(optimizer, lr_scheduler, 16, 0.01, 32)
    for epoch in range(10):
        for _ in range(32):
            print([p['lr'] for p in optimizer.param_groups])
            warmup_scheduler.step()


if __name__ == '__main__':
    _test_warmup_scheduler()
