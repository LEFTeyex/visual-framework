r"""
For validating model.
Consist of some Valers.
"""

from utils import MetaValDetect

__all__ = ['ValDetect']


class ValDetect(MetaValDetect):
    def __init__(self, args=None,
                 last=True, model=None, writer=None,
                 half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None):
        super(ValDetect, self).__init__(last=last, writer=writer, model=model)
        self.set_self_parameters_training(model, half, loss_fn, dataloader, cls_names, epoch)


if __name__ == '__main__':
    pass
