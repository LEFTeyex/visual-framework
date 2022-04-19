r"""
For validating model.
Consist of some Valers.
"""

from pathlib import Path

from metaclass.metavaler import MetaValClassify

__all__ = ['ValClassify']

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


class ValClassify(MetaValClassify):
    def __init__(self, args=None,
                 last=True, model=None, writer=None,
                 half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None, visual_image=None):
        super(ValClassify, self).__init__(last, model, writer, half, dataloader,
                                          loss_fn, cls_names, epoch, visual_image, args)
