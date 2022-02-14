r"""
Classify Model.
It is built by units.py or torch.nn Module.
"""

import torch.nn as nn

from models.units import Conv
from utils.log import LOGGER

__all__ = ['ModelClassify']


class ModelClassify(nn.Module):
    # TODO: Upgrade for somewhere in the future
    r"""
    Model of Classify which is a custom model.
    """

    def __init__(self, inc: int, num_class: int):
        super(ModelClassify, self).__init__()
        LOGGER.info('Initializing the model...')
        self.conv1 = Conv(inc, num_class, act='relu')
        LOGGER.info('Initialize model successfully')

    def forward(self, x):
        x = self.conv1(x)
        return x
