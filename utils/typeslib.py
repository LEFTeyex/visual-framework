r"""Set type hints in this lib but should use my data structure to avoid circular import error"""

from os import PathLike
from torch import Tensor
from numpy import ndarray
from typing import Union, Optional, TypeVar, Type, Tuple

from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler
from torch.nn import Module, Parameter
from torch.optim.lr_scheduler import _LRScheduler  # ignore _LRScheduler, no problem

__all__ = [
    # strpath
    'strpath',
    # python
    'int_or_None', 'str_or_None', 'float_or_None', 'complex_or_None',
    'set_or_None', 'list_or_None', 'dict_or_None', 'tuple_or_None',
    'int_or_tuple',
    'instance_',
    # pytorch
    'Tensor_or_None', 'int_or_Tensor', 'Tensor_or_ndarray',
    'module_', 'optimizer_', 'lr_scheduler_', 'gradscaler_', 'parameter_', 'dataset_',
    'module_or_parameter', 'module_or_None',
    # special designed
    'pkt_or_None', 'tuple_or_list',
]

# str pathlike
strpath = Union[str, PathLike]

# python
int_or_None = Optional[int]
str_or_None = Optional[str]
float_or_None = Optional[float]
complex_or_None = Optional[complex]
set_or_None = Optional[set]
list_or_None = Optional[list]
dict_or_None = Optional[dict]
tuple_or_None = Optional[tuple]
int_or_tuple = Union[int, tuple]
tuple_or_list = Union[tuple, list]

# python instance
instance_ = TypeVar('instance_', bound=object)

# pytorch
Tensor_or_None = Optional[Tensor]
int_or_Tensor = Union[int, Tensor]
Tensor_or_ndarray = Union[Tensor, ndarray]

# pytorch instance
module_ = TypeVar('module_', bound=Module)
dataset_ = TypeVar('dataset_', bound=Dataset)
optimizer_ = TypeVar('optimizer_', bound=Optimizer)
parameter_ = TypeVar('parameter_', bound=Parameter)
gradscaler_ = TypeVar('gradscaler_', bound=GradScaler)
lr_scheduler_ = TypeVar('lr_scheduler_', bound=_LRScheduler)
module_or_None = Optional[module_]
module_or_parameter = Union[module_, parameter_]

# special designed
_param_kind = Tuple[str, Type[module_or_parameter], dict]
pkt_or_None = Optional[Tuple[_param_kind, ...]]
