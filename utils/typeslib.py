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
    '_path',
    # python
    '_int_or_None', '_str_or_None', '_float_or_None', '_complex_or_None',
    '_set_or_None', '_list_or_None', '_dict_or_None', '_tuple_or_None',
    '_int_or_tuple',
    '_instance', '_instance_c',
    # pytorch
    '_Tensor_or_None', '_int_or_Tensor', '_Tensor_or_ndarray',
    '_module', '_optimizer', '_lr_scheduler', '_gradscaler', '_parameter',
    '_module_or_parameter', '_module_or_None',
    '_dataset_c',
    # special designed
    '_pkt_or_None',
]

# str pathlike
_path = Union[str, PathLike]

# python
_int_or_None = Optional[int]
_str_or_None = Optional[str]
_float_or_None = Optional[float]
_complex_or_None = Optional[complex]
_set_or_None = Optional[set]
_list_or_None = Optional[list]
_dict_or_None = Optional[dict]
_tuple_or_None = Optional[tuple]
_int_or_tuple = Union[int, tuple]

# python instance
_instance = TypeVar('_instance', bound=object)

# python class
_instance_c = Type[_instance]

# pytorch
_Tensor_or_None = Optional[Tensor]
_int_or_Tensor = Union[int, Tensor]
_Tensor_or_ndarray = Union[Tensor, ndarray]

# pytorch instance
_module = TypeVar('_module', bound=Module)
_dataset = TypeVar('_dataset', bound=Dataset)
_optimizer = TypeVar('_optimizer', bound=Optimizer)
_parameter = TypeVar('_parameter', bound=Parameter)
_gradscaler = TypeVar('_gradscaler', bound=GradScaler)
_lr_scheduler = TypeVar('_lr_scheduler', bound=_LRScheduler)
_module_or_None = Optional[_module]
_module_or_parameter = Union[_module, _parameter]

# pytorch class
_dataset_c = Type[_dataset]

# special designed
_param_kind = Tuple[str, Type[_module_or_parameter], dict]
_pkt_or_None = Optional[Tuple[_param_kind, ...]]
