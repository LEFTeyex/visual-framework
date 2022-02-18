r"""Set type hints in this lib"""

from os import PathLike
from typing import Union, Optional, TypeVar, Type, \
    Tuple

from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # ignore _LRScheduler, no problem
from torch.utils.data import Dataset

__all__ = ['_strpath', '_dataset_c',
           '_int_or_None', '_str_or_None', '_float_or_None', '_complex_or_None',
           '_set_or_None', '_list_or_None', '_dict_or_None', '_tuple_or_None',
           '_module', '_optimizer', '_lr_scheduler', '_parameter',
           '_module_or_parameter', '_module_or_None',
           # special design
           '_pkt_or_None']

# path type hints
_strpath = Union[str, PathLike[str]]

# Union[python built_in type, None]
_int_or_None = Optional[int]
_str_or_None = Optional[str]
_float_or_None = Optional[float]
_complex_or_None = Optional[complex]

_set_or_None = Optional[set]
_list_or_None = Optional[list]
_dict_or_None = Optional[dict]
_tuple_or_None = Optional[tuple]

# for instance type hints in pytorch
_module = TypeVar('_module', bound=Module)
_optimizer = TypeVar('_optimizer', bound=Optimizer)
_lr_scheduler = TypeVar('_lr_scheduler', bound=_LRScheduler)
_parameter = TypeVar('_parameter', bound=Parameter)
_dataset = TypeVar('_dataset', bound=Dataset)

# for class
_dataset_c = Type[_dataset]

_module_or_parameter = Union[_module, _parameter]

# Union[pytorch built_in type, None]
_module_or_None = Optional[_module]

# special design for self.set_param_groups
_param_kind = Tuple[str, Type[_module_or_parameter], dict]
_param_kind_tuple_any = Tuple[_param_kind, ...]
_pkt_or_None = Optional[_param_kind_tuple_any]
