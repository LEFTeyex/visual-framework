r"""
Transfer learning deal Module.
Consist of the utils for transfer learning.
How to use the module to transfer weights as follows first, second, last.
"""

import torch

from collections import OrderedDict

from utils.log import LOGGER
from utils.general import delete_list_indices


def get_name_weights_list_len(x, is_model: bool = True, require_print: bool = False):
    r"""
    Get model weights list[(save_name, shape), ...] for transfer learning.
    Args:
        x: = model or state_dict
        is_model: = whether is model
        require_print: = whether to print

    Returns:
        name_weights list
    """
    name_weights = []
    state = x.state_dict().items() if is_model else x.items()
    for name, weight in state:
        # get save_name and weight shape
        name2weight = (name, tuple(weight.shape))
        name_weights.append(name2weight)
        if require_print:
            print(name2weight)
    return name_weights


def print_pair_name_shape(x, y, x_position: int = 50, y_position: int = 50, x_del=None, y_del=None):
    r"""
    Print save_name and shape pair for checking to transfer weights.
    Args:
        x: = state_dict
        y: = state_dict
        x_position: = x print position
        y_position: = y print position
        x_del: = the index list of x for deleting
        y_del: = the index list of y for deleting
    """
    # get save_name and shape list
    x = get_name_weights_list_len(x, False)
    y = get_name_weights_list_len(y, False)

    # del no use name_shape pair
    if x_del is not None:
        delete_list_indices(x, x_del)
    if y_del is not None:
        delete_list_indices(y, y_del)

    # set position for print
    x_p = f'<{x_position}'
    y_p = f'<{y_position}'

    # make them the same length
    len_x = len(x)
    len_y = len(y)
    if len_x > len_y:
        y += [(None, None)] * (len_x - len_y)
    elif len_x < len_y:
        x += [(None, None)] * (len_y - len_x)
    print(f'len: {len(x)} {len(y)}')

    # print pair
    for x_ns, y_ns in zip(x, y):  # (save_name, shape)
        xn, xs = x_ns  # x save_name, x shape
        yn, ys = y_ns
        print(f'{str(xn):{x_p}}{str(xs):<20}{str(yn):{y_p}}{str(ys):<20}')


def _change_name_of_weights_in_state_dict(state_dict, name_to_change_pair: list):
    r"""It is inplace for OrderedDict"""
    name_list = list(state_dict.keys())
    for name in name_list:
        for old, new in name_to_change_pair:
            if old in name:
                # todo maybe apper bug because of repeat old in save_name that is wrong
                new_name = name.replace(old, new)
                state_dict[new_name] = state_dict.pop(name)
                break


def change_then_check(state_dict, name_pairs, state_dict_to_check):
    r"""
    Change save_name replace old with new and then print to check.
    Args:
        state_dict: = state_dict to change
        name_pairs: = the list consist of (old, new)
                      the old_name must be unique
        state_dict_to_check: = state_dict to check
    Returns:
        state_dict dealt
    """
    _change_name_of_weights_in_state_dict(state_dict, name_pairs)
    # for checking whether they are corresponding to each other one by one in (save_name, shape)
    print_pair_name_shape(state_dict, state_dict_to_check)
    return state_dict


def _exchange_name_when_shape_correspond(state_dict, state_dict_to_check, x_del=None, y_del=None):
    r"""It is not inplace"""
    # get new_names and weights then delete unexpected ones
    old_weights = list(state_dict.values())
    if x_del is not None:
        delete_list_indices(old_weights, x_del)
    new_names = list(state_dict_to_check.keys())
    if y_del is not None:
        delete_list_indices(new_names, y_del)

    # create OrderedDict for state_dict then check
    state_dict = OrderedDict(list(zip(new_names, old_weights)))
    return state_dict


def exchange_then_check(state_dict, state_dict_to_check, x_del=None, y_del=None):
    r"""
    Exchange save_name of them when their weights are corresponding one by one.
    Args:
        state_dict: = state_dict need its weights
        state_dict_to_check: = state_dict to check
        x_del: = the index list of x for deleting
        y_del: = the index list of y for deleting
    Returns:
        state_dict dealt
    """
    state_dict = _exchange_name_when_shape_correspond(state_dict, state_dict_to_check, x_del, y_del)
    print_pair_name_shape(state_dict, state_dict_to_check, y_del=y_del)
    return state_dict


def transfer_weights(state_dict, model_instance, save_path=None):
    r"""
    Transfer weights and load state_dict in model to check then save the state_dict.
    Args:
        state_dict: = the state_dict dealt
        model_instance: = model instance to load_state_dict
        save_path: = the path to save
    """
    save_path = '../models/weights/new_sd.pt' if save_path is None else save_path

    rest = model_instance.load_state_dict(state_dict, strict=False)
    LOGGER.warning(f'There are the rest of {len(rest[0])} missing_keys'
                   f' and {len(rest[1])} unexpected_keys when load model')
    num = 30
    if rest[0]:
        print('-' * num)
        print('missing_keys:')
        for missing_key in rest[0]:
            print(missing_key)
        print('-' * num)
    if rest[1]:
        print('-' * num)
        print('unexpected_keys:')
        for unexpected_key in rest[1]:
            print(unexpected_key)
        print('-' * num)
    checkpoint = {'model': model_instance.float()}  # for mixins to load
    torch.save(checkpoint, save_path)


def first_check_name_shape(x, y, x_position: int = 50, y_position: int = 50, x_del=None, y_del=None):
    r"""
    First, check the save_name shape pair and think how to deal them.
    Args:
        x: = state_dict
        y: = state_dict
        x_position: = x print position
        y_position: = y print position
        x_del: = the index list of x for deleting
        y_del: = the index list of y for deleting
    """
    print_pair_name_shape(x, y, x_position, y_position, x_del, y_del)


def second_change_name_of_weights(kind, state_dict, name_pairs, state_dict_to_check,
                                  x_del=None, y_del=None):
    r"""
    Second, change or exchange save_name of weights then check whether the changed state_dict is correct.
    Args:
        kind: = 'change' / 'exchange' the kind to change
        state_dict: = state_dict to change or exchange
        name_pairs: = the list consist of (old, new)
        state_dict_to_check: = state_dict to check
        x_del: = the index list of x for deleting
        y_del: = the index list of y for deleting
    """
    if kind == 'change':
        state_dict = change_then_check(state_dict, name_pairs, state_dict_to_check)
    elif kind == 'exchange':
        state_dict = exchange_then_check(state_dict, state_dict_to_check, x_del, y_del)
    else:
        raise ValueError(f"Unexpected kind {kind}, please input 'change' or 'exchange'")
    return state_dict


def last_transfer_weights_and_save(state_dict, model_instance, save_path=None):
    r"""
    Last, transfer weights and save the state_dict.
    Args:
        state_dict: = the state_dict dealt
        model_instance: = model instance to load_state_dict
        save_path: = the path to save
    """
    transfer_weights(state_dict, model_instance, save_path)


def run():
    r"""
    Run to transfer weights.
    You can only set the parameters below to transfer weights in this module.
    """
    # =========================================================================
    from pathlib import Path
    from models.yolov5 import yolov5_v6

    # the parameters needed to set consist of first, second and last
    first = True
    second = True
    last = True

    # first
    state_dict_path = '../models/yolov5/yolov5x_v6.pt'
    model = yolov5_v6.yolov5x_v6()  # the instance model
    x_position = 50
    y_position = 50
    x_del = None
    y_del = None

    # second
    kind = 'exchange'
    name_pairs = []  # only when kind == 'change'

    # last
    path_save = Path(state_dict_path).parent / 'yolov5x_v6.pt'
    # =========================================================================

    # others
    state_dict_to_transfer = torch.load(state_dict_path)['model'].state_dict()
    model_state_dict = model.state_dict()
    sd = None

    if first and not second:
        first_check_name_shape(state_dict_to_transfer, model_state_dict, x_position, y_position, x_del, y_del)
    if second:
        sd = second_change_name_of_weights(kind, state_dict_to_transfer, name_pairs, model_state_dict,
                                           x_del=x_del, y_del=y_del)
    if second and last:
        last_transfer_weights_and_save(sd, model, path_save)


if __name__ == '__main__':
    run()
