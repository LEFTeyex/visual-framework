r"""
Transfer learning deal Module.
Consist of the utils for transfer learning.
"""

from copy import copy, deepcopy
from utils.general import delete_list_indices


def get_name_weights_list_len(x, is_model: bool = True, require_print: bool = False):
    """Get model weights list[(name, shape), ...] for transfer learning"""
    name_weights = []
    state = x.state_dict().items() if is_model else x.items()
    for name, weight in state:
        name2weight = (name, tuple(weight.shape))
        name_weights.append(name2weight)
        if require_print:
            print(name2weight)
    return name_weights


def print_pair_name_shape(x, y, x_position: int = 50, y_position: int = 50, x_del=None, y_del=None):
    # get name and shape list
    x = get_name_weights_list_len(x, False)
    y = get_name_weights_list_len(y, False)

    # del no use name_shape pair
    if x_del is not None:
        x = delete_list_indices(x, x_del)
    if y_del is not None:
        y = delete_list_indices(y, y_del)

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
    for x_ns, y_ns in zip(x, y):  # (name, shape)
        xn, xs = x_ns
        yn, ys = y_ns
        print(f'{str(xn):{x_p}}{str(xs):<20}{str(yn):{y_p}}{str(ys):<20}')


def _change_name_of_weights_in_state_dict(state_dict, name_to_change_pair: list):
    # it is inplace for OrderedDict
    name_list = list(state_dict.keys())
    for name in name_list:
        for old, new in name_to_change_pair:
            if old in name:
                # todo maybe apper bug because of repeat old in name that is wrong
                new_name = name.replace(old, new)
                state_dict[new_name] = state_dict.pop(name)
                break


def change_name_for_state_dict_then_check(state_dict, state_dict_to_check):
    # list consist of (old_name_in, new_name_in)
    # the old_name_in must be unique
    name_pair = [
        ('model.0.conv', 'backbone.block1.0.conv'),
        ('model.0.bn', 'backbone.block1.0.bn'),
        ('model.1.conv', 'backbone.block1.1.conv'),
    ]
    _change_name_of_weights_in_state_dict(state_dict, name_pair)
    # for checking whether they are corresponding to each other one by one in (name, shape)
    print_pair_name_shape(state_dict, state_dict_to_check)


def transfer_weights(state_dict, save_path):
    # TODO 2022.3.23
    # TODO load_state_dict and output missing_keys and unexpected_keys to check
    # TODO it should load in model and then model.state_dict() to save
    # TODO first second ... to do explain how to transfer learning by weights
    pass


def shape_correspond_and_change_name_then_check(state_dict, state_dict_to_check):
    pass


'''def transfer():
    # 2022.01.28
    model = Model('models/yolov5s.yaml')

    weights = 'yolov5s.pt'
    checkpoint = torch.load(weights)
    csd = checkpoint['model'].float().state_dict()
    name = ('model.2.', 'model.4.', 'model.6.', 'model.8.', 'model.13.', 'model.17.', 'model.20.', 'model.23.')

    for k in deepcopy(csd):
        for n in name:
            if n in k:
                new_k = list(k)
                new_k.insert(len(n), 'cls_wrapper.')
                new_k = ''.join(new_k)
                csd[new_k] = csd.pop(k)

    model.load_state_dict(csd, strict=False)
    del checkpoint['model']
    checkpoint['model'] = deepcopy(model).state_dict()
    torch.save(checkpoint, 'transfer_yolov5s.pt')'''

if __name__ == '__main__':
    import torch
    from models.yolov5 import yolov5_v6

    path = '../models/yolov5/yolov5x_sd.pt'
    sd = torch.load(path)
    m = yolov5_v6.yolov5x_v6().state_dict()
    # print_pair_name_shape(sd, m, 40, x_del=[-7], y_del=[0, 1, 2])
    transfer_weights(sd, m)
