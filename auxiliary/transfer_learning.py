r"""
Transfer learning deal Module.
Consist of the utils for transfer learning.
"""


def print_name_model_weights(model):
    """Use Excel to deal with the weights for transfer learning"""
    name_weights = []
    for key in model.state_dict():
        name_weights.append(key)
        print(key)
    return name_weights


def print_name_sd_weights(state_dict):
    """Use Excel to deal with the weights for transfer learning"""
    name_weights = []
    for key in state_dict:
        name_weights.append(key)
        print(key)
    return name_weights


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
    pass
