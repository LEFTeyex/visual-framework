r"""
Tensorboard Modules.
Consist of all utils of writer in tensorboard.
"""

import torch

from tqdm import tqdm

from .bbox import xywh2xyxy

__all__ = ['add_model_graph', 'add_optimizer_lr', 'add_epoch_curve',
           'add_datasets_images_labels_detect', 'add_batch_images_predictions_detect']


@torch.no_grad()
def add_model_graph(writer, model, inc, image_size, epoch=0, verbose=False):
    r"""Add model graph to tensorboard by writer"""
    if epoch == 0:
        device = next(model.parameters()).device
        image = torch.zeros(1, inc, image_size, image_size, device=device)
        # todo args need to change
        writer.add_graph(model, image, verbose=verbose, use_strict_trace=False)


def add_optimizer_lr(writer, optimizer, epoch):
    r"""
    Add lr in optimizer of param_groups.
    """
    for index, param_group in enumerate(optimizer.param_groups):
        name = param_group.get('name', str(index))
        lr = param_group['lr']
        writer.add_scalar(f'train_optimizer_lr/{name}', lr, epoch)


def add_epoch_curve(writer, title, value, value_name, epoch):
    r"""
    Add all train loss with name.
    """
    if epoch != -1:
        for v, v_name in zip(value, value_name):
            writer.add_scalar(f'{title}/{v_name}', v, epoch)


@torch.no_grad()
def add_datasets_images_labels_detect(writer, datasets, title, dfmt='CHW'):
    r"""
    Add batch size images with labels and bboxes.
    """
    space = ' ' * 11
    with tqdm(enumerate(datasets), total=10, bar_format='{l_bar}{bar:20}{r_bar}',
              desc=f'{space}{title}: visualizing images with labels') as pbar:
        for index, (image, label, _, _) in pbar:
            # only consist of 10 images per plot in tensorboard
            if index <= 9:
                _, h, w = image.shape
                wh_tensor = torch.tensor([[w, h, w, h]], device=label.device)
                bboxes = xywh2xyxy(label[:, 2:] * wh_tensor)  # scale bbox really
                classes = [str(int(cls.tolist())) for cls in label[:, 1]]

                writer.add_image_with_boxes(f'{title}_image', image, bboxes, index, dataformats=dfmt, labels=classes)
            else:
                break


@torch.no_grad()
def add_batch_images_predictions_detect(writer, title, bs_index, images, predictions, epoch=-1, dfmt='CHW'):
    r"""
    Add batch size images with labels and bboxes.
    """
    # TODO BUG: the step is not continue because of the max is 10
    if epoch == -1 and bs_index < 3:
        bs = images.shape[0]
        for index in range(bs):
            # only consist of 10 images per plot in tensorboard
            if index <= 9:
                pred = predictions[index]
                if pred.shape[0]:
                    classes = [f'{int(cls.tolist())}-{conf.tolist():.3f}' for conf, cls in pred[:, 4:]]
                    writer.add_image_with_boxes(f'{title}_image/{bs_index}',
                                                images[index], pred[:, :4], index, dataformats=dfmt, labels=classes)
                else:
                    writer.add_image(f'{title}_image/{bs_index}', images[index], index, dataformats=dfmt)
            else:
                break
