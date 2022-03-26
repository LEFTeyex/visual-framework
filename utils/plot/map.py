r"""
Plot map Module.
Consist of feature map, class activation mapping(attention map).
"""

import cv2
import torch
import numpy as np

from torch import Tensor
from typing import List
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.datasets import load_image_resize


class _DetectOutputTarget(torch.nn.Module):
    def __init__(self, layer: int, anchor_idx: int, idx: int):
        super(_DetectOutputTarget, self).__init__()
        self.layer = layer
        self.anchor_idx = anchor_idx
        self.idx = idx

    def forward(self, model_output):
        r"""
        The model_output shape is list[ shape(bs, layer, h, w, (xywh + obj + cls)) ]
        Returns:
            model_output
        """
        model_output = model_output[self.layer][:, self.anchor_idx, :, :, self.idx]
        # TODO verify the bs can be multiple
        return model_output.sum(-1).sum(-1)  # shape(bs, n)


class _GradCAM(GradCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(_GradCAM, self).__init__(model, target_layers, use_cuda, reshape_transform)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum(sum([target(output) for target in targets]))
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)


def feature_map(features, to_wh, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET):
    features = np.array(features, dtype=np.float32)
    assert features.ndim == 4, f'The ndim of features should be 4 but got {features.ndim}'
    features = np.sum(np.max(features, 0), axis=1)
    features = cv2.resize(features, to_wh)
    features /= np.max(features)
    features = np.uint8(255 * features)
    # features = 255 - features
    features = cv2.applyColorMap(features, colormap)
    if use_rgb:
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
    return features


def get_attention_map(img_path, img_size, model_instance, target_layers: list, layer, anchor, idx, use_cuda=True):
    # only one now everything
    img = load_image_resize(img_path, img_size)[0].astype(np.float32) / 255  # RGB float 0-1
    img_tensor = np.transpose(img, (2, 0, 1))[None]
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor)
    cam = _GradCAM(model_instance, target_layers, use_cuda=use_cuda)
    targets = [_DetectOutputTarget(layer, anchor, idx)]
    gray_cam = cam(img_tensor, targets)[0]
    visualization = show_cam_on_image(img, gray_cam, use_rgb=True)[..., ::-1]
    return visualization  # np.uint8 BGR to show or cv2.imwrite


def get_feature_map(img_path, img_size, model_instance, target_layers: list, use_cuda=True):
    img = load_image_resize(img_path, img_size)[0].astype(np.float32) / 255  # RGB float 0-1
    img_tensor = np.transpose(img, (2, 0, 1))[None]
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor)
    wh = img_tensor.shape[3], img_tensor.shape[2]

    # TODO 2022.3.26 learn hook in torch to get features
    # TODO get features
    if use_cuda:
        model_instance.cuda()
        img_tensor.cuda()
    features = Tensor.cpu()
    # TODO

    features = feature_map(features, wh)
    return features  # np.uint8 BGR to show or cv2.imwrite


def demo_attention_map():
    from models.yolov5.yolov5_v6 import yolov5s_v6

    path = '../../data/images/dog.jpg'
    img_size = 640
    weights = torch.load('../../runs/train/exp1/weights/best.pt')['model'].state_dict()
    model_instance = yolov5s_v6(num_class=20, decode=True)
    model_instance.load_state_dict(weights)
    target_layers = [model_instance.head.m[0]]
    layer, anchor, idx = 0, 0, 14
    img_attention = get_attention_map(path, img_size, model_instance, target_layers, layer, anchor, idx)
    print(img_attention.shape)
    cv2.imshow('dog_attention', img_attention)
    cv2.waitKey(0)


def demo_feature_map():
    pass


if __name__ == '__main__':
    demo_attention_map()
