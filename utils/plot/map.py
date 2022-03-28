r"""
Plot map Module.
Consist of feature map, class activation mapping(attention map).
All map color is relative.
"""

import cv2
import torch
import numpy as np

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
        return model_output.sum(dim=(-2, -1))  # shape(bs, n)


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


class HookFeaturesFromModuleLayers:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module]):

        self.model = model
        self.features = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.hook_features))

    def __call__(self, x):
        self.features = []
        return self.model(x)

    def hook_features(self, module, input_tensor, output_tensor):
        self.features.append(output_tensor.cpu().detach())

    def release(self):
        for handle in self.handles:
            handle.remove()


def load_image_np_tensor(img_path, to_img_size):
    img = load_image_resize(img_path, to_img_size)[0].astype(np.float32) / 255  # RGB float 0-1
    img_tensor = np.transpose(img, (2, 0, 1))[None]
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor)
    return img, img_tensor


def parse_feature_map(features, to_wh, separate=False, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET):
    # for only one image feature
    features = np.array(features, dtype=np.float32)
    if not separate:
        assert features.ndim == 4, f'The ndim of features should be 4 but got {features.ndim}'
        features = np.sum(features.clip(0), axis=1)
    else:
        features = features.clip(0)
    features = features.squeeze()
    features = cv2.resize(features, to_wh)
    features /= np.max(features)
    features = np.uint8(255 * features)
    features = cv2.applyColorMap(features, colormap)
    if use_rgb:
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
    return features


def get_attention_map(img_path, img_size, model_instance, target_layers: list, layer, anchor, idx, use_cuda=True):
    img, img_tensor = load_image_np_tensor(img_path, img_size)
    cam = _GradCAM(model_instance, target_layers, use_cuda=use_cuda)
    targets = [_DetectOutputTarget(layer, anchor, idx)]
    gray_cams = cam(img_tensor, targets)
    attention_maps = []
    for gray_cam in gray_cams:
        attention_maps.append(show_cam_on_image(img, gray_cam, use_rgb=True)[..., ::-1])
    return attention_maps  # list of np.uint8 BGR to show or cv2.imwrite


def get_feature_map(img_path, img_size, model_instance, target_layers: list, separate=False, use_cuda=True):
    img, img_tensor = load_image_np_tensor(img_path, img_size)
    wh = img_tensor.shape[3], img_tensor.shape[2]

    if use_cuda:
        model_instance.cuda()
        img_tensor = img_tensor.cuda()

    # hook features
    hook_features = HookFeaturesFromModuleLayers(model_instance, target_layers)
    hook_features(img_tensor)
    features = hook_features.features
    hook_features.release()

    feature_maps = []
    for feature in features:
        if separate:
            feature_separates = []
            for feature_separate in feature.transpose(1, 0).contiguous():
                feature_separates.append(parse_feature_map(feature_separate, wh, separate))
            feature_maps.append(feature_separates)
        else:
            feature_maps.append(parse_feature_map(feature, wh))
    return feature_maps  # list of np.uint8 BGR to show or cv2.imwrite


def demo_attention_map():
    from models.yolov5.yolov5_v6 import yolov5s_v6

    path = '../../data/images/dog.jpg'
    img_size = 640
    weights = torch.load('../../runs/train/exp1/weights/best.pt')['model'].state_dict()
    model_instance = yolov5s_v6(num_class=20, decode=True)
    model_instance.load_state_dict(weights)
    target_layers = [model_instance.head.m[0]]
    layer, anchor, idx = 0, 0, 14
    img_attentions = get_attention_map(path, img_size, model_instance, target_layers, layer, anchor, idx)
    for img_attention in img_attentions:
        print(img_attention.shape)
        cv2.imshow('dog_attention', img_attention)
        cv2.waitKey(0)


def demo_feature_map():
    from models.yolov5.yolov5_v6 import yolov5s_v6

    path = '../../data/images/dog.jpg'
    img_size = 640
    weights = torch.load('../../models/yolov5/yolov5s_v6.pt')['model'].state_dict()
    model_instance = yolov5s_v6(num_class=80, decode=True)
    model_instance.load_state_dict(weights)
    target_layers = [model_instance.backbone.block3[3]]
    separate = True

    feature_maps = get_feature_map(path, img_size, model_instance, target_layers, separate)
    for feature_map in feature_maps:
        if separate:
            for feature in feature_map:
                print(feature.shape)
                cv2.imshow('dog_attention', feature)
                cv2.waitKey(0)
        else:
            print(feature_map.shape)
            cv2.imshow('dog_attention', feature_map)
            cv2.waitKey(0)


if __name__ == '__main__':
    demo_attention_map()
