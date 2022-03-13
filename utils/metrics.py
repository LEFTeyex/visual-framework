r"""
Metrics utils.
Consist of all metrics function for evaluating model.
"""

import torch
import numpy as np

from torchvision.ops import box_iou

__all__ = ['match_pred_label_iou_vector', 'compute_metrics_per_class',
           'compute_ap', 'compute_fps', 'compute_fitness']


def match_pred_label_iou_vector(pred, label, iou_vector):
    r"""
    Match prediction to label and compare it with iou_vector.
    Args:
        pred: = prediction (n, 6 (x,y,x,y,conf,cls) )
        label: = label (m, 5 (cls,x,y,x,y) )
        iou_vector: = iou_vector (10)

    Return pred_iou_level (n, 10) for IoU levels
    """
    pred_iou_level = torch.zeros((pred.shape[0], iou_vector.shape[0]), dtype=torch.bool, device=iou_vector.device)
    iou = box_iou(label[:, 1:], pred[:, :4])  # iou shape (n_label, n_pred)
    x = torch.nonzero((iou >= iou_vector[0]) & (label[:, 0:1] == pred[:, 5]))  # index tensor
    if x.shape[0]:
        iou = iou[x.T[0], x.T[1]].view(-1, 1)  # shape to x
        matches = torch.cat((x, iou), dim=-1).cpu().numpy()
        if x.shape[0] > 1:
            # make pred and label correspond one by one
            # need to think carefully it is interesting and great
            matches = matches[np.argsort(matches[:, 2])[::-1]]  # sort large to small
            # filter pred first corresponding to nms and avoiding labels from losing
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.argsort(matches[:, 2])[::-1]]
            # filter label second
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.tensor(matches, device=iou_vector.device)
        pred_iou_level[matches[:, 1].long()] = matches[:, 2:3] >= iou_vector
    return pred_iou_level


def compute_metrics_per_class(tp, conf, pred_cls, label_cls, eps=1e-16):
    r"""
    Compute the average precision(AP), F1 score(F1), precision(P), recall(R) and other metrics.
    Args:
        tp:
        conf:
        pred_cls:
        label_cls:
        eps:

    Return
    """
    # sort from large to small
    order_ls = np.argsort(conf)[::-1]
    tp, conf, pred_cls = tp[order_ls], conf[order_ls], pred_cls[order_ls]

    # get unique class, number of classes
    unique_cls, n_unique_cls = np.unique(label_cls, return_counts=True)
    nc = unique_cls.shape[0]  # number of classes
    niou = tp.shape[1]  # number of iou threshold   10-(0.50:0.95:0.05 usually)

    # init
    ap, r, p = np.zeros((nc, niou)), np.zeros((nc, niou)), np.zeros((nc, niou))
    pr_curve = [[] for _ in range(nc)]
    prx = np.linspace(0, 1, 1000)

    # compute precision-recall(P-R) curve and AP
    for index_unique, cls in enumerate(unique_cls):
        index_tp = pred_cls == cls
        n_label = n_unique_cls[index_unique]  # number of labels
        n_pred = np.sum(index_tp)  # number of predictions

        if n_label == 0 or n_pred == 0:
            continue
        else:
            tpc = tp[index_tp].cumsum(0)
            fpc = (1 - tp[index_tp]).cumsum(0)

            # R and P
            recall = tpc / n_label
            precision = tpc / (tpc + fpc)

            # get R and P
            r[index_unique] = recall[-1]
            p[index_unique] = precision[-1]

            # AP
            pry = np.zeros((niou, 1000))
            for index_iou in range(niou):
                ap[index_unique, index_iou], _r, _p = compute_ap(recall[:, index_iou], precision[:, index_iou])
                pry[index_iou] = np.interp(prx, _r, _p)
            pr_curve[index_unique].append(pry)
            # TODO plot P-R curve by pr_curve

    # do not consist of f1/r/p---confidence curve

    # TP and FP (no use usually)
    # n_tp = (r * n_unique_cls[:, None]).round()
    # n_fp = (n_tp / (p + eps) - tp).round()

    # F1
    f1 = 2 * (r * p) / (r + p + eps)
    index_cls = unique_cls  # corresponding to the position of result index
    return ap, f1, p, r, index_cls


def compute_ap(recall, precision):
    r"""
    Compute the average precision by COCO method.
    Args:
        recall:
        precision:

    Return
    """
    # add sentinel values to beginning and end
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))

    # compute the precision envelope
    p = np.flip(np.maximum.accumulate(np.flip(p)))

    # interpolation in P-R curve for ap
    x = np.linspace(0, 1, 101)
    y = np.interp(x, r, p)
    ap = np.trapz(y, x)

    # there is other method for ap before VOC-2010
    # index = np.where(r[1:] != r[:-1])[0]
    # ap = np.sum((r[index + 1] - r[index]) * p[index + 1])
    return ap, r, p


def compute_fps(seen: int, time: float):
    r"""
    Compute fps and time per image.
    ***** exclude image preprocessing time *****
    Return fps, time_per_image
    """
    time_per_image = time / seen
    fps = 1 / time_per_image
    return fps, time_per_image * 1000  # the unit is ms


def compute_fitness(results, weights):
    if sum(weights) != 1:
        raise ValueError(f'The sum of weights must be 1 but got {weights}')
    weights = np.asarray(weights).reshape(-1)
    results = np.asarray(results).reshape(-1)
    if results.shape != weights.shape:
        raise ValueError(f'The shape is not equal results shape {results.shape} and weights shape {weights.shape}')
    fitness = (results * weights).sum().tolist()
    return fitness


if __name__ == '__main__':
    pass
