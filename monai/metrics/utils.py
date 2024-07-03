# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

from monai.config import NdarrayOrTensor, NdarrayTensor
from monai.transforms.croppad.array import SpatialCrop
from monai.transforms.utils import generate_spatial_bounding_box
from monai.utils import MetricReduction, convert_data_type, look_up_option, optional_import

binary_erosion, _ = optional_import("scipy.ndimage.morphology", name="binary_erosion")
distance_transform_edt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_edt")
distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")

__all__ = ["ignore_background", "do_metric_reduction", "get_mask_edges", "get_surface_distance", "is_binary_tensor"]


def ignore_background(y_pred: NdarrayTensor, y: NdarrayTensor) -> tuple[NdarrayTensor, NdarrayTensor]:
    """
    This function is used to remove background (the first channel) for `y_pred` and `y`.

    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        y: ground truth, the first dim is batch.

    """

    y = y[:, 1:] if y.shape[1] > 1 else y  # type: ignore[assignment]
    y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred  # type: ignore[assignment]
    return y_pred, y


def do_metric_reduction(
    f: torch.Tensor, reduction: MetricReduction | str = MetricReduction.MEAN
) -> tuple[torch.Tensor | Any, torch.Tensor]:
    """
    This function is to do the metric reduction for calculated `not-nan` metrics of each sample's each class.
    The function also returns `not_nans`, which counts the number of not nans for the metric.

    Args:
        f: a tensor that contains the calculated metric scores per batch and
            per class. The first two dims should be batch and class.
        reduction: define the mode to reduce metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``.
            if "none", return the input f tensor and not_nans.

    Raises:
        ValueError: When ``reduction`` is not one of
            ["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].
    """

    # some elements might be Nan (if ground truth y was missing (zeros))
    # we need to account for it
    nans = torch.isnan(f)
    not_nans = (~nans).float()

    t_zero = torch.zeros(1, device=f.device, dtype=f.dtype)
    reduction = look_up_option(reduction, MetricReduction)
    if reduction == MetricReduction.NONE:
        return f, not_nans

    f[nans] = 0
    if reduction == MetricReduction.MEAN:
        # 2 steps, first, mean by channel (accounting for nans), then by batch
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average

        not_nans = (not_nans > 0).float().sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average

    elif reduction == MetricReduction.SUM:
        not_nans = not_nans.sum(dim=[0, 1])
        f = torch.sum(f, dim=[0, 1])  # sum over the batch and channel dims
    elif reduction == MetricReduction.MEAN_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average
    elif reduction == MetricReduction.SUM_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = f.sum(dim=0)  # the batch sum
    elif reduction == MetricReduction.MEAN_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average
    elif reduction == MetricReduction.SUM_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = f.sum(dim=1)  # the channel sum
    elif reduction != MetricReduction.NONE:
        raise ValueError(
            f"Unsupported reduction: {reduction}, available options are "
            '["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].'
        )
    return f, not_nans


def get_mask_edges(
    seg_pred: NdarrayOrTensor, seg_gt: NdarrayOrTensor, label_idx: int = 1, crop: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Do binary erosion and use XOR for input to get the edges. This
    function is helpful to further calculate metrics such as Average Surface
    Distance and Hausdorff Distance.
    The input images can be binary or labelfield images. If labelfield images
    are supplied, they are converted to binary images using `label_idx`.

    `scipy`'s binary erosion is used to calculate the edges of the binary
    labelfield.

    In order to improve the computing efficiency, before getting the edges,
    the images can be cropped and only keep the foreground if not specifies
    ``crop = False``.

    We require that images are the same size, and assume that they occupy the
    same space (spacing, orientation, etc.).

    Args:
        seg_pred: the predicted binary or labelfield image.
        seg_gt: the actual binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        crop: crop input images and only keep the foregrounds. In order to
            maintain two inputs' shapes, here the bounding box is achieved
            by ``(seg_pred | seg_gt)`` which represents the union set of two
            images. Defaults to ``True``.
    """

    # Get both labelfields as np arrays
    if isinstance(seg_pred, torch.Tensor):
        seg_pred = seg_pred.detach().cpu().numpy()
    if isinstance(seg_gt, torch.Tensor):
        seg_gt = seg_gt.detach().cpu().numpy()

    if seg_pred.shape != seg_gt.shape:
        raise ValueError(f"seg_pred and seg_gt should have same shapes, got {seg_pred.shape} and {seg_gt.shape}.")

    # If not binary images, convert them
    if seg_pred.dtype != bool:
        seg_pred = seg_pred == label_idx
    if seg_gt.dtype != bool:
        seg_gt = seg_gt == label_idx

    if crop:
        if not np.any(seg_pred | seg_gt):
            return np.zeros_like(seg_pred), np.zeros_like(seg_gt)

        channel_dim = 0
        seg_pred, seg_gt = np.expand_dims(seg_pred, axis=channel_dim), np.expand_dims(seg_gt, axis=channel_dim)
        box_start, box_end = generate_spatial_bounding_box(np.asarray(seg_pred | seg_gt))
        cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        seg_pred = convert_data_type(np.squeeze(cropper(seg_pred), axis=channel_dim), np.ndarray)[0]  # type: ignore[arg-type]
        seg_gt = convert_data_type(np.squeeze(cropper(seg_gt), axis=channel_dim), np.ndarray)[0]  # type: ignore[arg-type]

    # Do binary erosion and use XOR to get edges
    edges_pred = binary_erosion(seg_pred) ^ seg_pred
    edges_gt = binary_erosion(seg_gt) ^ seg_gt

    return edges_pred, edges_gt


def get_surface_distance(seg_pred: np.ndarray, seg_gt: np.ndarray, distance_metric: str = "euclidean") -> np.ndarray:
    """
    This function is used to compute the surface distances from `seg_pred` to `seg_gt`.

    Args:
        seg_pred: the edge of the predictions.
        seg_gt: the edge of the ground truth.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.

            - ``"euclidean"``, uses Exact Euclidean distance transform.
            - ``"chessboard"``, uses `chessboard` metric in chamfer type of transform.
            - ``"taxicab"``, uses `taxicab` metric in chamfer type of transform.

    Note:
        If seg_pred or seg_gt is all 0, may result in nan/inf distance.

    """

    if not np.any(seg_gt):
        dis = np.inf * np.ones_like(seg_gt)
    else:
        if not np.any(seg_pred):
            dis = np.inf * np.ones_like(seg_gt)
            return np.asarray(dis[seg_gt])
        if distance_metric == "euclidean":
            dis = distance_transform_edt(~seg_gt)
        elif distance_metric in {"chessboard", "taxicab"}:
            dis = distance_transform_cdt(~seg_gt, metric=distance_metric)
        else:
            raise ValueError(f"distance_metric {distance_metric} is not implemented.")

    return np.asarray(dis[seg_pred])


def is_binary_tensor(input: torch.Tensor, name: str) -> None:
    """Determines whether the input tensor is torch binary tensor or not.

    Args:
        input (torch.Tensor): tensor to validate.
        name (str): name of the tensor being checked.

    Raises:
        ValueError: if `input` is not a PyTorch Tensor.

    Note:
        A warning message is printed, if the tensor is not binary.
    """
    if not isinstance(input, torch.Tensor):
        raise ValueError(f"{name} must be of type PyTorch Tensor.")
    if not torch.all(input.byte() == input) or input.max() > 1 or input.min() < 0:
        warnings.warn(f"{name} should be a binarized tensor.")


def remap_instance_id(pred: torch.Tensor, by_size: bool = False) -> torch.Tensor:
    """
    This function is used to rename all instance id of `pred`, so that the id is
    contiguous.
    For example: all ids of the input can be [0, 1, 2] rather than [0, 2, 5].
    This function is helpful for calculating metrics like Panoptic Quality (PQ).
    The implementation refers to:

    https://github.com/vqdang/hover_net

    Args:
        pred: segmentation predictions in the form of torch tensor. Each
            value of the tensor should be an integer, and represents the prediction of its corresponding instance id.
        by_size: if True, largest instance will be assigned a smaller id.

    """
    pred_id = list(pred.unique())
    # the original implementation has the limitation that if there is no 0 in pred, error will happen
    pred_id = [i for i in pred_id if i != 0]

    if len(pred_id) == 0:
        return pred
    if by_size is True:
        instance_size = []
        for instance_id in pred_id:
            instance_size.append((pred == instance_id).sum())

        pair_data = zip(pred_id, instance_size)
        pair_list = sorted(pair_data, key=lambda x: x[1], reverse=True)  # type: ignore
        pred_id, _ = zip(*pair_list)

    new_pred = torch.zeros_like(pred, dtype=torch.int)
    for idx, instance_id in enumerate(pred_id):
        new_pred[pred == instance_id] = idx + 1
    return new_pred
