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

import math
from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction
from monai.utils.type_conversion import convert_to_dst_type

from .metric import CumulativeIterationMetric


class RegressionMetric(CumulativeIterationMetric):
    """
    Base class for regression metrics.
    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def _check_shape(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y shapes dont match, received y_pred: [{y_pred.shape}] and y: [{y.shape}]")

        # also check if there is atleast one non-batch dimension i.e. num_dims >= 2
        if len(y_pred.shape) < 2:
            raise ValueError("either channel or spatial dimensions required, found only batch dimension")

    @abstractmethod
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        self._check_shape(y_pred, y)
        return self._compute_metric(y_pred, y)


class MSEMetric(RegressionMetric):
    r"""Compute Mean Squared Error between two tensors using function:

    .. math::
        \operatorname {MSE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i} \right)^{2}.

    More info: https://en.wikipedia.org/wiki/Mean_squared_error

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

        return compute_mean_error_metrics(y_pred, y, func=self.sq_func)


class MAEMetric(RegressionMetric):
    r"""Compute Mean Absolute Error between two tensors using function:

    .. math::
        \operatorname {MAE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left|y_i-\hat{y_i}\right|.

    More info: https://en.wikipedia.org/wiki/Mean_absolute_error

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.abs_func = torch.abs

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

        return compute_mean_error_metrics(y_pred, y, func=self.abs_func)


class RMSEMetric(RegressionMetric):
    r"""Compute Root Mean Squared Error between two tensors using function:

    .. math::
        \operatorname {RMSE}\left(Y, \hat{Y}\right) ={ \sqrt{ \frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i}\right)^2 } } \
        = \sqrt {\operatorname{MSE}\left(Y, \hat{Y}\right)}.

    More info: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

        mse_out = compute_mean_error_metrics(y_pred, y, func=self.sq_func)
        return torch.sqrt(mse_out)


class PSNRMetric(RegressionMetric):
    r"""Compute Peak Signal To Noise Ratio between two tensors using function:

    .. math::
        \operatorname{PSNR}\left(Y, \hat{Y}\right) = 20 \cdot \log_{10} \left({\mathit{MAX}}_Y\right) \
        -10 \cdot \log_{10}\left(\operatorname{MSE\left(Y, \hat{Y}\right)}\right)

    More info: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Help taken from:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py line 4139

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        max_val: The dynamic range of the images/volumes (i.e., the difference between the
            maximum and the minimum allowed values e.g. 255 for a uint8 image).
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(
        self, max_val: int | float, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.max_val = max_val
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> Any:
        y_pred = y_pred.float()
        y = y.float()

        mse_out = compute_mean_error_metrics(y_pred, y, func=self.sq_func)
        return 20 * math.log10(self.max_val) - 10 * torch.log10(mse_out)


def compute_mean_error_metrics(y_pred: torch.Tensor, y: torch.Tensor, func: Callable) -> torch.Tensor:
    # reducing in only channel + spatial dimensions (not batch)
    # reduction of batch handled inside __call__() using do_metric_reduction() in respective calling class
    flt = partial(torch.flatten, start_dim=1)
    return torch.mean(flt(func(y - y_pred)), dim=-1, keepdim=True)


class SSIMMetric(RegressionMetric):
    r"""
    Build a Pytorch version of the SSIM metric based on the original formula of SSIM

    .. math::
        \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/ssim_loss_mixin.py

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

    Args:
        data_range: dynamic range of the data
        win_size: gaussian weighting window size
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        spatial_dims: if 2, input shape is expected to be (B,C,W,H). if 3, it is expected to be (B,C,W,H,D)
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        data_range: torch.Tensor,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
        spatial_dims: int = 2,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ):
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.data_range = data_range
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.spatial_dims = spatial_dims
        self.cov_norm = (win_size**2) / (win_size**2 - 1)
        self.w = torch.ones([1, 1] + [win_size for _ in range(spatial_dims)]) / win_size**spatial_dims

    def _compute_intermediate_statistics(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        data_range = self.data_range[(None,) * (self.spatial_dims + 2)]
        # determine whether to work with 2D convolution or 3D
        conv = getattr(F, f"conv{self.spatial_dims}d")
        w = convert_to_dst_type(src=self.w, dst=x)[0]

        c1 = (self.k1 * data_range) ** 2  # stability constant for luminance
        c2 = (self.k2 * data_range) ** 2  # stability constant for contrast
        ux = conv(x, w)  # mu_x
        uy = conv(y, w)  # mu_y
        uxx = conv(x * x, w)  # mu_x^2
        uyy = conv(y * y, w)  # mu_y^2
        uxy = conv(x * y, w)  # mu_xy
        vx = self.cov_norm * (uxx - ux * ux)  # sigma_x
        vy = self.cov_norm * (uyy - uy * uy)  # sigma_y
        vxy = self.cov_norm * (uxy - ux * uy)  # sigma_xy

        return c1, c2, ux, uy, vx, vy, vxy

    def _compute_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D data and (B,C,W,H,D) for 3D.
                A fastMRI sample should use the 2D format with C being the number of slices.
            y: second sample (e.g., the reconstructed image). It has similar shape as x

        Returns:
            ssim_value

        Example:
            .. code-block:: python

                import torch
                x = torch.ones([1,1,10,10])/2 # ground truth
                y = torch.ones([1,1,10,10])/2 # prediction
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(SSIMMetric(data_range=data_range,spatial_dims=2)._compute_metric(x,y))
        """
        if x.shape[1] > 1:  # handling multiple channels (C>1)
            if x.shape[1] != y.shape[1]:
                raise ValueError(
                    f"x and y should have the same number of channels, "
                    f"but x has {x.shape[1]} channels and y has {y.shape[1]} channels."
                )

            ssim = torch.stack(
                [
                    SSIMMetric(self.data_range, self.win_size, self.k1, self.k2, self.spatial_dims)(  # type: ignore[misc]
                        x[:, i, ...].unsqueeze(1), y[:, i, ...].unsqueeze(1)
                    )
                    for i in range(x.shape[1])
                ]
            )
            channel_wise_ssim = ssim.mean(1).view(-1, 1)
            return channel_wise_ssim

        c1, c2, ux, uy, vx, vy, vxy = self._compute_intermediate_statistics(x, y)

        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denom = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        ssim_value = numerator / denom
        # [B, 1]
        ssim_per_batch: torch.Tensor = ssim_value.view(ssim_value.shape[1], -1).mean(1, keepdim=True)

        return ssim_per_batch

    def _compute_metric_and_contrast(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D data and (B,C,W,H,D) for 3D.
                A fastMRI sample should use the 2D format with C being the number of slices.
            y: second sample (e.g., the reconstructed image). It has similar shape as x

        Returns:
            ssim_value, cs_value
        """
        if x.shape[1] > 1:  # handling multiple channels (C>1)
            if x.shape[1] != y.shape[1]:
                raise ValueError(
                    f"x and y should have the same number of channels, "
                    f"but x has {x.shape[1]} channels and y has {y.shape[1]} channels."
                )

            ssim_ls = []
            cs_ls = []
            for i in range(x.shape[1]):
                ssim_val, cs_val = SSIMMetric(
                    self.data_range, self.win_size, self.k1, self.k2, self.spatial_dims
                )._compute_metric_and_contrast(x[:, i, ...].unsqueeze(1), y[:, i, ...].unsqueeze(1))
                ssim_ls.append(ssim_val)
                cs_ls.append(cs_val)
            channel_wise_ssim: torch.Tensor = torch.stack(ssim_ls).mean(1).view(-1, 1)
            channel_wise_cs: torch.Tensor = torch.stack(cs_ls).mean(1).view(-1, 1)
            return channel_wise_ssim, channel_wise_cs

        c1, c2, ux, uy, vx, vy, vxy = self._compute_intermediate_statistics(x, y)

        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denom = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        ssim_value = numerator / denom
        # [B, 1]
        ssim_per_batch: torch.Tensor = ssim_value.view(ssim_value.shape[1], -1).mean(1, keepdim=True)

        cs_per_batch: torch.Tensor = (2 * vxy + c2) / (vx + vy + c2)  # contrast sensitivity function
        cs_per_batch = cs_per_batch.view(cs_per_batch.shape[0], -1).mean(1, keepdim=True)  # [B, 1]
        return ssim_per_batch, cs_per_batch
