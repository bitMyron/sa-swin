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
"""
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from itertools import zip_longest
from typing import Any, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch

from monai.config import USE_COMPILED, DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, affine_to_spacing, compute_shape_offset, iter_patch, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.networks.utils import meshgrid_ij, normalize_transform
from monai.transforms.croppad.array import CenterSpatialCrop, ResizeWithPadOrCrop
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import (
    convert_pad_mode,
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
    scale_affine,
)
from monai.transforms.utils_pytorch_numpy_unification import allclose, linalg_inv, moveaxis, where
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NdimageMode,
    NumpyPadMode,
    SplineMode,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.enums import GridPatchSort, PytorchPadMode, TraceKeys, TransformBackends, WSIPatchKeys
from monai.utils.misc import ImageMetaKey as Key
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, get_equivalent_dtype, get_torch_dtype_from_string

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = [
    "SpatialResample",
    "ResampleToMatch",
    "Spacing",
    "Orientation",
    "Flip",
    "GridDistortion",
    "GridSplit",
    "GridPatch",
    "RandGridPatch",
    "Resize",
    "Rotate",
    "Zoom",
    "Rotate90",
    "RandRotate90",
    "RandRotate",
    "RandFlip",
    "RandGridDistortion",
    "RandAxisFlip",
    "RandZoom",
    "AffineGrid",
    "RandAffineGrid",
    "RandDeformGrid",
    "Resample",
    "Affine",
    "RandAffine",
    "Rand2DElastic",
    "Rand3DElastic",
]

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]


class SpatialResample(InvertibleTransform):
    """
    Resample input image from the orientation/spacing defined by ``src_affine`` affine matrix into
    the ones specified by ``dst_affine`` affine matrix.

    Internally this transform computes the affine transform matrix from ``src_affine`` to ``dst_affine``,
    by ``xform = linalg.solve(src_affine, dst_affine)``, and call ``monai.transforms.Affine`` with ``xform``.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY, TransformBackends.CUPY]

    def __init__(
        self,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
    ):
        """
        Args:
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

    def _post_process(
        self,
        img: torch.Tensor,
        src_affine: torch.Tensor,
        dst_affine: torch.Tensor,
        mode,
        padding_mode,
        align_corners,
        original_spatial_shape,
    ) -> torch.Tensor:
        """
        Small fn to simplify returning data. If `MetaTensor`, update affine. Elif
        tracking metadata is desired, create `MetaTensor` with affine. Else, return
        image as `torch.Tensor`. Output type is always `float32`.

        Also append the transform to the stack.
        """
        dtype = img.dtype
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        if get_track_meta():
            self.update_meta(img, dst_affine)
            self.push_transform(
                img,
                extra_info={
                    "dtype": str(dtype)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                    "src_affine": src_affine,
                },
                orig_size=original_spatial_shape,
            )
        return img

    def update_meta(self, img, dst_affine):
        img.affine = dst_affine

    def __call__(
        self,
        img: torch.Tensor,
        dst_affine: torch.Tensor | None = None,
        spatial_size: Sequence[int] | torch.Tensor | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike = None,
    ) -> torch.Tensor:
        """
        Args:
            img: input image to be resampled. It currently supports channel-first arrays with
                at most three spatial dimensions.
            dst_affine: destination affine matrix. Defaults to ``None``, which means the same as `img.affine`.
                the shape should be `(r+1, r+1)` where `r` is the spatial rank of ``img``.
                when `dst_affine` and `spatial_size` are None, the input will be returned without resampling,
                but the data type will be `float32`.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined,
                the transform will compute a spatial size automatically containing the previous field of view.
                if `spatial_size` is ``-1`` are the transform will use the corresponding input img size.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                Defaults to ``None``, effectively using the value of `self.align_corners`.
            dtype: data type for resampling computation. Defaults to ``self.dtype`` or
                ``np.float64`` (for best precision). If ``None``, use the data type of input data.
                To be compatible with other modules, the output data type is always `float32`.

        The spatial rank is determined by the smallest among ``img.ndim -1``, ``len(src_affine) - 1``, and ``3``.

        When both ``monai.config.USE_COMPILED`` and ``align_corners`` are set to ``True``,
        MONAI's resampling implementation will be used.
        Set `dst_affine` and `spatial_size` to `None` to turn off the resampling step.
        """
        # get dtype as torch (e.g., torch.float64)
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        align_corners = self.align_corners if align_corners is None else align_corners
        mode = mode if mode is not None else self.mode
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        original_spatial_shape = img.shape[1:]

        src_affine_: torch.Tensor = img.affine if isinstance(img, MetaTensor) else torch.eye(4)
        img = convert_to_tensor(data=img, track_meta=get_track_meta(), dtype=_dtype)
        spatial_rank = min(len(img.shape) - 1, src_affine_.shape[0] - 1, 3)
        if (not isinstance(spatial_size, int) or spatial_size != -1) and spatial_size is not None:
            spatial_rank = min(len(ensure_tuple(spatial_size)), 3)  # infer spatial rank based on spatial_size
        src_affine_ = to_affine_nd(spatial_rank, src_affine_).to(_dtype)
        dst_affine = to_affine_nd(spatial_rank, dst_affine) if dst_affine is not None else src_affine_
        dst_affine = convert_to_dst_type(dst_affine, src_affine_)[0]
        if not isinstance(dst_affine, torch.Tensor):
            raise ValueError(f"dst_affine should be a torch.Tensor, got {type(dst_affine)}")

        in_spatial_size = torch.tensor(img.shape[1 : spatial_rank + 1])
        if isinstance(spatial_size, int) and (spatial_size == -1):  # using the input spatial size
            spatial_size = in_spatial_size
        elif spatial_size is None and spatial_rank > 1:  # auto spatial size
            spatial_size, _ = compute_shape_offset(in_spatial_size, src_affine_, dst_affine)  # type: ignore
        spatial_size = torch.tensor(fall_back_tuple(ensure_tuple(spatial_size)[:spatial_rank], in_spatial_size))

        if (
            allclose(src_affine_, dst_affine, atol=AFFINE_TOL)
            and allclose(spatial_size, in_spatial_size)
            or spatial_rank == 1
        ):
            # no significant change, return original image
            return self._post_process(
                img, src_affine_, src_affine_, mode, padding_mode, align_corners, original_spatial_shape
            )

        try:
            _s = convert_to_numpy(src_affine_)
            _d = convert_to_numpy(dst_affine)
            xform = np.linalg.solve(_s, _d)  # monai#5983
        except (np.linalg.LinAlgError, RuntimeError) as e:
            raise ValueError(f"src affine is not invertible {_s}, {_d}.") from e
        xform = convert_to_tensor(to_affine_nd(spatial_rank, xform)).to(device=img.device, dtype=_dtype)
        # no resampling if it's identity transform
        if allclose(xform, torch.eye(len(xform)), atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size):
            return self._post_process(
                img, src_affine_, src_affine_, mode, padding_mode, align_corners, original_spatial_shape
            )

        in_spatial_size = in_spatial_size.tolist()  # type: ignore
        chns, additional_dims = img.shape[0], img.shape[spatial_rank + 1 :]  # beyond three spatial dims

        if additional_dims:
            xform_shape = [-1] + in_spatial_size
            img = img.reshape(xform_shape)  # type: ignore
        if isinstance(mode, int):
            dst_xform_1 = normalize_transform(spatial_size, "cpu", xform.dtype, True, True)[0].numpy()  # to (-1, 1)
            if not align_corners:
                norm = create_scale(spatial_rank, [(max(d, 2) - 1) / d for d in spatial_size])
                dst_xform_1 = norm.astype(float) @ dst_xform_1  # type: ignore  # scaling (num_step - 1) / num_step
            dst_xform_d = normalize_transform(spatial_size, "cpu", xform.dtype, align_corners, False)[0].numpy()
            xform @= convert_to_dst_type(np.linalg.solve(dst_xform_d, dst_xform_1), xform)[0]
            affine_xform = Affine(
                affine=xform, spatial_size=spatial_size, normalized=True, image_only=True, dtype=_dtype  # type: ignore
            )
            with affine_xform.trace_transform(False):
                img = affine_xform(img, mode=mode, padding_mode=padding_mode)  # type: ignore
        else:
            affine_xform = AffineTransform(  # type: ignore
                normalized=False,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                reverse_indexing=True,
            )
            img = affine_xform(img.unsqueeze(0), theta=xform, spatial_size=spatial_size).squeeze(0)  # type: ignore
        if additional_dims:
            full_shape = (chns, *spatial_size, *additional_dims)
            img = img.reshape(full_shape)

        return self._post_process(
            img, src_affine_, dst_affine, mode, padding_mode, align_corners, original_spatial_shape
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # Create inverse transform
        kw_args = transform[TraceKeys.EXTRA_INFO]
        # need to convert dtype from string back to torch.dtype
        kw_args["dtype"] = get_torch_dtype_from_string(kw_args["dtype"])
        # source becomes destination
        kw_args["dst_affine"] = kw_args.pop("src_affine")
        kw_args["spatial_size"] = transform[TraceKeys.ORIG_SIZE]
        if kw_args.get("align_corners") == TraceKeys.NONE:
            kw_args["align_corners"] = False
        with self.trace_transform(False):
            # we can't use `self.__call__` in case a child class calls this inverse.
            out: torch.Tensor = SpatialResample.__call__(self, data, **kw_args)
        return out


class ResampleToMatch(SpatialResample):
    """Resample an image to match given metadata. The affine matrix will be aligned,
    and the size of the output image will match."""

    def update_meta(self, img: torch.Tensor, dst_affine=None, img_dst=None):
        if dst_affine is not None:
            super().update_meta(img, dst_affine)
        if isinstance(img_dst, MetaTensor) and isinstance(img, MetaTensor):
            original_fname = img.meta.get(Key.FILENAME_OR_OBJ, "resample_to_match_source")
            img.meta = deepcopy(img_dst.meta)
            img.meta[Key.FILENAME_OR_OBJ] = original_fname  # keep the original name, the others are overwritten

    def __call__(  # type: ignore
        self,
        img: torch.Tensor,
        img_dst: torch.Tensor,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike = None,
    ) -> torch.Tensor:
        """
        Args:
            img: input image to be resampled to match ``img_dst``. It currently supports channel-first arrays with
                at most three spatial dimensions.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                Defaults to ``None``, effectively using the value of `self.align_corners`.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype`` or
                ``np.float64`` (for best precision). If ``None``, use the data type of input data.
                To be compatible with other modules, the output data type is always `float32`.
        Raises:
            ValueError: When the affine matrix of the source image is not invertible.
        Returns:
            Resampled input tensor or MetaTensor.
        """
        if img_dst is None:
            raise RuntimeError("`img_dst` is missing.")
        dst_affine = img_dst.affine if isinstance(img_dst, MetaTensor) else torch.eye(4)
        img = super().__call__(
            img=img,
            dst_affine=dst_affine,
            spatial_size=img_dst.shape[1:],  # skip channel
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=dtype,
        )
        self.update_meta(img, dst_affine=dst_affine, img_dst=img_dst)
        return img


class Spacing(InvertibleTransform):
    """
    Resample input image into the specified `pixdim`.
    """

    backend = SpatialResample.backend

    def __init__(
        self,
        pixdim: Sequence[float] | float | np.ndarray,
        diagonal: bool = False,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        scale_extent: bool = False,
        recompute_affine: bool = False,
        min_pixdim: Sequence[float] | float | np.ndarray | None = None,
        max_pixdim: Sequence[float] | float | np.ndarray | None = None,
    ) -> None:
        """
        Args:
            pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
                items of the pixdim sequence map to the spatial dimensions of input image, if length
                of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
                if shorter, will pad with the last value. For example, for 3D image if pixdim is [1.0, 2.0] it
                will be padded to [1.0, 2.0, 2.0]
                if the components of the `pixdim` are non-positive values, the transform will use the
                corresponding components of the original pixdim, which is computed from the `affine`
                matrix of input image.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            scale_extent: whether the scale is computed based on the spacing or the full extent of voxels,
                default False. The option is ignored if output spatial size is specified when calling this transform.
                See also: :py:func:`monai.data.utils.compute_shape_offset`. When this is True, `align_corners`
                should be `True` because `compute_shape_offset` already provides the corner alignment shift/scaling.
            recompute_affine: whether to recompute affine based on the output shape. The affine computed
                analytically does not reflect the potential quantization errors in terms of the output shape.
                Set this flag to True to recompute the output affine based on the actual pixdim. Default to ``False``.
            min_pixdim: minimal input spacing to be resampled. If provided, input image with a larger spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.
            max_pixdim: maximal input spacing to be resampled. If provided, input image with a smaller spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.

        """
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.min_pixdim = np.array(ensure_tuple(min_pixdim), dtype=np.float64)
        self.max_pixdim = np.array(ensure_tuple(max_pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.scale_extent = scale_extent
        self.recompute_affine = recompute_affine

        for mn, mx in zip(self.min_pixdim, self.max_pixdim):
            if (not np.isnan(mn)) and (not np.isnan(mx)) and ((mx < mn) or (mn < 0)):
                raise ValueError(f"min_pixdim {self.min_pixdim} must be positive, smaller than max {self.max_pixdim}.")

        self.sp_resample = SpatialResample(
            mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype=dtype
        )

    @deprecated_arg(name="affine", since="0.9", msg_suffix="Not needed, input should be `MetaTensor`.")
    def __call__(
        self,
        data_array: torch.Tensor,
        affine: NdarrayOrTensor | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike = None,
        scale_extent: bool | None = None,
        output_spatial_shape: Sequence[int] | np.ndarray | int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"self.mode"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"self.padding_mode"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                Defaults to ``None``, effectively using the value of `self.align_corners`.
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            scale_extent: whether the scale is computed based on the spacing or the full extent of voxels,
                The option is ignored if output spatial size is specified when calling this transform.
                See also: :py:func:`monai.data.utils.compute_shape_offset`. When this is True, `align_corners`
                should be `True` because `compute_shape_offset` already provides the corner alignment shift/scaling.
            output_spatial_shape: specify the shape of the output data_array. This is typically useful for
                the inverse of `Spacingd` where sometimes we could not compute the exact shape due to the quantization
                error with the affine.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``pixdim`` is nonpositive.

        Returns:
            data tensor or MetaTensor (resampled into `self.pixdim`).

        """
        original_spatial_shape = data_array.shape[1:]
        sr = len(original_spatial_shape)
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        affine_: np.ndarray
        if affine is not None:
            warnings.warn("arg `affine` is deprecated, the affine of MetaTensor in data_array has higher priority.")
        input_affine = data_array.affine if isinstance(data_array, MetaTensor) else affine
        if input_affine is None:
            warnings.warn("`data_array` is not of type MetaTensor, assuming affine to be identity.")
            # default to identity
            input_affine = np.eye(sr + 1, dtype=np.float64)
        affine_ = to_affine_nd(sr, convert_data_type(input_affine, np.ndarray)[0])

        out_d = self.pixdim[:sr].copy()
        if out_d.size < sr:
            out_d = np.append(out_d, [out_d[-1]] * (sr - out_d.size))

        orig_d = affine_to_spacing(affine_, sr, out_d.dtype)
        for idx, (_d, mn, mx) in enumerate(
            zip_longest(orig_d, self.min_pixdim[:sr], self.max_pixdim[:sr], fillvalue=np.nan)
        ):
            target = out_d[idx]
            mn = target if np.isnan(mn) else min(mn, target)
            mx = target if np.isnan(mx) else max(mx, target)
            if mn > mx:
                raise ValueError(f"min_pixdim is larger than max_pixdim at dim {idx}: min {mn} max {mx} out {target}.")
            out_d[idx] = _d if (mn - AFFINE_TOL) <= _d <= (mx + AFFINE_TOL) else target

        if not align_corners and scale_extent:
            warnings.warn("align_corners=False is not compatible with scale_extent=True.")

        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        scale_extent = self.scale_extent if scale_extent is None else scale_extent
        output_shape, offset = compute_shape_offset(data_array.shape[1:], affine_, new_affine, scale_extent)
        new_affine[:sr, -1] = offset[:sr]
        # convert to MetaTensor if necessary
        data_array = convert_to_tensor(data_array, track_meta=get_track_meta())
        if isinstance(data_array, MetaTensor):
            data_array.affine = torch.as_tensor(affine_)

        # we don't want to track the nested transform otherwise two will be appended
        actual_shape = list(output_shape) if output_spatial_shape is None else output_spatial_shape
        data_array = self.sp_resample(
            data_array,
            dst_affine=torch.as_tensor(new_affine),
            spatial_size=actual_shape,  # type: ignore
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=dtype,
        )
        if self.recompute_affine and isinstance(data_array, MetaTensor):
            data_array.affine = scale_affine(affine_, original_spatial_shape, actual_shape)
        return data_array

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return self.sp_resample.inverse(data)


class Orientation(InvertibleTransform):
    """
    Change the input image's orientation into the specified based on `axcodes`.
    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        axcodes: str | None = None,
        as_closest_canonical: bool = False,
        labels: Sequence[tuple[str, str]] | None = (("L", "R"), ("P", "A"), ("I", "S")),
    ) -> None:
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.

        Raises:
            ValueError: When ``axcodes=None`` and ``as_closest_canonical=True``. Incompatible values.

        See Also: `nibabel.orientations.ornt2axcodes`.

        """
        if axcodes is None and not as_closest_canonical:
            raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
        if axcodes is not None and as_closest_canonical:
            warnings.warn("using as_closest_canonical=True, axcodes ignored.")
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels

    def __call__(self, data_array: torch.Tensor) -> torch.Tensor:
        """
        If input type is `MetaTensor`, original affine is extracted with `data_array.affine`.
        If input type is `torch.Tensor`, original affine is assumed to be identity.

        Args:
            data_array: in shape (num_channels, H[, W, ...]).

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``axcodes`` spatiality differs from ``data_array``.

        Returns:
            data_array [reoriented in `self.axcodes`]. Output type will be `MetaTensor`
                unless `get_track_meta() == False`, in which case it will be
                `torch.Tensor`.

        """
        spatial_shape = data_array.shape[1:]
        sr = len(spatial_shape)
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        affine_: np.ndarray
        affine_np: np.ndarray
        if isinstance(data_array, MetaTensor):
            affine_np, *_ = convert_data_type(data_array.affine, np.ndarray)
            affine_ = to_affine_nd(sr, affine_np)
        else:
            warnings.warn("`data_array` is not of type `MetaTensor, assuming affine to be identity.")
            # default to identity
            affine_np = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)

        src = nib.io_orientation(affine_)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            if self.axcodes is None:
                raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
            if sr < len(self.axcodes):
                warnings.warn(
                    f"axcodes ('{self.axcodes}') length is smaller than the number of input spatial dimensions D={sr}.\n"
                    f"{self.__class__.__name__}: input spatial shape is {spatial_shape}, num. channels is {data_array.shape[0]},"
                    "please make sure the input is in the channel-first format."
                )
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
            if len(dst) < sr:
                raise ValueError(
                    f"axcodes must match data_array spatially, got axcodes={len(self.axcodes)}D data_array={sr}D"
                )
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        new_affine = affine_ @ nib.orientations.inv_ornt_aff(spatial_ornt, spatial_shape)

        # convert to MetaTensor if necessary
        data_array = convert_to_tensor(data_array, track_meta=get_track_meta())

        spatial_ornt[:, 0] += 1  # skip channel dim
        spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
        axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
        if axes:
            data_array = torch.flip(data_array, dims=axes)
        full_transpose = np.arange(len(data_array.shape))
        full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
        if not np.all(full_transpose == np.arange(len(data_array.shape))):
            data_array = data_array.permute(full_transpose.tolist())

        new_affine = to_affine_nd(affine_np, new_affine)
        new_affine, *_ = convert_data_type(new_affine, torch.Tensor, dtype=torch.float32, device=data_array.device)

        if get_track_meta():
            self.update_meta(data_array, new_affine)
            self.push_transform(data_array, extra_info={"original_affine": affine_np})
        return data_array

    def update_meta(self, img, new_affine):
        img.affine = new_affine

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # Create inverse transform
        orig_affine = transform[TraceKeys.EXTRA_INFO]["original_affine"]
        orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
        inverse_transform = Orientation(axcodes=orig_axcodes, as_closest_canonical=False, labels=self.labels)
        # Apply inverse
        with inverse_transform.trace_transform(False):
            data = inverse_transform(data)

        return data


class Flip(InvertibleTransform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    See `torch.flip` documentation for additional details:
    https://pytorch.org/docs/stable/generated/torch.flip.html

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, spatial_axis: Sequence[int] | int | None = None) -> None:
        self.spatial_axis = spatial_axis

    def update_meta(self, img, shape, axes):
        # shape and axes include the channel dim
        affine = img.affine
        mat = convert_to_dst_type(torch.eye(len(affine)), affine)[0]
        for axis in axes:
            sp = axis - 1
            mat[sp, sp], mat[sp, -1] = mat[sp, sp] * -1, shape[axis] - 1
        img.affine = affine @ mat

    def forward_image(self, img, axes) -> torch.Tensor:
        return torch.flip(img, axes)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ])
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        axes = map_spatial_axes(img.ndim, self.spatial_axis)
        out = self.forward_image(img, axes)
        if get_track_meta():
            self.update_meta(out, out.shape, axes)
            self.push_transform(out)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        self.pop_transform(data)
        flipper = Flip(spatial_axis=self.spatial_axis)
        with flipper.trace_transform(False):
            return flipper(data)


class Resize(InvertibleTransform):
    """
    Resize the input image to given spatial size (with scaling, not cropping/padding).
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        size_mode: str = "all",
        mode: str = InterpolateMode.AREA,
        align_corners: bool | None = None,
        anti_aliasing: bool = False,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
    ) -> None:
        self.size_mode = look_up_option(size_mode, ["all", "longest"])
        self.spatial_size = spatial_size
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        align_corners: bool | None = None,
        anti_aliasing: bool | None = None,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            anti_aliasing: bool, optional
                Whether to apply a Gaussian filter to smooth the image prior
                to downsampling. It is crucial to filter when downsampling
                the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
            anti_aliasing_sigma: {float, tuple of floats}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1. For the up-size case, s < 1, no
                anti-aliasing is performed prior to rescaling.

        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

        """
        anti_aliasing = self.anti_aliasing if anti_aliasing is None else anti_aliasing
        anti_aliasing_sigma = self.anti_aliasing_sigma if anti_aliasing_sigma is None else anti_aliasing_sigma

        input_ndim = img.ndim - 1  # spatial ndim
        if self.size_mode == "all":
            output_ndim = len(ensure_tuple(self.spatial_size))
            if output_ndim > input_ndim:
                input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
                img = img.reshape(input_shape)
            elif output_ndim < input_ndim:
                raise ValueError(
                    "len(spatial_size) must be greater or equal to img spatial dimensions, "
                    f"got spatial_size={output_ndim} img={input_ndim}."
                )
            spatial_size_ = fall_back_tuple(self.spatial_size, img.shape[1:])
        else:  # for the "longest" mode
            img_size = img.shape[1:]
            if not isinstance(self.spatial_size, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
            scale = self.spatial_size / max(img_size)
            spatial_size_ = tuple(int(round(s * scale)) for s in img_size)

        original_sp_size = img.shape[1:]
        _mode = look_up_option(self.mode if mode is None else mode, InterpolateMode)
        _align_corners = self.align_corners if align_corners is None else align_corners
        if tuple(img.shape[1:]) == spatial_size_:  # spatial shape is already the desired
            img = convert_to_tensor(img, track_meta=get_track_meta())

            return self._post_process(img, original_sp_size, spatial_size_, _mode, _align_corners, input_ndim)
        img_ = convert_to_tensor(img, dtype=torch.float, track_meta=False)

        if anti_aliasing and any(x < y for x, y in zip(spatial_size_, img_.shape[1:])):
            factors = torch.div(torch.Tensor(list(img_.shape[1:])), torch.Tensor(spatial_size_))
            if anti_aliasing_sigma is None:
                # if sigma is not given, use the default sigma in skimage.transform.resize
                anti_aliasing_sigma = torch.maximum(torch.zeros(factors.shape), (factors - 1) / 2).tolist()
            else:
                # if sigma is given, use the given value for downsampling axis
                anti_aliasing_sigma = list(ensure_tuple_rep(anti_aliasing_sigma, len(spatial_size_)))
                for axis in range(len(spatial_size_)):
                    anti_aliasing_sigma[axis] = anti_aliasing_sigma[axis] * int(factors[axis] > 1)
            anti_aliasing_filter = GaussianSmooth(sigma=anti_aliasing_sigma)
            img_ = convert_to_tensor(anti_aliasing_filter(img_), track_meta=False)

        img = convert_to_tensor(img, track_meta=get_track_meta())
        resized = torch.nn.functional.interpolate(
            input=img_.unsqueeze(0), size=spatial_size_, mode=_mode, align_corners=_align_corners
        )
        out, *_ = convert_to_dst_type(resized.squeeze(0), img)
        return self._post_process(out, original_sp_size, spatial_size_, _mode, _align_corners, input_ndim)

    def _post_process(self, img: torch.Tensor, orig_size, sp_size, mode, align_corners, ndim) -> torch.Tensor:
        if get_track_meta():
            self.update_meta(img, orig_size, sp_size)
            self.push_transform(
                img,
                orig_size=orig_size,
                extra_info={
                    "mode": mode,
                    "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                    "new_dim": len(orig_size) - ndim,  # additional dims appended
                },
            )
        return img

    def update_meta(self, img, spatial_size, new_spatial_size):
        affine = convert_to_tensor(img.affine, track_meta=False)
        img.affine = scale_affine(affine, spatial_size, new_spatial_size)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        orig_size = transform[TraceKeys.ORIG_SIZE]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        xform = Resize(
            spatial_size=orig_size, mode=mode, align_corners=None if align_corners == TraceKeys.NONE else align_corners
        )
        with xform.trace_transform(False):
            data = xform(data)
        for _ in range(transform[TraceKeys.EXTRA_INFO]["new_dim"]):
            data = data.squeeze(-1)  # remove the additional dims
        return data


class Rotate(InvertibleTransform):
    """
    Rotates an input image by given angle using :py:class:`monai.networks.layers.AffineTransform`.

    Args:
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        angle: Sequence[float] | float,
        keep_size: bool = True,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike | torch.dtype = torch.float32,
    ) -> None:
        self.angle = angle
        self.keep_size = keep_size
        self.mode: str = look_up_option(mode, GridSampleMode)
        self.padding_mode: str = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.

        Raises:
            ValueError: When ``img`` spatially is not one of [2D, 3D].

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)

        im_shape = np.asarray(img.shape[1:])  # spatial dimensions
        input_ndim = len(im_shape)
        if input_ndim not in (2, 3):
            raise ValueError(f"Unsupported image dimension: {input_ndim}, available options are [2, 3].")
        _angle = ensure_tuple_rep(self.angle, 1 if input_ndim == 2 else 3)
        transform = create_rotate(input_ndim, _angle)
        shift = create_translate(input_ndim, ((im_shape - 1) / 2).tolist())
        if self.keep_size:
            output_shape = im_shape
        else:
            corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape(
                (len(im_shape), -1)
            )
            corners = transform[:-1, :-1] @ corners  # type: ignore
            output_shape = np.asarray(corners.ptp(axis=1) + 0.5, dtype=int)
        shift_1 = create_translate(input_ndim, (-(output_shape - 1) / 2).tolist())
        transform = shift @ transform @ shift_1

        img_t = img.to(_dtype)
        transform_t, *_ = convert_to_dst_type(transform, img_t)
        _mode = look_up_option(mode or self.mode, GridSampleMode)
        _padding_mode = look_up_option(padding_mode or self.padding_mode, GridSamplePadMode)
        _align_corners = self.align_corners if align_corners is None else align_corners
        xform = AffineTransform(
            normalized=False,
            mode=_mode,
            padding_mode=_padding_mode,
            align_corners=_align_corners,
            reverse_indexing=True,
        )
        output: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=output_shape).float().squeeze(0)
        out, *_ = convert_to_dst_type(output, dst=img, dtype=output.dtype)
        if get_track_meta():
            self.update_meta(out, transform_t)
            self.push_transform(
                out,
                orig_size=img_t.shape[1:],
                extra_info={
                    "rot_mat": transform,
                    "mode": _mode,
                    "padding_mode": _padding_mode,
                    "align_corners": _align_corners if _align_corners is not None else TraceKeys.NONE,
                    "dtype": str(_dtype)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
                },
            )
        return out

    def update_meta(self, img, rotate_mat):
        affine = convert_to_tensor(img.affine, track_meta=False)
        mat = to_affine_nd(len(affine) - 1, rotate_mat)
        img.affine = affine @ convert_to_dst_type(mat, affine)[0]

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        fwd_rot_mat = transform[TraceKeys.EXTRA_INFO]["rot_mat"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        dtype = transform[TraceKeys.EXTRA_INFO]["dtype"]
        inv_rot_mat = linalg_inv(convert_to_numpy(fwd_rot_mat))

        xform = AffineTransform(
            normalized=False,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False if align_corners == TraceKeys.NONE else align_corners,
            reverse_indexing=True,
        )
        img_t: torch.Tensor = convert_data_type(data, MetaTensor, dtype=dtype)[0]
        transform_t, *_ = convert_to_dst_type(inv_rot_mat, img_t)
        sp_size = transform[TraceKeys.ORIG_SIZE]
        out: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=sp_size).float().squeeze(0)
        out = convert_to_dst_type(out, dst=data, dtype=out.dtype)[0]
        if isinstance(data, MetaTensor):
            self.update_meta(out, transform_t)
        return out


class Zoom(InvertibleTransform):
    """
    Zooms an ND image using :py:class:`torch.nn.functional.interpolate`.
    For details, please see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

    Different from :py:class:`monai.transforms.resize`, this transform takes scaling factors
    as input, and provides an option of preserving the input spatial size.

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"edge"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        keep_size: Should keep original size (padding/slicing if needed), default is True.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        zoom: Sequence[float] | float,
        mode: str = InterpolateMode.AREA,
        padding_mode: str = NumpyPadMode.EDGE,
        align_corners: bool | None = None,
        keep_size: bool = True,
        **kwargs,
    ) -> None:
        self.zoom = zoom
        self.mode: InterpolateMode = InterpolateMode(mode)
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.keep_size = keep_size
        self.kwargs = kwargs

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"edge"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = img.to(torch.float32)

        _zoom = ensure_tuple_rep(self.zoom, img.ndim - 1)  # match the spatial image dim
        _mode = look_up_option(self.mode if mode is None else mode, InterpolateMode).value
        _align_corners = self.align_corners if align_corners is None else align_corners
        _padding_mode = padding_mode or self.padding_mode

        zoomed: NdarrayOrTensor = torch.nn.functional.interpolate(
            recompute_scale_factor=True,
            input=img_t.unsqueeze(0),
            scale_factor=list(_zoom),
            mode=_mode,
            align_corners=_align_corners,
        )
        zoomed = zoomed.squeeze(0)
        orig_size, z_size = img_t.shape, zoomed.shape

        out, *_ = convert_to_dst_type(zoomed, dst=img)
        if get_track_meta():
            self.update_meta(out, orig_size[1:], z_size[1:])
        do_pad_crop = self.keep_size and not np.allclose(orig_size, z_size)
        if do_pad_crop:
            _pad_crop = ResizeWithPadOrCrop(spatial_size=img_t.shape[1:], mode=_padding_mode)
            out = _pad_crop(out)
        if get_track_meta():
            padcrop_xform = self.pop_transform(out, check=False) if do_pad_crop else {}
            self.push_transform(
                out,
                orig_size=orig_size[1:],
                extra_info={
                    "mode": _mode,
                    "align_corners": _align_corners if _align_corners is not None else TraceKeys.NONE,
                    "do_padcrop": do_pad_crop,
                    "padcrop": padcrop_xform,
                },
            )
        return out

    def update_meta(self, img, spatial_size, new_spatial_size):
        affine = convert_to_tensor(img.affine, track_meta=False)
        img.affine = scale_affine(affine, spatial_size, new_spatial_size)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        if transform[TraceKeys.EXTRA_INFO]["do_padcrop"]:
            orig_size = transform[TraceKeys.ORIG_SIZE]
            pad_or_crop = ResizeWithPadOrCrop(spatial_size=orig_size, mode="edge")
            padcrop_xform = transform[TraceKeys.EXTRA_INFO]["padcrop"]
            padcrop_xform[TraceKeys.EXTRA_INFO]["pad_info"][TraceKeys.ID] = TraceKeys.NONE
            padcrop_xform[TraceKeys.EXTRA_INFO]["crop_info"][TraceKeys.ID] = TraceKeys.NONE
            # this uses inverse because spatial_size // 2 in the forward pass of center crop may cause issues
            data = pad_or_crop.inverse_transform(data, padcrop_xform)  # type: ignore
        # Create inverse transform
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        inverse_transform = Resize(spatial_size=transform[TraceKeys.ORIG_SIZE])
        # Apply inverse
        with inverse_transform.trace_transform(False):
            out = inverse_transform(
                data, mode=mode, align_corners=None if align_corners == TraceKeys.NONE else align_corners
            )
        return out


class Rotate90(InvertibleTransform):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    See `torch.rot90` for additional details:
    https://pytorch.org/docs/stable/generated/torch.rot90.html#torch-rot90.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, k: int = 1, spatial_axes: tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                If axis is negative it counts from the last to the first axis.
        """
        self.k = k
        spatial_axes_: tuple[int, int] = ensure_tuple(spatial_axes)  # type: ignore
        if len(spatial_axes_) != 2:
            raise ValueError("spatial_axes must be 2 int numbers to indicate the axes to rotate 90 degrees.")
        self.spatial_axes = spatial_axes_

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        axes = map_spatial_axes(img.ndim, self.spatial_axes)
        ori_shape = img.shape[1:]
        out: NdarrayOrTensor = torch.rot90(img, self.k, axes)
        out = convert_to_dst_type(out, img)[0]
        if get_track_meta():
            self.update_meta(out, ori_shape, out.shape[1:], axes, self.k)
            self.push_transform(out, extra_info={"axes": [d - 1 for d in axes], "k": self.k})  # compensate spatial dim
        return out

    def update_meta(self, img, spatial_size, new_spatial_size, axes, k):
        affine = convert_data_type(img.affine, torch.Tensor)[0]
        r, sp_r = len(affine) - 1, len(spatial_size)
        mat = to_affine_nd(r, create_translate(sp_r, [-float(d - 1) / 2 for d in new_spatial_size]))
        s = -1.0 if int(axes[0]) - int(axes[1]) in (-1, 2) else 1.0
        if sp_r == 2:
            rot90 = to_affine_nd(r, create_rotate(sp_r, [s * np.pi / 2]))
        else:
            idx = {1, 2, 3} - set(axes)
            angle: list[float] = [0, 0, 0]
            angle[idx.pop() - 1] = s * np.pi / 2
            rot90 = to_affine_nd(r, create_rotate(sp_r, angle))
        for _ in range(k):
            mat = rot90 @ mat
        mat = to_affine_nd(r, create_translate(sp_r, [float(d - 1) / 2 for d in spatial_size])) @ mat
        img.affine = affine @ convert_to_dst_type(mat, affine)[0]

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        axes = transform[TraceKeys.EXTRA_INFO]["axes"]
        k = transform[TraceKeys.EXTRA_INFO]["k"]
        inv_k = 4 - k % 4
        xform = Rotate90(k=inv_k, spatial_axes=axes)
        with xform.trace_transform(False):
            return xform(data)


class RandRotate90(RandomizableTransform, InvertibleTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    backend = Rotate90.backend

    def __init__(self, prob: float = 0.1, max_k: int = 3, spatial_axes: tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        RandomizableTransform.__init__(self, prob)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._rand_k = self.R.randint(self.max_k) + 1

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize()

        if self._do_transform:
            out = Rotate90(self._rand_k, self.spatial_axes)(img)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta())

        if get_track_meta():
            maybe_rot90_info = self.pop_transform(out, check=False) if self._do_transform else {}
            self.push_transform(out, extra_info=maybe_rot90_info)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        rotate_xform = xform_info[TraceKeys.EXTRA_INFO]
        return Rotate90().inverse_transform(data, rotate_xform)


class RandRotate(RandomizableTransform, InvertibleTransform):
    """
    Randomly rotate the input arrays.

    Args:
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y). only work for 3D data.
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z). only work for 3D data.
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
    """

    backend = Rotate.backend

    def __init__(
        self,
        range_x: tuple[float, float] | float = 0.0,
        range_y: tuple[float, float] | float = 0.0,
        range_z: tuple[float, float] | float = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike | torch.dtype = np.float32,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.keep_size = keep_size
        self.mode: str = look_up_option(mode, GridSampleMode)
        self.padding_mode: str = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        randomize: bool = True,
    ):
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize()

        if self._do_transform:
            rotator = Rotate(
                angle=self.x if img.ndim == 3 else (self.x, self.y, self.z),
                keep_size=self.keep_size,
                mode=look_up_option(mode or self.mode, GridSampleMode),
                padding_mode=look_up_option(padding_mode or self.padding_mode, GridSamplePadMode),
                align_corners=self.align_corners if align_corners is None else align_corners,
                dtype=dtype or self.dtype or img.dtype,
            )
            out = rotator(img)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        if get_track_meta():
            rot_info = self.pop_transform(out, check=False) if self._do_transform else {}
            self.push_transform(out, extra_info=rot_info)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        return Rotate(0).inverse_transform(data, xform_info[TraceKeys.EXTRA_INFO])


class RandFlip(RandomizableTransform, InvertibleTransform):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1, spatial_axis: Sequence[int] | int | None = None) -> None:
        RandomizableTransform.__init__(self, prob)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize(None)
        out = self.flipper(img) if self._do_transform else img
        out = convert_to_tensor(out, track_meta=get_track_meta())
        if get_track_meta():
            xform_info = self.pop_transform(out, check=False) if self._do_transform else {}
            self.push_transform(out, extra_info=xform_info)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        if not transform[TraceKeys.DO_TRANSFORM]:
            return data
        data.applied_operations.append(transform[TraceKeys.EXTRA_INFO])  # type: ignore
        return self.flipper.inverse(data)


class RandAxisFlip(RandomizableTransform, InvertibleTransform):
    """
    Randomly select a spatial axis and flip along it.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.

    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1) -> None:
        RandomizableTransform.__init__(self, prob)
        self._axis: int | None = None
        self.flipper = Flip(spatial_axis=self._axis)

    def randomize(self, data: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._axis = self.R.randint(data.ndim - 1)

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ])
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize(data=img)

        if self._do_transform:
            self.flipper.spatial_axis = self._axis
            out = self.flipper(img)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta())
        if get_track_meta():
            xform = self.pop_transform(out, check=False) if self._do_transform else {}
            xform["axes"] = self._axis
            self.push_transform(out, extra_info=xform)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        if not transform[TraceKeys.DO_TRANSFORM]:
            return data
        flipper = Flip(spatial_axis=transform[TraceKeys.EXTRA_INFO]["axes"])
        with flipper.trace_transform(False):
            return flipper(data)


class RandZoom(RandomizableTransform, InvertibleTransform):
    """
    Randomly zooms input arrays with given probability within given zoom range.

    Args:
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        keep_size: Should keep original size (pad if needed), default is True.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Zoom.backend

    def __init__(
        self,
        prob: float = 0.1,
        min_zoom: Sequence[float] | float = 0.9,
        max_zoom: Sequence[float] | float = 1.1,
        mode: str = InterpolateMode.AREA,
        padding_mode: str = NumpyPadMode.EDGE,
        align_corners: bool | None = None,
        keep_size: bool = True,
        **kwargs,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.min_zoom = ensure_tuple(min_zoom)
        self.max_zoom = ensure_tuple(max_zoom)
        if len(self.min_zoom) != len(self.max_zoom):
            raise ValueError(
                f"min_zoom and max_zoom must have same length, got {len(self.min_zoom)} and {len(self.max_zoom)}."
            )
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.keep_size = keep_size
        self.kwargs = kwargs

        self._zoom: Sequence[float] = [1.0]

    def randomize(self, img: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._zoom = [self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom)]
        if len(self._zoom) == 1:
            # to keep the spatial shape ratio, use same random zoom factor for all dims
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 1)
        elif len(self._zoom) == 2 and img.ndim > 3:
            # if 2 zoom factors provided for 3D data, use the first factor for H and W dims, second factor for D dim
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 2) + ensure_tuple(self._zoom[-1])

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``,
                ``"area"``}, the interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            randomize: whether to execute `randomize()` function first, default to True.

        """
        # match the spatial image dim
        if randomize:
            self.randomize(img=img)

        if not self._do_transform:
            out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        else:
            out = Zoom(
                self._zoom,
                keep_size=self.keep_size,
                mode=look_up_option(mode or self.mode, InterpolateMode),
                padding_mode=padding_mode or self.padding_mode,
                align_corners=self.align_corners if align_corners is None else align_corners,
                **self.kwargs,
            )(img)
        if get_track_meta():
            z_info = self.pop_transform(out, check=False) if self._do_transform else {}
            self.push_transform(out, extra_info=z_info)
        return out  # type: ignore

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        return Zoom(self._zoom).inverse_transform(data, xform_info[TraceKeys.EXTRA_INFO])


class AffineGrid(Transform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
            Defaults to no rotation.
        shear_params: shearing factors for affine matrix, take a 3D affine as example::

            [
                [1.0, params[0], params[1], 0.0],
                [params[2], 1.0, params[3], 0.0],
                [params[4], params[5], 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

            a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
        translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
            pixel/voxel relative to the center of the input image. Defaults to no translation.
        scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
            a tuple of 3 floats for 3D. Defaults to `1.0`.
        dtype: data type for the grid computation. Defaults to ``float32``.
            If ``None``, use the data type of input data (if `grid` is provided).
        device: device on which the tensor will be allocated, if a new grid is generated.
        affine: If applied, ignore the params (`rotate_params`, etc.) and use the
            supplied matrix. Should be square with each side = num of image spatial
            dimensions + 1.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        affine: NdarrayOrTensor | None = None,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.device = device
        self.dtype = dtype
        self.affine = affine

    def __call__(
        self, spatial_size: Sequence[int] | None = None, grid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The grid can be initialized with a `spatial_size` parameter, or provided directly as `grid`.
        Therefore, either `spatial_size` or `grid` must be provided.
        When initialising from `spatial_size`, the backend "torch" will be used.

        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:  # create grid from spatial_size
            if spatial_size is None:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")
            grid_ = create_grid(spatial_size, device=self.device, backend="torch", dtype=self.dtype)
        else:
            grid_ = grid
        _dtype = self.dtype or grid_.dtype
        grid_: torch.Tensor = convert_to_tensor(grid_, dtype=_dtype, track_meta=get_track_meta())  # type: ignore
        _b = TransformBackends.TORCH
        _device = grid_.device  # type: ignore
        affine: NdarrayOrTensor
        if self.affine is None:
            spatial_dims = len(grid_.shape) - 1
            affine = torch.eye(spatial_dims + 1, device=_device)
            if self.rotate_params:
                affine = affine @ create_rotate(spatial_dims, self.rotate_params, device=_device, backend=_b)
            if self.shear_params:
                affine = affine @ create_shear(spatial_dims, self.shear_params, device=_device, backend=_b)
            if self.translate_params:
                affine = affine @ create_translate(spatial_dims, self.translate_params, device=_device, backend=_b)
            if self.scale_params:
                affine = affine @ create_scale(spatial_dims, self.scale_params, device=_device, backend=_b)
        else:
            affine = self.affine

        affine = to_affine_nd(len(grid_) - 1, affine)
        affine = convert_to_tensor(affine, device=grid_.device, dtype=grid_.dtype, track_meta=False)  # type: ignore
        grid_ = (affine @ grid_.reshape((grid_.shape[0], -1))).reshape([-1] + list(grid_.shape[1:]))
        return grid_, affine  # type: ignore


class RandAffineGrid(Randomizable, Transform):
    """
    Generate randomised affine grid.

    """

    backend = AffineGrid.backend

    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxels to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            device: device to store the output grid data.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`

        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: list[float] | None = None
        self.shear_params: list[float] | None = None
        self.translate_params: list[float] | None = None
        self.scale_params: list[float] | None = None

        self.device = device
        self.affine: torch.Tensor | None = torch.eye(4, dtype=torch.float64)

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Any | None = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(
        self, spatial_size: Sequence[int] | None = None, grid: NdarrayOrTensor | None = None, randomize: bool = True
    ) -> torch.Tensor:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            randomize: boolean as to whether the grid parameters governing the grid should be randomized.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        if randomize:
            self.randomize()
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            device=self.device,
        )
        _grid: torch.Tensor
        _grid, self.affine = affine_grid(spatial_size, grid)  # type: ignore
        return _grid

    def get_transformation_matrix(self) -> torch.Tensor | None:
        """Get the most recently applied transformation matrix"""
        return self.affine


class RandDeformGrid(Randomizable, Transform):
    """
    Generate random deformation grid.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self, spacing: Sequence[float] | float, magnitude_range: tuple[float, float], device: torch.device | None = None
    ) -> None:
        """
        Args:
            spacing: spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range: the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            device: device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range

        self.rand_mag = 1.0
        self.random_offset: np.ndarray
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        self.random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size))).astype(np.float32, copy=False)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]) -> torch.Tensor:
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing, device=self.device, backend="torch")
        self.randomize(control_grid.shape[1:])
        _offset, *_ = convert_to_dst_type(self.rand_mag * self.random_offset, control_grid)
        control_grid[: len(spatial_size)] += _offset
        return control_grid  # type: ignore


class Resample(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        norm_coords: bool = True,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float64,
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            norm_coords: whether to normalize the coordinates from `[-(size-1)/2, (size-1)/2]` to
                `[0, size - 1]` (for ``monai/csrc`` implementation) or
                `[-1, 1]` (for torch ``grid_sample`` implementation) to be compatible with the underlying
                resampling API.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.

        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm_coords = norm_coords
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        grid: torch.Tensor | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        dtype: DtypeLike = None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
                if ``norm_coords`` is True, the grid values must be in `[-(size-1)/2, (size-1)/2]`.
                if ``USE_COMPILED=True`` and ``norm_coords=False``, grid values must be in `[0, size-1]`.
                if ``USE_COMPILED=False`` and ``norm_coords=False``, grid values must be in `[-1, 1]`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                To be compatible with other modules, the output data type is always `float32`.

        See also:
            :py:const:`monai.config.USE_COMPILED`
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if grid is None:
            return img
        _device = img.device if isinstance(img, torch.Tensor) else self.device
        _dtype = dtype or self.dtype or img.dtype
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=_dtype, device=_device)
        grid_t, *_ = convert_to_dst_type(grid, img_t, dtype=grid.dtype, wrap_sequence=True)
        grid_t = grid_t.clone(memory_format=torch.contiguous_format)

        if self.norm_coords:
            grid_t[-1] = where(grid_t[-1] != 0, grid_t[-1], 1.0)  # type: ignore
        sr = min(len(img_t.shape[1:]), 3)

        _interp_mode = self.mode if mode is None else mode
        _padding_mode = self.padding_mode if padding_mode is None else padding_mode
        if look_up_option(str(_interp_mode), SplineMode, default=None) is not None:
            self._backend = TransformBackends.NUMPY
        else:
            self._backend = TransformBackends.TORCH

        if USE_COMPILED or self._backend == TransformBackends.NUMPY:
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                    grid_t[i] = (max(dim, 2) / 2.0 - 0.5 + grid_t[i]) / grid_t[-1:]
            grid_t = grid_t[:sr]
            if USE_COMPILED and self._backend == TransformBackends.TORCH:  # compiled is using torch backend param name
                grid_t = moveaxis(grid_t, 0, -1)  # type: ignore
                bound = 1 if _padding_mode == "reflection" else _padding_mode
                if _interp_mode == "bicubic":
                    interp = 3
                elif _interp_mode == "bilinear":
                    interp = 1
                else:
                    interp = GridSampleMode(_interp_mode)  # type: ignore
                out = grid_pull(
                    img_t.unsqueeze(0),
                    grid_t.unsqueeze(0).to(img_t),
                    bound=bound,
                    extrapolate=True,
                    interpolation=interp,
                )[0]
            elif self._backend == TransformBackends.NUMPY:
                is_cuda = img_t.is_cuda
                img_np = (convert_to_cupy if is_cuda else convert_to_numpy)(img_t, wrap_sequence=True)
                grid_np, *_ = convert_to_dst_type(grid_t, img_np, wrap_sequence=True)
                _map_coord = (cupy_ndi if is_cuda else np_ndi).map_coordinates
                out = (cupy if is_cuda else np).stack(
                    [
                        _map_coord(c, grid_np, order=int(_interp_mode), mode=look_up_option(_padding_mode, NdimageMode))
                        for c in img_np
                    ]
                )
                out = convert_to_dst_type(out, img_t)[0]
        else:
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                    grid_t[i] = 2.0 / (max(2, dim) - 1.0) * grid_t[i] / grid_t[-1:]
            index_ordering: list[int] = list(range(sr - 1, -1, -1))
            grid_t = moveaxis(grid_t[index_ordering], 0, -1)  # type: ignore
            out = torch.nn.functional.grid_sample(
                img_t.unsqueeze(0),
                grid_t.unsqueeze(0).to(img_t),
                mode=GridSampleMode(_interp_mode),
                padding_mode=GridSamplePadMode(_padding_mode),
                align_corners=True,
            )[0]
        out_val, *_ = convert_to_dst_type(out, dst=img, dtype=np.float32)
        return out_val


class Affine(InvertibleTransform):
    """
    Transform ``img`` given the affine parameters.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = list(set(AffineGrid.backend) & set(Resample.backend))

    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        affine: NdarrayOrTensor | None = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        normalized: bool = False,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        image_only: bool = False,
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: shearing factors for affine matrix, take a 3D affine as example::

                [
                    [1.0, params[0], params[1], 0.0],
                    [params[2], 1.0, params[3], 0.0],
                    [params[4], params[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
                a tuple of 3 floats for 3D. Defaults to `1.0`.
            affine: If applied, ignore the params (`rotate_params`, etc.) and use the
                supplied matrix. Should be square with each side = num of image spatial
                dimensions + 1.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            normalized: indicating whether the provided `affine` is defined to include a normalization
                transform converting the coordinates from `[-(size-1)/2, (size-1)/2]` (defined in ``create_grid``) to
                `[0, size - 1]` or `[-1, 1]` in order to be compatible with the underlying resampling API.
                If `normalized=False`, additional coordinate normalization will be applied before resampling.
                See also: :py:func:`monai.networks.utils.normalize_transform`.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``float32``.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.
            image_only: if True return only the image volume, otherwise return (image, affine).

        """
        self.affine_grid = AffineGrid(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            affine=affine,
            dtype=dtype,
            device=device,
        )
        self.image_only = image_only
        self.norm_coord = not normalized
        self.resampler = Resample(norm_coords=self.norm_coord, device=device, dtype=dtype)
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, NdarrayOrTensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_size = img.shape[1:]
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img_size)
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        grid, affine = self.affine_grid(spatial_size=sp_size)
        out = self.resampler(img, grid=grid, mode=_mode, padding_mode=_padding_mode)
        if not isinstance(out, MetaTensor):
            return out if self.image_only else (out, affine)
        if get_track_meta():
            out.meta = img.meta  # type: ignore
            self.update_meta(out, affine, img_size, sp_size)
            self.push_transform(
                out, orig_size=img_size, extra_info={"affine": affine, "mode": _mode, "padding_mode": _padding_mode}
            )
        return out if self.image_only else (out, affine)

    @classmethod
    def compute_w_affine(cls, affine, mat, img_size, sp_size):
        r = len(affine) - 1
        mat = to_affine_nd(r, mat)
        shift_1 = create_translate(r, [float(d - 1) / 2 for d in img_size[:r]])
        shift_2 = create_translate(r, [-float(d - 1) / 2 for d in sp_size[:r]])
        mat = shift_1 @ convert_data_type(mat, np.ndarray)[0] @ shift_2
        return affine @ convert_to_dst_type(mat, affine)[0]

    def update_meta(self, img, mat, img_size, sp_size):
        affine = convert_data_type(img.affine, torch.Tensor)[0]
        img.affine = Affine.compute_w_affine(affine, mat, img_size, sp_size)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        orig_size = transform[TraceKeys.ORIG_SIZE]
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        inv_affine = linalg_inv(convert_to_numpy(fwd_affine))
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]

        affine_grid = AffineGrid(affine=inv_affine)
        grid, _ = affine_grid(orig_size)
        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        self.update_meta(out, inv_affine, data.shape[1:], orig_size)
        return out


class RandAffine(RandomizableTransform, InvertibleTransform):
    """
    Random affine transform.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Affine.backend

    def __init__(
        self,
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``bilinear``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``reflection``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        RandomizableTransform.__init__(self, prob)

        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.cache_grid = cache_grid
        self._cached_grid = self._init_identity_cache()
        self.mode = mode
        self.padding_mode: str = padding_mode

    def _init_identity_cache(self):
        """
        Create cache of the identity grid if cache_grid=True and spatial_size is known.
        """
        if self.spatial_size is None:
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size, please specify 'spatial_size'."
                )
            return None
        _sp_size = ensure_tuple(self.spatial_size)
        _ndim = len(_sp_size)
        if _sp_size != fall_back_tuple(_sp_size, [1] * _ndim) or _sp_size != fall_back_tuple(_sp_size, [2] * _ndim):
            # dynamic shape because it falls back to different outcomes
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size "
                    f"'spatial_size={self.spatial_size}', please specify 'spatial_size'."
                )
            return None
        return create_grid(spatial_size=_sp_size, device=self.rand_affine_grid.device, backend="torch")

    def get_identity_grid(self, spatial_size: Sequence[int]):
        """
        Return a cached or new identity grid depends on the availability.

        Args:
            spatial_size: non-dynamic spatial size
        """
        ndim = len(spatial_size)
        if spatial_size != fall_back_tuple(spatial_size, [1] * ndim) or spatial_size != fall_back_tuple(
            spatial_size, [2] * ndim
        ):
            raise RuntimeError(f"spatial_size should not be dynamic, got {spatial_size}.")
        return (
            create_grid(spatial_size=spatial_size, device=self.rand_affine_grid.device, backend="torch")
            if self._cached_grid is None
            else self._cached_grid
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAffine:
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
        grid=None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            randomize: whether to execute `randomize()` function first, default to True.
            grid: precomputed grid to be used (mainly to accelerate `RandAffined`).

        """
        if randomize:
            self.randomize()
        # if not doing transform and spatial size doesn't change, nothing to do
        # except convert to float and device
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        do_resampling = self._do_transform or (sp_size != ensure_tuple(img.shape[1:]))
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if not do_resampling:
            out: torch.Tensor = convert_data_type(img, dtype=torch.float32, device=self.resampler.device)[0]
        else:
            if grid is None:
                grid = self.get_identity_grid(sp_size)
                if self._do_transform:
                    grid = self.rand_affine_grid(grid=grid, randomize=randomize)
            out = self.resampler(img=img, grid=grid, mode=_mode, padding_mode=_padding_mode)
        mat = self.rand_affine_grid.get_transformation_matrix()
        out = convert_to_tensor(out, track_meta=get_track_meta())
        if get_track_meta():
            self.push_transform(
                out,
                orig_size=img.shape[1:],
                extra_info={
                    "affine": mat,
                    "mode": _mode,
                    "padding_mode": _padding_mode,
                    "do_resampling": do_resampling,
                },
            )
            self.update_meta(out, mat, img.shape[1:], sp_size)
        return out

    def update_meta(self, img, mat, img_size, sp_size):
        affine = convert_data_type(img.affine, torch.Tensor)[0]
        img.affine = Affine.compute_w_affine(affine, mat, img_size, sp_size)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # if transform was not performed nothing to do.
        if not transform[TraceKeys.EXTRA_INFO]["do_resampling"]:
            return data
        orig_size = transform[TraceKeys.ORIG_SIZE]
        orig_size = fall_back_tuple(orig_size, data.shape[1:])
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        inv_affine = linalg_inv(convert_to_numpy(fwd_affine))
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]
        affine_grid = AffineGrid(affine=inv_affine)
        grid, _ = affine_grid(orig_size)

        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        self.update_meta(out, inv_affine, data.shape[1:], orig_size)
        return out


class Rand2DElastic(RandomizableTransform):
    """
    Random elastic deformation and affine in 2D.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Resample.backend

    def __init__(
        self,
        spacing: tuple[float, float] | float,
        magnitude_range: tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: tuple[int, int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            spacing : distance in between the control points.
            magnitude_range: the random offsets will be generated from ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D) for affine matrix, take a 2D affine as example::

                    [
                        [1.0, params[0], 0.0],
                        [params[1], 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        RandomizableTransform.__init__(self, prob)
        self.deform_grid = RandDeformGrid(spacing=spacing, magnitude_range=magnitude_range, device=device)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.device = device
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand2DElastic:
        self.deform_grid.set_random_state(seed, state)
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def set_device(self, device):
        self.deform_grid.device = device
        self.rand_affine_grid.device = device
        self.resampler.device = device
        self.device = device

    def randomize(self, spatial_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.deform_grid.randomize(spatial_size)
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: tuple[int, int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W),
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            randomize: whether to execute `randomize()` function first, default to True.
        """
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(spatial_size=sp_size)

        if self._do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = self.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(
                recompute_scale_factor=True,
                input=grid.unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            _device = img.device if isinstance(img, torch.Tensor) else self.device
            grid = cast(torch.Tensor, create_grid(spatial_size=sp_size, device=_device, backend="torch"))
        out: torch.Tensor = self.resampler(
            img,
            grid,
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        return out


class Rand3DElastic(RandomizableTransform):
    """
    Random elastic deformation and affine in 3D.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Resample.backend

    def __init__(
        self,
        sigma_range: tuple[float, float],
        magnitude_range: tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: tuple[int, int, int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 6 floats for 3D) for affine matrix, take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        RandomizableTransform.__init__(self, prob)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode
        self.device = device

        self.rand_offset: np.ndarray
        self.magnitude = 1.0
        self.sigma = 1.0

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand3DElastic:
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def set_device(self, device):
        self.rand_affine_grid.device = device
        self.resampler.device = device
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32, copy=False)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: tuple[int, int, int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W, D),
            spatial_size: specifying spatial 3D output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            randomize: whether to execute `randomize()` function first, default to True.
        """
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(grid_size=sp_size)

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        grid = create_grid(spatial_size=sp_size, device=_device, backend="torch")
        if self._do_transform:
            if self.rand_offset is None:
                raise RuntimeError("rand_offset is not initialized.")
            gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=_device)
            offset = torch.as_tensor(self.rand_offset, device=_device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        out: torch.Tensor = self.resampler(
            img,
            grid,  # type: ignore
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        return out


class GridDistortion(Transform):
    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        num_cells: tuple[int] | int,
        distort_steps: Sequence[Sequence[float]],
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
    ) -> None:
        """
        Grid distortion transform. Refer to:
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py

        Args:
            num_cells: number of grid cells on each dimension.
            distort_steps: This argument is a list of tuples, where each tuple contains the distort steps of the
                corresponding dimensions (in the order of H, W[, D]). The length of each tuple equals to `num_cells + 1`.
                Each value in the tuple represents the distort step of the related cell.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            device: device on which the tensor will be allocated.

        """
        self.resampler = Resample(mode=mode, padding_mode=padding_mode, device=device)
        self.num_cells = num_cells
        self.distort_steps = distort_steps
        self.device = device

    def __call__(
        self,
        img: torch.Tensor,
        distort_steps: Sequence[Sequence] | None = None,
        mode: str | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            distort_steps: This argument is a list of tuples, where each tuple contains the distort steps of the
                corresponding dimensions (in the order of H, W[, D]). The length of each tuple equals to `num_cells + 1`.
                Each value in the tuple represents the distort step of the related cell.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html

        """
        distort_steps = self.distort_steps if distort_steps is None else distort_steps
        if len(img.shape) != len(distort_steps) + 1:
            raise ValueError("the spatial size of `img` does not match with the length of `distort_steps`")

        all_ranges = []
        num_cells = ensure_tuple_rep(self.num_cells, len(img.shape) - 1)
        for dim_idx, dim_size in enumerate(img.shape[1:]):
            dim_distort_steps = distort_steps[dim_idx]
            ranges = torch.zeros(dim_size, dtype=torch.float32)
            cell_size = dim_size // num_cells[dim_idx]
            prev = 0
            for idx in range(num_cells[dim_idx] + 1):
                start = int(idx * cell_size)
                end = start + cell_size
                if end > dim_size:
                    end = dim_size
                    cur = dim_size
                else:
                    cur = prev + cell_size * dim_distort_steps[idx]
                ranges[start:end] = torch.linspace(prev, cur, end - start)
                prev = cur
            ranges = ranges - (dim_size - 1.0) / 2.0
            all_ranges.append(ranges)

        coords = meshgrid_ij(*all_ranges)
        grid = torch.stack([*coords, torch.ones_like(coords[0])])

        return self.resampler(img, grid=grid, mode=mode, padding_mode=padding_mode)


class RandGridDistortion(RandomizableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        num_cells: tuple[int] | int = 5,
        prob: float = 0.1,
        distort_limit: tuple[float, float] | float = (-0.03, 0.03),
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
    ) -> None:
        """
        Random grid distortion transform. Refer to:
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py

        Args:
            num_cells: number of grid cells on each dimension.
            prob: probability of returning a randomized grid distortion transform. Defaults to 0.1.
            distort_limit: range to randomly distort.
                If single number, distort_limit is picked from (-distort_limit, distort_limit).
                Defaults to (-0.03, 0.03).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            device: device on which the tensor will be allocated.

        """
        RandomizableTransform.__init__(self, prob)
        self.num_cells = num_cells
        if isinstance(distort_limit, (int, float)):
            self.distort_limit = (min(-distort_limit, distort_limit), max(-distort_limit, distort_limit))
        else:
            self.distort_limit = (min(distort_limit), max(distort_limit))
        self.distort_steps: Sequence[Sequence[float]] = ((1.0,),)
        self.grid_distortion = GridDistortion(
            num_cells=num_cells, distort_steps=self.distort_steps, mode=mode, padding_mode=padding_mode, device=device
        )

    def randomize(self, spatial_shape: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return
        self.distort_steps = tuple(
            tuple(1.0 + self.R.uniform(low=self.distort_limit[0], high=self.distort_limit[1], size=n_cells + 1))
            for n_cells in ensure_tuple_rep(self.num_cells, len(spatial_shape))
        )

    def __call__(
        self, img: torch.Tensor, mode: str | None = None, padding_mode: str | None = None, randomize: bool = True
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            randomize: whether to shuffle the random factors using `randomize()`, default to True.
        """
        if randomize:
            self.randomize(img.shape[1:])
        if not self._do_transform:
            return convert_to_tensor(img, track_meta=get_track_meta())  # type: ignore
        return self.grid_distortion(img, distort_steps=self.distort_steps, mode=mode, padding_mode=padding_mode)


class GridSplit(Transform, MultiSampleTrait):
    """
    Split the image into patches based on the provided grid in 2D.

    Args:
        grid: a tuple define the shape of the grid upon which the image is split. Defaults to (2, 2)
        size: a tuple or an integer that defines the output patch sizes.
            If it's an integer, the value will be repeated for each dimension.
            The default is None, where the patch size will be inferred from the grid shape.

    Example:
        Given an image (torch.Tensor or numpy.ndarray) with size of (3, 10, 10) and a grid of (2, 2),
        it will return a Tensor or array with the size of (4, 3, 5, 5).
        Here, if the `size` is provided, the returned shape will be (4, 3, size, size)

    Note: This transform currently support only image with two spatial dimensions.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, grid: tuple[int, int] = (2, 2), size: int | tuple[int, int] | None = None):
        # Grid size
        self.grid = grid

        # Patch size
        self.size = None if size is None else ensure_tuple_rep(size, len(self.grid))

    def __call__(
        self, image: NdarrayOrTensor, size: int | tuple[int, int] | np.ndarray | None = None
    ) -> list[NdarrayOrTensor]:
        input_size = self.size if size is None else ensure_tuple_rep(size, len(self.grid))

        if self.grid == (1, 1) and input_size is None:
            return [image]

        split_size, steps = self._get_params(image.shape[1:], input_size)
        patches: list[NdarrayOrTensor]
        as_strided_func: Callable
        if isinstance(image, torch.Tensor):
            as_strided_func = torch.as_strided
            c_stride, x_stride, y_stride = image.stride()  # type: ignore
        elif isinstance(image, np.ndarray):
            as_strided_func = np.lib.stride_tricks.as_strided
            c_stride, x_stride, y_stride = image.strides
        else:
            raise ValueError(f"Input type [{type(image)}] is not supported.")

        x_step, y_step = steps
        n_channels = image.shape[0]
        strided_image = as_strided_func(
            image,
            (*self.grid, n_channels, split_size[0], split_size[1]),
            (x_stride * x_step, y_stride * y_step, c_stride, x_stride, y_stride),
        )
        # Flatten the first two dimensions
        strided_image = strided_image.reshape(-1, *strided_image.shape[2:])
        # Make a list of contiguous patches
        if isinstance(image, torch.Tensor):
            patches = [p.contiguous() for p in strided_image]
        elif isinstance(image, np.ndarray):
            patches = [np.ascontiguousarray(p) for p in strided_image]

        return patches

    def _get_params(self, image_size: Sequence[int] | np.ndarray, size: Sequence[int] | np.ndarray | None = None):
        """
        Calculate the size and step required for splitting the image
        Args:
            The size of the input image
        """
        if size is None:
            # infer each sub-image size from the image size and the grid
            size = tuple(image_size[i] // self.grid[i] for i in range(len(self.grid)))

        if any(size[i] > image_size[i] for i in range(len(self.grid))):
            raise ValueError(f"The image size ({image_size})is smaller than the requested split size ({size})")

        steps = tuple(
            (image_size[i] - size[i]) // (self.grid[i] - 1) if self.grid[i] > 1 else image_size[i]
            for i in range(len(self.grid))
        )

        return size, steps


class GridPatch(Transform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps.
    It can sort the patches and return all or a subset of them.

    Args:
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        offset: offset of starting position in the array, default is 0 for each dimension.
        num_patches: number of patches to return. Defaults to None, which returns all the available patches.
            If the required patches are more than the available patches, padding will be applied.
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
            lowest values (`"min"`), or in their default order (`None`). Default to None.
        threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
            Defaults to no filtering.
        pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to ``"constant"``.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

    Returns:
        MetaTensor: A MetaTensor consisting of a batch of all the patches with associated metadata

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        patch_size: Sequence[int],
        offset: Sequence[int] | None = None,
        num_patches: int | None = None,
        overlap: Sequence[float] | float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ):
        self.patch_size = ensure_tuple(patch_size)
        self.offset = ensure_tuple(offset) if offset else (0,) * len(self.patch_size)
        self.pad_mode: NumpyPadMode | None = convert_pad_mode(dst=np.zeros(1), mode=pad_mode) if pad_mode else None
        self.pad_kwargs = pad_kwargs
        self.overlap = overlap
        self.num_patches = num_patches
        self.sort_fn = sort_fn.lower() if sort_fn else None
        self.threshold = threshold

    def filter_threshold(self, image_np: np.ndarray, locations: np.ndarray):
        """
        Filter the patches and their locations according to a threshold
        Args:
            image_np: a numpy.ndarray representing a stack of patches
            locations: a numpy.ndarray representing the stack of location of each patch
        """
        if self.threshold is not None:
            n_dims = len(image_np.shape)
            idx = np.argwhere(image_np.sum(axis=tuple(range(1, n_dims))) < self.threshold).reshape(-1)
            image_np = image_np[idx]
            locations = locations[idx]
        return image_np, locations

    def filter_count(self, image_np: np.ndarray, locations: np.ndarray):
        """
        Sort the patches based on the sum of their intensity, and just keep `self.num_patches` of them.
        Args:
            image_np: a numpy.ndarray representing a stack of patches
            locations: a numpy.ndarray representing the stack of location of each patch
        """
        if self.sort_fn is None:
            image_np = image_np[: self.num_patches]
            locations = locations[: self.num_patches]
        elif self.num_patches is not None:
            n_dims = len(image_np.shape)
            if self.sort_fn == GridPatchSort.MIN:
                idx = np.argsort(image_np.sum(axis=tuple(range(1, n_dims))))
            elif self.sort_fn == GridPatchSort.MAX:
                idx = np.argsort(-image_np.sum(axis=tuple(range(1, n_dims))))
            else:
                raise ValueError(f'`sort_fn` should be either "min", "max" or None! {self.sort_fn} provided!')
            idx = idx[: self.num_patches]
            image_np = image_np[idx]
            locations = locations[idx]
        return image_np, locations

    def __call__(self, array: NdarrayOrTensor):
        # create the patch iterator which sweeps the image row-by-row
        array_np, *_ = convert_data_type(array, np.ndarray)
        patch_iterator = iter_patch(
            array_np,
            patch_size=(None,) + self.patch_size,  # expand to have the channel dim
            start_pos=(0,) + self.offset,  # expand to have the channel dim
            overlap=self.overlap,
            copy_back=False,
            mode=self.pad_mode,
            **self.pad_kwargs,
        )
        patches = list(zip(*patch_iterator))
        patched_image = np.array(patches[0])
        locations = np.array(patches[1])[:, 1:, 0]  # only keep the starting location

        # Filter patches
        if self.num_patches:
            patched_image, locations = self.filter_count(patched_image, locations)
        elif self.threshold:
            patched_image, locations = self.filter_threshold(patched_image, locations)

        # Pad the patch list to have the requested number of patches
        if self.num_patches:
            padding = self.num_patches - len(patched_image)
            if padding > 0:
                patched_image = np.pad(
                    patched_image,
                    [[0, padding], [0, 0]] + [[0, 0]] * len(self.patch_size),
                    constant_values=self.pad_kwargs.get("constant_values", 0),
                )
                locations = np.pad(locations, [[0, padding], [0, 0]], constant_values=0)

        # Convert to MetaTensor
        metadata = array.meta if isinstance(array, MetaTensor) else MetaTensor.get_default_meta()
        metadata[WSIPatchKeys.LOCATION] = locations.T
        metadata[WSIPatchKeys.COUNT] = len(locations)
        metadata["spatial_shape"] = np.tile(np.array(self.patch_size), (len(locations), 1)).T
        output = MetaTensor(x=patched_image, meta=metadata)
        output.is_batch = True

        return output


class RandGridPatch(GridPatch, RandomizableTransform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps,
    and with random offset for the minimal corner of the image, (0,0) for 2D and (0,0,0) for 3D.
    It can sort the patches and return all or a subset of them.

    Args:
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        min_offset: the minimum range of offset to be selected randomly. Defaults to 0.
        max_offset: the maximum range of offset to be selected randomly.
            Defaults to image size modulo patch size.
        num_patches: number of patches to return. Defaults to None, which returns all the available patches.
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
            lowest values (`"min"`), or in their default order (`None`). Default to None.
        threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
            Defaults to no filtering.
        pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to ``"constant"``.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

    Returns:
        MetaTensor: A MetaTensor consisting of a batch of all the patches with associated metadata

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        patch_size: Sequence[int],
        min_offset: Sequence[int] | int | None = None,
        max_offset: Sequence[int] | int | None = None,
        num_patches: int | None = None,
        overlap: Sequence[float] | float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            offset=(),
            num_patches=num_patches,
            overlap=overlap,
            sort_fn=sort_fn,
            threshold=threshold,
            pad_mode=pad_mode,
            **pad_kwargs,
        )
        self.min_offset = min_offset
        self.max_offset = max_offset

    def randomize(self, array):
        if self.min_offset is None:
            min_offset = (0,) * len(self.patch_size)
        else:
            min_offset = ensure_tuple_rep(self.min_offset, len(self.patch_size))
        if self.max_offset is None:
            max_offset = tuple(s % p for s, p in zip(array.shape[1:], self.patch_size))
        else:
            max_offset = ensure_tuple_rep(self.max_offset, len(self.patch_size))

        self.offset = tuple(self.R.randint(low=low, high=high + 1) for low, high in zip(min_offset, max_offset))

    def __call__(self, array: NdarrayOrTensor, randomize: bool = True):
        if randomize:
            self.randomize(array)
        return super().__call__(array)
