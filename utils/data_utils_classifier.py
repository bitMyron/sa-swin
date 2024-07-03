# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import CacheDataset, DataLoader, Dataset, SmartCacheDataset, load_decathlon_datalist
from utils.data_utils import Sampler
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesdWithSymmetricCounterpart,
    ScaleIntensityRanged,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
import os, json


def datafold_read(datalist, basedir, fold=0, key="training"):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv.strip('/')) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k].strip('/')) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val



def get_loader(args):

    num_workers = args.workers

    datadir = args.data_dir
    datalist_json = args.json_dir

    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=datadir, fold=args.fold)

    print("Dataset all training: number of data: {}".format(len(train_files)))
    print("Dataset all validation: number of data: {}".format(len(validation_files)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label", "id"]),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_files,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=train_files, transform=train_transforms)

    val_ds = Dataset(data=validation_files, transform=val_transforms)
    if args.distributed:
        train_sampler = Sampler(dataset=train_ds, shuffle=True)
        val_sampler = Sampler(dataset=val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
                                  drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
                                  drop_last=True, shuffle=True)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
    #                           drop_last=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=val_sampler,
                            shuffle=False, drop_last=True)

    return train_loader, val_loader



def get_loader_with_symmetric_patch(args):

    num_workers = args.workers

    datadir = args.data_dir
    datalist_json = args.json_dir

    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=datadir, fold=args.fold)

    print("Dataset all training: number of data: {}".format(len(train_files)))
    print("Dataset all validation: number of data: {}".format(len(validation_files)))

    # Need to add this to ensure whole image input
    # ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesdWithSymmetricCounterpart(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image", "image_symmetrical", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesdWithSymmetricCounterpart(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image", "image_symmetrical", "label"]),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_files,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=train_files, transform=train_transforms)

    val_ds = Dataset(data=validation_files, transform=val_transforms)
    if args.distributed:
        train_sampler = Sampler(dataset=train_ds, shuffle=True)
        val_sampler = Sampler(dataset=val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
                                  drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
                                  drop_last=True, shuffle=True)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
    #                           drop_last=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=val_sampler,
                            shuffle=False, drop_last=True)

    return train_loader, val_loader



def get_loader_with_symmetric_whole(args):

    num_workers = args.workers

    datadir = args.data_dir
    datalist_json = args.json_dir

    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=datadir, fold=args.fold)

    print("Dataset all training: number of data: {}".format(len(train_files)))
    print("Dataset all validation: number of data: {}".format(len(validation_files)))

    # Need to add this to ensure whole image input
    # ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesdWithSymmetricCounterpart(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image", "image_symmetrical", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesdWithSymmetricCounterpart(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image", "image_symmetrical", "label"]),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_files,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=train_files, transform=train_transforms)

    val_ds = Dataset(data=validation_files, transform=val_transforms)
    if args.distributed:
        train_sampler = Sampler(dataset=train_ds, shuffle=True)
        val_sampler = Sampler(dataset=val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
                                  drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
                                  drop_last=True, shuffle=True)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
    #                           drop_last=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=val_sampler,
                            shuffle=False, drop_last=True)

    return train_loader, val_loader