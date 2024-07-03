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

import argparse
import os
import datetime
import torch
import torch.distributed as dist
from torch.nn import functional as F
from monai.metrics import ROCAUCMetric
from monai.networks.utils import one_hot
from models.ssl_head import SSLAsymmetryHead
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from utils.data_utils_classifier import get_loader
import resource
import json

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--data_dir", default="/dataset/brats2021/", type=str, help="dataset directory")
parser.add_argument("--json_dir", default="./pretrain/t1_ixi/jsons/", type=str, help="dataset json file")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--loss_type", default="SSL", type=str)
parser.add_argument("--resume_checkpoint", default=None, type=str, help="checkpoint for resuming training")
parser.add_argument("--distributed_checkpoint", default=None,  action="store_true", help="checkpoint from dpp")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--num_classes", default=2, type=int, help="number of classification target classes")
parser.add_argument("--output_dir", default="./datasets/adni/validation/adni_val.json", type=str, help="distributed url")


def main():
    # Solve for "received 0 items of ancdata" problem
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print('resource limit: ', rlimit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    def eval(args, model, val_dataloader):
        model.eval()
        eval_results = []
        auc_metric = ROCAUCMetric(average="macro")
        auc_metric.reset()
        with torch.no_grad():
            for idx, batch_data in enumerate(val_dataloader):
                data = batch_data["image"]
                data_flip = torch.flip(data, dims=(3,))
                target = batch_data["label"]
                data, data_flip, target = data.cuda(args.rank), data_flip.cuda(args.rank), target.cuda(args.rank)
                ids, labels = batch_data["id"], batch_data["label"]
                with autocast(enabled=args.amp):
                    y_contrastive = model(data)
                    y_flip_contrastive = model(data_flip)
                    similarity = F.cosine_similarity(y_contrastive, y_flip_contrastive, dim=1)
                    auc_metric(y_pred= (1-similarity)/ 2, y=target)
                for id, prob, label, y, y_flip in zip(ids.tolist(), similarity.tolist(), labels.tolist(),
                                           y_contrastive.tolist(), y_flip_contrastive.tolist()):
                    eval_results.append({'id': id,
                                         'label': label,
                                         'pred':prob,
                                         'embedding_org': y,
                                         'embedding_flip': y_flip,
                                         })

        auc_val = auc_metric.aggregate()
        print("Validation aggregated, AUC is: ", auc_val)
        json.dump({'results': eval_results}, open(args.output_dir, 'w'), indent=4)

    args = parser.parse_args()
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:%d" % args.local_rank
    args.world_size = os.environ["WORLD_SIZE"]
    args.rank = args.local_rank

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=5400))
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.rank)
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    model = SSLAsymmetryHead(args)
    model.cuda()

    if args.resume_checkpoint:
        model_pth = args.resume_checkpoint
        model_dict = torch.load(model_pth)
        state_dict = model_dict["state_dict"]
        if args.distributed_checkpoint:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model.epoch = model_dict["global_step"]
        model.optimizer = model_dict["optimizer"]

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Applying data parallel to model on device %d' % args.rank)
        # torch.cuda.set_device(args.rank)
        model = DistributedDataParallel(model, device_ids=[args.rank])

    _, test_loader = get_loader(args)
    eval(args, model, test_loader)


if __name__ == "__main__":
    main()
