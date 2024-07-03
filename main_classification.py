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
import time
import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import SoftmaxLoss
from monai.metrics import ROCAUCMetric
from monai.networks.utils import one_hot
from monai.data import decollate_batch
from models.ssl_head import SSLClassifier
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils_classifier import get_loader
from utils.utils import AverageMeter, distributed_all_gather
import resource
import wandb
import json

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--data_dir", default="/dataset/brats2021/", type=str, help="dataset directory")
parser.add_argument("--json_dir", default="./pretrain/t1_ixi/jsons/", type=str, help="dataset json file")
parser.add_argument("--max_epochs", default=300, type=int, help="number of training epochs")
parser.add_argument("--val_every", default=20, type=int, help="evaluation frequency")
parser.add_argument("--warmup_epochs", default=5, type=int, help="warmup steps")
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
parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
parser.add_argument("--loss_type", default="SSL", type=str)
parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
parser.add_argument("--resume_checkpoint", default=None, type=str, help="checkpoint for resuming training")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--num_classes", default=3, type=int, help="number of classification target classes")
parser.add_argument("--class_weights", default="[3, 1, 6]", type=str, help="number of classification target classes")
parser.add_argument("--distributed_checkpoint", default=None,  action="store_true", help="checkpoint from dpp")
parser.add_argument("--wandb_project", default="Brain-Classification", help="project names for wandb")
parser.add_argument("--wandb_disable", action="store_true", help="disable wandb logging")
parser.add_argument("--run_name", default="random", help="disable wandb logging")
parser.add_argument("--patch_size", default=4, type=int, help="patch size for embedding")

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data = batch_data["image"]
        target = one_hot(labels=batch_data["label"], num_classes=args.num_classes)
        data, target = data.cuda(args.rank), target.cuda(args.rank )
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def val_epoch(model, loader, epoch, loss_function, args, model_inferer=None):
    model.eval()
    start_time = time.time()
    auc_metric = ROCAUCMetric(average="macro")
    run_loss = AverageMeter()
    auc_metric.reset()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], one_hot(labels=batch_data["label"], num_classes=args.num_classes)
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model(data)
                loss = loss_function(logits, target)
                auc_metric(y_pred=logits, y=target)
            if args.distributed:
                loss_list = distributed_all_gather([loss], out_numpy=True,
                                                   is_valid=idx < loader.sampler.valid_length)
                run_loss.update(
                    np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                    n=args.batch_size * args.world_size
                )
            else:
                run_loss.update(loss.item(), n=args.batch_size)

            if args.rank == 0:
                print("Validation epoch:{}, Loss:{:.4f}".format(epoch, loss.item()),
                      ", time {:.2f}s".format(time.time() - start_time))
            start_time = time.time()

        print("Validation aggregating")
        auc_val = auc_metric.aggregate()

    return run_loss.avg, auc_val

def main():
    # Solve for "received 0 items of ancdata" problem
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print('resource limit: ', rlimit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, start_epoch, train_loader, val_best, scaler, scheduler):

        model.train()

        for epoch in range(start_epoch, args.max_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            print(args.rank, time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_function, args=args
            )
            if args.rank == 0:
                print(
                    "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
            try:
                if args.rank == 0 and writer is not None:
                    writer.add_scalar("train_loss", train_loss, epoch)
                    if not args.wandb_disable:
                        wandb.log({"train/loss": train_loss})
            except Exception as error:
                print("An exception occurred:", error)

            b_new_best = False
            if (epoch + 1) % args.val_every == 0:
                if args.distributed:
                    torch.distributed.barrier()
                val_loss, val_auc_val = val_epoch(
                    model,
                    test_loader,
                    epoch=epoch,
                    loss_function=loss_function,
                    args=args)
                if args.rank == 0:
                    writer.add_scalar("Validation/loss", scalar_value=val_loss, global_step=epoch)
                    writer.add_scalar("Validation/auc", scalar_value=val_auc_val, global_step=epoch)
                    # Add wandb log
                    if not args.wandb_disable:
                        wandb.log({"Validation/loss": val_loss,
                                   "Validation/auc": val_auc_val})

                    if val_loss < val_best:
                        val_best = val_loss
                        checkpoint = {
                            "global_step": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                        print(
                            "Model was saved ! Best Recon. Val Loss: {:.4f}".format(val_best.item())
                        )
                    else:
                        print(
                            "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon".format(val_best.item())
                        )
            if scheduler is not None:
                scheduler.step()
        print('exiting epoch num:', epoch)
        return args.max_epochs, train_loss, val_best


    args = parser.parse_args()
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
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


    if args.rank == 0:
        if not args.wandb_disable:
            if args.run_name == 'random':
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=args.wandb_project,

                    # track hyperparameters and run metadata
                    config={
                        "learning_rate": args.lr,
                        "architecture": "SwinUNETR",
                        "dataset": "ADNI",
                        "epochs": args.max_epochs,
                    }
                )
            else:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=args.wandb_project,
                    name=args.run_name,
                    # track hyperparameters and run metadata
                    config={
                        "learning_rate": args.lr,
                        "architecture": "SwinUNETR",
                        "dataset": "ADNI",
                        "epochs": args.max_epochs,
                    }
                )

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None

    model = SSLClassifier(args)
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    start_epoch = 0
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
        model.load_state_dict(state_dict, strict=False)
        model.epoch = start_epoch
        # model.epoch = model_dict["global_step"]
        # start_epoch = model_dict["global_step"]
        # model.optimizer = model_dict["optimizer"]

    scheduler = None
    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.max_epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    class_weights = torch.tensor(json.loads(args.class_weights))
    loss_function = SoftmaxLoss(weights=class_weights)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Applying data parallel to model on device %d' % args.rank)
        # torch.cuda.set_device(args.rank)
        model = DistributedDataParallel(model, device_ids=[args.rank])
    train_loader, test_loader = get_loader(args)

    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while start_epoch < args.max_epochs:
        start_epoch, loss, best_val = train(args, start_epoch, train_loader, best_val, scaler, scheduler)
    checkpoint = {"epoch": args.max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if not args.wandb_disable:
        wandb.finish()

    if args.distributed:
        if args.rank == 0:
            torch.save(model.state_dict(), os.path.join(logdir,"final_model.pth"))
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), os.path.join(logdir,"final_model.pth"))

    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


if __name__ == "__main__":
    main()
