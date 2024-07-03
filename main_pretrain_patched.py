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
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss_Asymmetry
from models.ssl_head import SymmetryEnhancedHead
from optimizers.lr_scheduler_pretrain import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils_classifier import get_loader_with_symmetric_patch
from utils.ops import aug_rand, rot_rand_for_double
import resource
import wandb

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--data_dir", default="/dataset/brats2021/", type=str, help="dataset directory")
parser.add_argument("--json_dir", default="./pretrain/t1_ixi/jsons/", type=str, help="dataset json file")
parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
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
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
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
parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--wandb_project", default="CT-Pretrain", help="project names for wandb")
parser.add_argument("--wandb_disable", action="store_true", help="disable wandb logging")
parser.add_argument("--run_name", default="random", help="disable wandb logging")
parser.add_argument("--sa_loss", default="cosine_nce", help="loss type for symmetry-aware loss")
parser.add_argument("--use_rotation", action="store_true", help="use rotation loss")
parser.add_argument("--use_reconstruciton", action="store_true", help="use reconstruction loss")
parser.add_argument("--use_contrast", action="store_true", help="use contrastive loss")
parser.add_argument("--use_symmetry", action="store_true", help="use contrastive loss")

def main():
    # Solve for "received 0 items of ancdata" problem
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print('resource limit: ', rlimit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, scaler):

        model.train()
        loss_train = []
        loss_train_recon = []

        for step, batch in enumerate(train_loader):
            t1 = time()
            x = batch["image"].cuda()
            x_flip = batch["image_symmetrical"].cuda()

            label_health = 1 - batch["label"].cuda()
            # label_health = 1 - batch["label"].cuda().repeat_interleave(args.sw_batch_size)

            x1, x1_flip, rot1 = rot_rand_for_double(args, x, x_flip)
            x2, x2_flip, rot2 = rot_rand_for_double(args, x, x_flip)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)

            # # Debug save image to double-check
            # import nibabel as nib
            # import torch
            # numpy_image = x1[0].squeeze(0).cpu().numpy()
            # print(numpy_image.shape)
            # nifti_image = nib.Nifti1Image(numpy_image, affine=np.eye(4))
            # nib.save(nifti_image, '/home/yang/debug/patch_org_aug1.nii.gz')
            # numpy_image_flip = x1_flip[0].squeeze(0).cpu().numpy()
            # nifti_image_flip = nib.Nifti1Image(numpy_image_flip, affine=np.eye(4))
            # nib.save(nifti_image_flip, '/home/yang/debug/patch_flip_rot1.nii.gz')

            target_rots = torch.cat([rot1, rot2], dim=0)
            target_recons = torch.cat([x1, x2], dim=0)

            with autocast(enabled=args.amp):
                loss, losses_tasks, _ = model(x1_augment,
                                           x2_augment,
                                           x1_flip,
                                           x2_flip,
                                           target_rots,
                                           target_recons,
                                           label_health)
            loss_train.append(loss.item())
            loss_train_recon.append(losses_tasks[2].item())
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))
            else:
                print("Step:{}/{}, Loss:{:.4f}, "
                      "Loss_Asym:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps,
                                                             loss.item(), losses_tasks[-1].item(), time() - t1))

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_loss, val_loss_recon, val_loss_asym, img_list = validation(args, test_loader)
                writer.add_scalar("Validation/loss_recon", scalar_value=val_loss_recon, global_step=global_step)
                writer.add_scalar("Validation/loss_asym", scalar_value=val_loss_asym, global_step=global_step)
                writer.add_scalar("train/loss_total", scalar_value=np.mean(loss_train), global_step=global_step)
                writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=global_step)
                # Add wandb log
                if not args.wandb_disable:
                    wandb.log({"Validation/loss_recon": val_loss_recon,
                               "Validation/loss_asym": val_loss_asym,
                               "train/loss_total": np.mean(loss_train),
                               "train/loss_recon": np.mean(loss_train_recon)})
                writer.add_image("Validation/x1_gt", img_list[0], global_step, dataformats="HW")
                writer.add_image("Validation/x1_aug", img_list[1], global_step, dataformats="HW")
                writer.add_image("Validation/x1_recon", img_list[2], global_step, dataformats="HW")

                if val_loss_recon < val_best:
                    val_best = val_loss_recon
                    checkpoint = {
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                    print(
                        "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                            val_best.item(), val_loss_recon.item()
                        )
                    )
                else:
                    print(
                        "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                            val_best.item(), val_loss_recon.item()
                        )
                    )

        return global_step, loss, val_best

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_recon = []
        loss_val_asym = []
        with torch.no_grad():
            print('Validation setting: Num of batchs %d, Num of samples %d' % (
                len(test_loader), len(test_loader.dataset)))
            for step, batch in enumerate(test_loader):
                x = batch["image"].cuda()
                x_flip = batch["image_symmetrical"].cuda()
                label_health = 1 - batch["label"].cuda()

                x1, x1_flip, rot1 = rot_rand_for_double(args, x, x_flip)
                x2, x2_flip, rot2 = rot_rand_for_double(args, x, x_flip)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)

                target_rots = torch.cat([rot1, rot2], dim=0)
                target_recons = torch.cat([x1, x2], dim=0)

                with autocast(enabled=args.amp):
                    loss, losses_tasks, x1_rec = model(x1_augment,
                                                       x2_augment,
                                                       x1_flip,
                                                       x2_flip,
                                                       target_rots,
                                                       target_recons,
                                                       label_health)

                loss_recon = losses_tasks[2]
                loss_asymmetry = losses_tasks[-1]
                loss_val.append(loss.item())
                loss_val_recon.append(loss_recon.item())
                loss_val_asym.append(loss_asymmetry.item())
                x_gt = x1.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
                xgt = x_gt[0][0][:, :, 48] * 255.0
                xgt = xgt.astype(np.uint8)
                x1_augment = x1_augment.detach().cpu().numpy()
                x1_augment = (x1_augment - np.min(x1_augment)) / (np.max(x1_augment) - np.min(x1_augment))
                x_aug = x1_augment[0][0][:, :, 48] * 255.0
                x_aug = x_aug.astype(np.uint8)

                x1_rec = x1_rec.detach().cpu().numpy()
                x1_rec = (x1_rec - np.min(x1_rec)) / (np.max(x1_rec) - np.min(x1_rec))
                recon = x1_rec[0][0][:, :, 48] * 255.0
                recon = recon.astype(np.uint8)
                img_list = [xgt, x_aug, recon]
                print("Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}, Loss Asymmetry:{:.4f}".format(
                    step, loss.item(), loss_recon.item(), loss_asymmetry.item()))

        return np.mean(loss_val), np.mean(loss_val_recon), np.mean(loss_val_asym), img_list

    args = parser.parse_args()
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = args.local_rank

    if not args.wandb_disable:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            name=args.run_name,
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": "SwinUNETR",
                "dataset": "IXI",
                "epochs": args.epochs,
            }
        )

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
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
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None

    model = SymmetryEnhancedHead(args)
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.resume_checkpoint:
        model_pth = args.resume_checkpoint
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        model.epoch = model_dict["epoch"]
        model.optimizer = model_dict["optimizer"]

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Applying data parallel to model on device %d' % args.rank)
        # torch.cuda.set_device(args.rank)
        model = DistributedDataParallel(model, device_ids=[args.rank])
    train_loader, test_loader = get_loader_with_symmetric_patch(args)

    global_step = 0
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if not args.wandb_disable:
        wandb.finish()

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


if __name__ == "__main__":
    main()
