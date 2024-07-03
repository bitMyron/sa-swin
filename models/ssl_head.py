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

import torch
import torch.nn as nn
import numpy as np

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.blocks.transformerblock import TransformerCABlock
from monai.utils import ensure_tuple_rep
from timm.models.layers import trunc_normal_
from transformers.models.bert.configuration_bert import BertConfig
from losses.loss import Contrast, Loss_AA


class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=768):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )

        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_rot, x_contrastive, x_rec


class SymmetryEnhancedHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=768):
        super(SymmetryEnhancedHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )

        self.bert_config = BertConfig.from_json_file('./configs/bert_config.json')
        self.symmetryAwareTX = TransformerCABlock(
            config=self.bert_config,
            layer_num=1
        )
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        self.aaw_pre = nn.Identity()
        self.aaw_head = nn.Linear(dim, 512)

        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm3d(dim // 2),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                # nn.BatchNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                # nn.BatchNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                # nn.BatchNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                # nn.BatchNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1),
            )

        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, args.batch_size * args.sw_batch_size).cuda()
        temp = 0.07
        self.aa_loss = Loss_AA(temp, args).cuda()
        self.epsilon = 1e-6

        self.alpha1 = 1.0 if args.use_rotation else 0.0
        self.alpha2 = 1.0 if args.use_contrast else 0.0
        self.alpha3 = 1.0 if args.use_reconstruciton else 0.0
        self.alpha0 = 0.5 if args.use_symmetry else 0.0

        print("rotation loss weight: ", self.alpha1)
        print("contrastive loss weight: ", self.alpha2)
        print("reconstruction loss weight: ", self.alpha3)
        print("symmetry loss weight: ", self.alpha0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x2, x1_flip, x2_flip, target_rots, target_recons, label_health):
        x1_out = self.swinViT(x1.contiguous())[4]
        x1_flip_out = self.swinViT(x1_flip.contiguous())[4]
        x2_out = self.swinViT(x2.contiguous())[4]
        x2_flip_out = self.swinViT(x2_flip.contiguous())[4]

        _, c, h, w, d = x1_out.shape
        x1_reshape = x1_out.flatten(start_dim=2, end_dim=4)
        x1_reshape = x1_reshape.transpose(1, 2)
        x1_flip_reshape = x1_flip_out.flatten(start_dim=2, end_dim=4)
        x1_flip_reshape = x1_flip_reshape.transpose(1, 2)

        x2_reshape = x2_out.flatten(start_dim=2, end_dim=4)
        x2_reshape = x2_reshape.transpose(1, 2)
        x2_flip_reshape = x2_flip_out.flatten(start_dim=2, end_dim=4)
        x2_flip_reshape = x2_flip_reshape.transpose(1, 2)

        # Proxy0 - Asymmetry-aware
        x1_aw = self.aaw_pre(x1_reshape[:, 1])
        x1_aw = self.aaw_head(x1_aw)
        x1_flip_aw = self.aaw_pre(x1_flip_reshape[:, 1])
        x1_flip_aw = self.aaw_head(x1_flip_aw)
        x2_aw = self.aaw_pre(x2_reshape[:, 1])
        x2_aw = self.aaw_head(x2_aw)
        x2_flip_aw = self.aaw_pre(x2_flip_reshape[:, 1])
        x2_flip_aw = self.aaw_head(x2_flip_aw)

        asymmetry_loss = self.alpha0 * (self.aa_loss(x1_aw, x1_flip_aw, label_health)
                                        + self.aa_loss(x2_aw, x2_flip_aw, label_health))

        ### Apply asymmetry-aware attention
        x1_reshape = self.symmetryAwareTX(x1_reshape, encoder_hidden_states=x1_flip_reshape)[0]
        x2_reshape = self.symmetryAwareTX(x2_reshape, encoder_hidden_states=x2_flip_reshape)[0]

        # Proxy1 - Rotation
        x1_rot = self.rotation_pre(x1_reshape[:, 0])
        x1_rot = self.rotation_head(x1_rot)
        x2_rot = self.rotation_pre(x2_reshape[:, 0])
        x2_rot = self.rotation_head(x2_rot)
        output_rots = torch.cat([x1_rot, x2_rot], dim=0)
        rot_loss = self.alpha1 * self.rot_loss(output_rots, target_rots)

        # Proxy2 - Contrastive
        x1_contrastive = self.contrastive_pre(x1_reshape[:, 1])
        x1_contrastive = self.contrastive_head(x1_contrastive)
        x2_contrastive = self.contrastive_pre(x2_reshape[:, 1])
        x2_contrastive = self.contrastive_head(x2_contrastive)
        contrast_loss = self.alpha2 * self.contrast_loss(x1_contrastive, x2_contrastive)

        # Proxy3 - Reconstruct
        x1_rec = x1_reshape.transpose(1, 2)
        x1_rec = x1_rec.view(-1, c, h, w, d)
        x1_rec = self.conv(x1_rec)
        x2_rec = x2_reshape.transpose(1, 2)
        x2_rec = x2_rec.view(-1, c, h, w, d)
        x2_rec = self.conv(x2_rec)
        output_recons = torch.cat([x1_rec, x2_rec], dim=0)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)

        # total_loss = rot_loss + contrast_loss + recon_loss
        total_loss = rot_loss + contrast_loss + recon_loss + asymmetry_loss
        return total_loss, (rot_loss, contrast_loss, recon_loss, asymmetry_loss), x1_rec


class SACAClassifierHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=768):
        super(SACAClassifierHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        self.bert_config = BertConfig.from_json_file('./configs/bert_config.json')
        self.symmetryAwareTX = TransformerCABlock(
            config=self.bert_config,
            layer_num=1
        )
        self.classification_pre = nn.Identity()
        self.classification_head = nn.Linear(dim, args.num_classes)

    def forward(self, x, x_flip):
        x_out = self.swinViT(x.contiguous())[4]
        x_flip_out = self.swinViT(x_flip.contiguous())[4]
        _, c, h, w, d = x_out.shape
        x_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x_reshape = x_reshape.transpose(1, 2)
        x_flip_reshape = x_flip_out.flatten(start_dim=2, end_dim=4)
        x_flip_reshape = x_flip_reshape.transpose(1, 2)

        ### Apply asymmetry-aware attention
        attention_mask = [1] * len(x_reshape) * 2
        np_mask = np.tril(np.expand_dims(np.array(attention_mask), 0).repeat(len(attention_mask), 0))
        attention_mask = torch.from_numpy(np_mask).unsqueeze(0).unsqueeze(0)
        x_reshape = self.symmetryAwareTX(x_reshape, encoder_hidden_states=x_flip_reshape, attention_mask=attention_mask)[0]

        x_cls = self.classification_pre(x_reshape[:, 0])
        y_out = self.classification_head(x_cls)

        return y_out


class SSLClassifier(nn.Module):
    def __init__(self, args, dim=768):
        super(SSLClassifier, self).__init__()
        patch_size = ensure_tuple_rep(args.patch_size, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.classification_pre = nn.Identity()
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        self.classification_head = nn.Linear(dim, args.num_classes)

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_cls = self.classification_pre(x4_reshape[:, 0])
        y_out = self.classification_head(x_cls)

        return y_out


class SSLAsymmetryHead(nn.Module):
    def __init__(self, args, dim=768):
        super(SSLAsymmetryHead, self).__init__()
        patch_size = ensure_tuple_rep(4, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.classification_pre = nn.Identity()
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        return x_contrastive
