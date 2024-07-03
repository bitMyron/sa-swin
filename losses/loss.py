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
from torch.nn import functional as F

class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)

class AsymmetrySimilarityLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, method='siamese', margin=2.0):
        super(AsymmetrySimilarityLoss, self).__init__()
        self.margin = margin
        self.temp = torch.nn.Parameter(0.07 * torch.ones([])).cuda()
        self.method = method
        self.epsilon = 1e-6
        self.criterion = torch.nn.BCEWithLogitsLoss().cuda()

        print('using loss method: ', self.method)

    def forward(self, output1, output2, label):

        if self.method == 'siamese':

            # 0 healthy, 1 not healthy
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                              2))
        elif self.method == 'cosine_nce':
            # v0.1 implementation
            similarity = F.cosine_similarity(output1, output2, dim=1)
            # Based on label values, calculate the loss
            loss_contrastive = self.criterion(similarity, label.float())
        else:
            # output1 = F.normalize(output1, dim=-1)
            # output2 = F.normalize(output2, dim=-1)
            sim = F.cosine_similarity(output1, output2, dim=1)
            sim = sim / self.temp
            # v0.1 implementation
            # similarity = F.cosine_similarity(x, x_flip, dim=1)
            # Based on label values, calculate the loss
            # loss = self.criterion(similarity, label.float())
            log_numerator = torch.log(torch.sum(torch.exp(sim) * label, dim=-1) + self.epsilon)  # (B)
            log_denominator = torch.logsumexp(sim, dim=-1)  # (B)

            loss = -log_numerator + log_denominator
            loss_contrastive = loss.mean(0)

        return loss_contrastive


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()

        self.alpha1 = 1.0 if args.use_rotation else 0.0
        self.alpha2 = 1.0 if args.use_contrast else 0.0
        self.alpha3 = 1.0 if args.use_reconstruciton else 0.0

        print("rotation loss weight: ", self.alpha1)
        print("contrastive loss weight: ", self.alpha2)
        print("reconstruction loss weight: ", self.alpha3)

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)


class Loss_Asymmetry(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        rot_label_weight = torch.Tensor([1, 5, 5, 5]).cuda()
        self.rot_loss = torch.nn.CrossEntropyLoss(weight=rot_label_weight).cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.temp = torch.nn.Parameter(0.07 * torch.ones([])).cuda()
        self.epsilon = 1e-6
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0
        self.alpha4 = 0.1

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, flip_contrastive, label_health,
                 output_recons, target_recons):

        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        sim = F.cosine_similarity(output_contrastive, flip_contrastive, dim=1)
        sim = sim / self.temp
        log_numerator = torch.log(torch.sum(torch.exp(sim) * label_health, dim=-1) + self.epsilon)  # (B)
        log_denominator = torch.logsumexp(sim, dim=-1)  # (B)
        loss = -log_numerator + log_denominator
        asymmetry_loss = self.alpha4 * loss.mean(0)
        total_loss = rot_loss + contrast_loss + recon_loss + asymmetry_loss
        return total_loss, (rot_loss, contrast_loss, recon_loss, asymmetry_loss)


class Loss_AA(torch.nn.Module):
    def __init__(self, temp, args):
        super().__init__()
        self.temp = torch.nn.Parameter(temp * torch.ones([])).cuda()
        self.epsilon = 1e-6
        self.margin = 20
        self.method = args.sa_loss
        self.criterion = torch.nn.BCEWithLogitsLoss().cuda()

    def __call__(self, output_contrastive, flip_contrastive, label_health):

        sim = F.cosine_similarity(output_contrastive, flip_contrastive, dim=1)

        if self.method == 'siamese':
            # 0 healthy, 1 not healthy
            # euclidean_distance = F.pairwise_distance(output_contrastive, flip_contrastive, keepdim=True)
            # asymmetry_loss = torch.mean((1 - label_health) * torch.pow(euclidean_distance, 2) +
            #                               (label_health) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
            #                                                   2))
            asymmetry_loss = torch.mean((1 - label_health) * sim + label_health * (1 - sim))
            return asymmetry_loss
        elif self.method == 'cosine_nce':
            # Based on label values, calculate the loss
            asymmetry_loss = self.criterion(sim, label_health.float())
            return asymmetry_loss
        else:
            sim = sim / self.temp
            log_numerator = torch.log(torch.sum(torch.exp(sim) * label_health, dim=-1) + self.epsilon)  # (B)
            log_denominator = torch.logsumexp(sim, dim=-1)  # (B)
            loss = -log_numerator + log_denominator
            asymmetry_loss = loss.mean(0)
            return asymmetry_loss



class Loss_PP(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().to('cuda:3')
        self.recon_loss = torch.nn.L1Loss().to('cuda:3')
        self.contrast_loss = Contrast(args, batch_size).to('cuda:3')
        self.alpha1 = torch.Tensor([1.0])
        self.alpha2 = torch.Tensor([1.0])
        self.alpha3 = torch.Tensor([1.0])

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1.to('cuda:3') * self.rot_loss(output_rot, target_rot)
        # print('alpha, output_contrastive, target_contrastive are on divice: %s, %s, %s' % (self.alpha2.get_device(), output_contrastive.get_device(), target_contrastive.get_device()))
        # contrast_loss = self.alpha2.to('cuda:3') * self.contrast_loss(output_contrastive, target_contrastive).to('cuda:3')
        # recon_loss = self.alpha3.to('cuda:3') * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss

        return total_loss, (total_loss, total_loss, total_loss)


class SoftmaxLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            self.loss = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=weights).cuda()

    def __call__(self, logits, target):
        classification_loss = self.loss(logits, target)
        return classification_loss
