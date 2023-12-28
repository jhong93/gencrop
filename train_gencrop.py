#!/usr/bin/env python3

import os
import argparse
import math
from contextlib import nullcontext
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import einops
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from rod_align.modules.rod_align import RoDAlignAvg
from util.io import load_gz_json, store_json
from util.nn import PositionalEncoding
from util.dataset import OutpaintImageCropDataset

cv2.setNumThreads(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--max_epoch_len', type=int)
    parser.add_argument('--use_subject', action='store_true',
                        help='Use the subject mask as input')
    parser.add_argument('--arch', default='resnet50')

    parser.add_argument('--no_bad_filter', action='store_true',
                        help='Do not use the results from the bad image model')
    parser.add_argument('--no_heuristic_filter', action='store_true',
                        help='Do not filter with subject heuristics')

    parser.add_argument('--no_sbl', action='store_true',
                        help='No subject boundary loss')

    parser.add_argument('--no_invert', action='store_true',
                        help='Do not sample inverted image orientations')
    parser.add_argument('-s', '--save_dir')
    return parser.parse_args()


class CropTransformer(nn.Module):

    def __init__(self, anchor_stride, feat_dim, spatial_dim,
                 positional_embedding=False):
        super().__init__()
        out_channel = int((16 / anchor_stride) ** 2 * 4)

        enc_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        if positional_embedding:
            self.pe = nn.Embedding(spatial_dim ** 2, feat_dim)
        else:
            self.pe = PositionalEncoding(
                feat_dim, dropout=0.1, max_len=spatial_dim ** 2)
        self.positional_embedding = positional_embedding
        self.crop = nn.Linear(feat_dim, out_channel)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(2, 0, 1)
        if self.positional_embedding:
            x = x + self.pe.weight.unsqueeze(1)
        else:
            x = self.pe(x)
        x = self.enc(x).permute(1, 0, 2)
        D = x.shape[-1]
        x = self.crop(x.reshape(-1, D)).reshape(B, H, W, -1).permute(
            0, 3, 1, 2)
        return x


def generate_anchors(anchor_stride):
    assert anchor_stride == 16
    # HACK: assume downsample 16x
    P_h = np.array([8])
    P_w = np.array([8])

    # assert anchor_stride <= 16, 'not implement for anchor_stride{} > 16'.format(anchor_stride)
    # P_h = np.array([2+i*4 for i in range(16 // anchor_stride)])
    # P_w = np.array([2+i*4 for i in range(16 // anchor_stride)])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = torch.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1
    return anchors


def shift(shape, stride, anchors):
    shift_w = torch.arange(0, shape[0]) * stride
    shift_h = torch.arange(0, shape[1]) * stride
    shift_w, shift_h = torch.meshgrid([shift_w, shift_h])
    shifts  = torch.stack([shift_w, shift_h], dim=-1)  # h,w,2
    # add A anchors (A,2) to
    # shifts (h,w,2) to get
    # shift anchors (A,h,w,2)
    trans_anchors = einops.rearrange(anchors, 'a c -> a 1 1 c')
    trans_shifts  = einops.rearrange(shifts,  'h w c -> 1 h w c')
    all_anchors   = trans_anchors + trans_shifts
    return all_anchors


class CropRegression(nn.Module):

    def __init__(self, anchor_stride, img_size):
        super().__init__()
        self.num_anchors = (16 // anchor_stride) ** 2

        anchors = generate_anchors(anchor_stride)
        feat_shape  = (img_size[0] // 16, img_size[1] // 16)
        all_anchors = shift(feat_shape, 16, anchors)
        all_anchors = all_anchors.float().unsqueeze(0)
        # 1,num_anchors,h//16,w//16,2

        all_anchors[..., 0] /= img_size[0]
        all_anchors[..., 1] /= img_size[1]

        self.upscale_factor = max(1, self.num_anchors // 2)
        anchors_x   = F.pixel_shuffle(
            all_anchors[...,0], upscale_factor=self.upscale_factor)
        anchors_y   = F.pixel_shuffle(
            all_anchors[...,1], upscale_factor=self.upscale_factor)
        # 1,h//s,w//s,2 where s=16//anchor_stride

        all_anchors = torch.stack([anchors_x, anchors_y], dim=-1).squeeze(1)
        self.register_buffer('all_anchors', all_anchors)

    def forward(self, offsets):
        '''
        :param offsets: b,num_anchors*4,h//16,w//16
        :return: b,4
        '''
        offsets = einops.rearrange(offsets, 'b (n c) h w -> b n h w c',
                                   n=self.num_anchors, c=4)
        coords  = [F.pixel_shuffle(
            offsets[...,i],
            upscale_factor=self.upscale_factor) for i in range(4)]
        # b, h//s, w//s, 4, where s=16//anchor_stride
        offsets = torch.stack(coords, dim=-1).squeeze(1)
        regression = torch.zeros_like(offsets) # b,h,w,4
        regression[...,0::2] = offsets[..., 0::2] + self.all_anchors[...,0:1]
        regression[...,1::2] = offsets[..., 1::2] + self.all_anchors[...,1:2]
        regression = einops.rearrange(regression, 'b h w c -> b (h w) c')
        return regression


class CropScore(nn.Module):

    def __init__(self, img_size, feat_dim):
        super().__init__()
        assert img_size[0] == img_size[1]
        self.img_size = img_size

        self.rod_align = RoDAlignAvg(5, 5, 1 / 16)

        self.net = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 128, kernel_size=5),
            nn.ReLU(True),
            nn.Flatten(1),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 1))

    def forward(self, xyxy, f):
        B, M, N = xyxy.shape
        xyxy = xyxy.detach().clamp(min=0, max=1) * self.img_size[0]
        x1 = torchvision.ops.roi_align(
            f, [xyxy[i] for i in range(B)], output_size=(5, 5),
            spatial_scale=f.shape[-1] / self.img_size[0])

        index = torch.arange(B).view(-1, 1, 1).repeat(1, M, 1).to(f.device)
        crop_xyxy_rod = torch.cat(
            [index, xyxy], dim=-1).reshape(-1, 5).contiguous()
        x2 = self.rod_align(f.float(), crop_xyxy_rod.float())

        x = torch.cat([x1, x2], dim=1)
        x = self.net(x).reshape(B, M)
        return x


class CroppingModel(nn.Module):

    def __init__(self, in_channels, img_dim, arch, debug=False):
        super().__init__()
        self.debug = debug

        anchor_stride = 16

        self.feature = timm.create_model(
            arch, pretrained=True, in_chans=in_channels, features_only=True)

        f_dim = 256
        f_down = 32

        # 16x downsample
        self.crop = CropTransformer(
            anchor_stride=anchor_stride, feat_dim=f_down,
            spatial_dim=img_dim // 16)
        self.regress = CropRegression(anchor_stride=anchor_stride,
                                      img_size=(img_dim, img_dim))
        self.score = CropScore((img_dim, img_dim), f_down)

        if arch in 'resnet50':
            self.is_vgg = False
            self.f3_unify = nn.Conv2d(512, f_dim, 1)
            self.f4_unify = nn.Conv2d(1024, f_dim, 1)
            self.f5_unify = nn.Conv2d(2048, f_dim, 1)
        elif arch == 'resnet18':
            self.is_vgg = False
            self.f3_unify = nn.Conv2d(128, f_dim, 1)
            self.f4_unify = nn.Conv2d(256, f_dim, 1)
            self.f5_unify = nn.Conv2d(512, f_dim, 1)
        elif arch == 'vgg16':
            self.is_vgg = True
            self.f3_unify = nn.Conv2d(256, f_dim, 1)
            self.f4_unify = nn.Conv2d(512, f_dim, 1)
            self.f5_unify = nn.Conv2d(512, f_dim, 1)
        elif arch == 'mobilenetv2_100':
            self.is_vgg = False
            self.f3_unify = nn.Conv2d(32, f_dim, 1)
            self.f4_unify = nn.Conv2d(96, f_dim, 1)
            self.f5_unify = nn.Conv2d(320, f_dim, 1)
        else:
            raise ValueError('Unknown arch: {}'.format(arch))

        self.f_down = nn.Conv2d(f_dim, f_down, 1)

    def forward(self, x, subject_xyxy):
        B, C, H, W = x.shape
        x = self.feature(x)

        if not self.is_vgg:
            f4 = x[3]
            f3 = F.interpolate(
                x[2], size=f4.shape[2:], mode='bilinear',
                align_corners=True)
            f5 = F.interpolate(
                x[4], size=f4.shape[2:], mode='bilinear',
                align_corners=True)
        else:
            f5 = x[4]
            f3 = F.interpolate(
                x[2], size=f5.shape[2:], mode='bilinear',
                align_corners=True)
            f4 = F.interpolate(
                x[3], size=f5.shape[2:], mode='bilinear',
                align_corners=True)

        # 16x downsample
        f = self.f3_unify(f3) + self.f4_unify(f4) + self.f5_unify(f5)
        f = F.relu(self.f_down(f))
        FH, FW = f.shape[2:]

        x = self.crop(f)
        x = self.regress(x)

        # print(FW, FH, x.shape)

        m = torch.zeros(B, FH, FW, device=x.device)
        for i in range(B):
            sx1, sy1, sx2, sy2 = subject_xyxy[i].int().tolist()
            m[i, math.floor(sy1 / 16):math.ceil(sy2 / 16),
               math.floor(sx1 / 16):math.ceil(sx2 / 16)] = 1
        m = m.reshape(B, -1)
        m /= m.sum(dim=1, keepdim=True) + 1e-4

        w = self.score(x, f)
        w = F.softmax(w * m, dim=1)

        # assert w.isnan().sum() == 0
        r = torch.sum(x * w.unsqueeze(-1), dim=1)

        if not self.debug:
            return r, x, w.reshape(B, FH, FW)

        else:
            # print(w.max(), w.median())
            weights = torch.zeros(B, H, W)
            for i in range(B):
                for j in range(x.shape[1]):
                    x1, y1, x2, y2 = (x[i, j] * H).int().detach().cpu().tolist()
                    q = w[i, j].cpu().item()
                    # print(x2 - x1, y2 - y1, w[i, j].cpu().item())
                    if (x1 > x2 or y1 > y2 or x1 < 0 or y1 < 0
                        or x2 >= W or y2 >= H
                    ):
                        continue
                    weights[i, y1:y2, x1] += q
                    weights[i, y1:y2, x2] += q
                    weights[i, y1, x1:x2] += q
                    weights[i, y2, x1:x2] += q
            return r, x, weights


def subject_boundary_loss(subject_mask_xyxy, crop_xyxy, valid_xyxy, pred_xyxy,
                      margin=0.025):
    loss = (margin - (pred_xyxy - subject_mask_xyxy).abs()).clip(min=0)
    mask = (torch.isclose(crop_xyxy, subject_mask_xyxy, atol=0.01) |
            torch.isclose(valid_xyxy, subject_mask_xyxy, atol=0.01))
    loss[mask] = 0
    return loss.mean()


class GenCrop:

    def __init__(self, use_subject, in_dim,  arch, device='cuda', debug=False,
                 subject_boundary_loss=False):
        self.use_subject = use_subject
        self.use_sbl = subject_boundary_loss
        in_channels = 3
        if use_subject:
            in_channels += 1

        backbone = CroppingModel(in_channels, in_dim, arch=arch, debug=debug)
        backbone.to(device)
        print('Num parameters:',
              sum(p.numel() for p in backbone.parameters() if p.requires_grad))

        self.backbone = backbone
        self.device = device

    def predict(self, img, subject_xywh, subject=None):
        self.backbone.eval()

        B, C, H, W = img.shape
        assert H == W

        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16):
            img = img.half().to(self.device)

            subject_xyxy = subject_xywh.clone()
            subject_xyxy[:, 2:] += subject_xyxy[:, :2]

            cond = [img]
            if self.use_subject:
                cond.append(subject.half().to(self.device))
            cond = torch.cat(cond, dim=1) if len(cond) > 1 else cond[0]
            pred, _, weight = self.backbone(cond, subject_xyxy)

        pred *= H
        pred[:, 2] -= pred[:, 0]
        pred[:, 3] -= pred[:, 1]
        pred[pred < 0] = 0
        return pred.cpu().numpy(), weight.cpu().numpy()

    def epoch(self, loader, optimizer=None, lr_scheduler=None,
              scaler=None, epoch=-1, max_epoch_len=None):
        if optimizer is None:
            self.backbone.eval()
            mode = 'eval'
        else:
            self.backbone.train()
            optimizer.zero_grad()
            mode = 'train'
        if epoch >= 0:
            mode = '{}:{}'.format(epoch, mode)

        ewma_loss = None
        epoch_loss = 0
        steps = 0

        with (tqdm(loader) as pbar,
              torch.no_grad() if optimizer is None else nullcontext()):
            for batch in pbar:
                img = batch['img']
                crop_xywh = batch['crop_xywh']
                valid_xywh = batch['valid_xywh']
                subject_xywh = batch['subject_xywh']
                subject_xyxy = subject_xywh.clone()
                subject_xyxy[:, 2:] += subject_xyxy[:, :2]

                B, C, H, W = img.shape
                img = img.half().to(self.device)

                assert H == W
                crop_xyxy_scaled = crop_xywh.float() / H
                crop_xyxy_scaled[:, 2:] += crop_xyxy_scaled[:, :2]
                crop_xyxy_scaled = crop_xyxy_scaled.to(self.device)

                valid_xyxy_scaled = valid_xywh.float() / H
                valid_xyxy_scaled[:, 2:] += valid_xyxy_scaled[:, :2]
                valid_xyxy_scaled = valid_xyxy_scaled.to(self.device)


                cond = [img]
                if self.use_subject:
                    cond.append(batch['subject'].half().to(self.device))
                cond = torch.cat(cond, dim=1) if len(cond) > 1 else cond[0]

                with torch.autocast(self.device, dtype=torch.float16):
                    pred, pred_all, _ = self.backbone(cond, subject_xyxy)

                    # loss against ground truth
                    loss = F.l1_loss(pred, crop_xyxy_scaled)

                    loss += 0.1 * F.l1_loss(
                        pred_all, crop_xyxy_scaled.unsqueeze(1).repeat(
                            1, pred_all.shape[1], 1))

                    # loss against subject edge
                    if self.use_sbl:
                        subject_mask_xyxy = \
                            batch['subject_mask_xywh'].float() / H
                        subject_mask_xyxy[:, 2:] += subject_mask_xyxy[:, :2]
                        subject_mask_xyxy = subject_mask_xyxy.to(self.device)

                        loss += 10 * subject_boundary_loss(
                            subject_mask_xyxy, valid_xyxy_scaled,
                            crop_xyxy_scaled, pred)

                if optimizer is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                loss_py = loss.detach().item()
                if ewma_loss is None:
                    ewma_loss = loss_py
                else:
                    ewma_loss = 0.9 * ewma_loss + 0.1 * loss_py
                epoch_loss += loss_py
                steps += 1

                pbar.set_description('{}: {:0.8f}'.format(mode, ewma_loss))

                if max_epoch_len is not None and steps >= max_epoch_len:
                    break
        return epoch_loss / steps


def main(args):
    num_workers = 4

    det_path = os.path.join(args.dataset, 'detect.json.gz')
    if os.path.exists(det_path):
        det_dict = load_gz_json(det_path)
        mask_file = os.path.join(args.dataset, 'mask.npz')
    else:
        print('No detect.json.gz found, using None')
        det_dict = mask_file = None

    if args.no_bad_filter:
        bad_img_file = None
    else:
        bad_img_file = os.path.join(args.dataset, 'bad.json')

    train_dataset = OutpaintImageCropDataset(
        os.path.join(args.dataset, '..', 'train.json'),
        os.path.join(args.dataset, 'images'),
        det_dict,
        args.dim, augment=True,
        bad_img_file=bad_img_file, mask_file=mask_file,
        invert_prob=0 if args.no_invert else 0.2,
        heuristic_filter=not args.no_heuristic_filter)
    train_dataset.print_info()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=num_workers,
        shuffle=True)

    val_dataset = OutpaintImageCropDataset(
        os.path.join(args.dataset, '..', 'val.json'),
        os.path.join(args.dataset, 'images'),
        det_dict,
        args.dim, bad_img_file=bad_img_file, mask_file=mask_file,
        invert_prob=0 if args.no_invert else 0.2)
    val_dataset.print_info()
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=num_workers)
    del det_dict

    model = GenCrop(
        use_subject=args.use_subject, in_dim=args.dim, arch=args.arch,
        subject_boundary_loss=not args.no_sbl)
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=0.0001)

    num_steps_per_epoch = len(train_loader)
    if args.max_epoch_len is not None:
        num_steps_per_epoch = min(num_steps_per_epoch, args.max_epoch_len)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(num_steps_per_epoch * args.num_epochs))
    scaler = torch.cuda.amp.GradScaler()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        store_json(os.path.join(args.save_dir, 'config.json'), {
            'dataset': args.dataset,
            'arch': args.arch,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'img_dim': args.dim,
            'use_subject': args.use_subject,
            'use_subject_boundary_loss': not args.no_sbl
        })

    losses = []
    for epoch in range(args.num_epochs):
        train_loss = model.epoch(train_loader, optimizer, lr_scheduler, scaler,
                                 epoch=epoch, max_epoch_len=args.max_epoch_len)
        val_loss = model.epoch(val_loader, epoch=epoch,
                               max_epoch_len=args.max_epoch_len)
        losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})

        if args.save_dir is not None:
            store_json(os.path.join(args.save_dir, 'loss.json'), losses)
            torch.save(model.backbone.state_dict(),
                       os.path.join(args.save_dir, 'model{}.pt'.format(epoch)))
    print('Done!')


if __name__ == '__main__':
    main(get_args())