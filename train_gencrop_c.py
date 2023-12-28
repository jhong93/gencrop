#!/usr/bin/env python3

import os
import argparse
import math
from contextlib import nullcontext
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from util.io import load_gz_json, store_json
from util.box import Box
from util.nn import PositionalEncoding
from util.dataset import OutpaintImageCropDataset
from train_gencrop import CropRegression, CropScore, subject_boundary_loss

cv2.setNumThreads(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--max_epoch_len', type=int)
    parser.add_argument('--use_subject', action='store_true'
                        help='Use the subject mask as input')

    parser.add_argument('--no_sbl', action='store_true',
                        help='No subject boundary loss')

    parser.add_argument('--no_invert', action='store_true',
                        help='Do not sample inverted image orientations')
    parser.add_argument('-s', '--save_dir')
    return parser.parse_args()


class ConditionalDatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ret = self.dataset[index]

        gt_crop = ret['crop_xywh']
        if len(gt_crop.shape) > 1:
            # Pick first one if many
            gt_crop = gt_crop[0]
        crop = Box(*gt_crop.tolist())
        w, h = ret['valid_xywh'][2:].tolist()

        crop_hw = crop.h / crop.w
        ret['cond'] = torch.tensor(
            [crop.h * crop.w / (h * w), crop_hw, 1 / crop_hw])
        return ret

    def print_info(self):
        self.dataset.print_info()


class CropTransformer(nn.Module):

    def __init__(self, anchor_stride, feat_dim, spatial_dim,
                 positional_embedding=False):
        super().__init__()
        out_channel = int((16 / anchor_stride) ** 2 * 4)

        self.cond = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout())

        dec_layer = nn.TransformerDecoderLayer(d_model=feat_dim, nhead=8)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=2)
        if positional_embedding:
            self.pe = nn.Embedding(spatial_dim ** 2, feat_dim)
        else:
            self.pe = PositionalEncoding(
                feat_dim, dropout=0.1, max_len=spatial_dim ** 2)
        self.positional_embedding = positional_embedding
        self.crop = nn.Linear(feat_dim, out_channel)

    def forward(self, x, c):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(2, 0, 1)
        if self.positional_embedding:
            x = x + self.pe.weight.unsqueeze(1)
        else:
            x = self.pe(x)
        c = self.cond(c).unsqueeze(0)

        x = self.dec(x, c).permute(1, 0, 2)
        D = x.shape[-1]
        x = self.crop(x.reshape(-1, D)).reshape(B, H, W, -1).permute(
            0, 3, 1, 2)
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

        if arch == 'resnet50':
            self.f3_unify = nn.Conv2d(512, f_dim, 1)
            self.f4_unify = nn.Conv2d(1024, f_dim, 1)
            self.f5_unify = nn.Conv2d(2048, f_dim, 1)
        else:
            raise ValueError('Unknown arch: {}'.format(arch))

        self.f_down = nn.Conv2d(f_dim, f_down, 1)

    def forward(self, x, c, subject_xyxy):
        B, C, H, W = x.shape
        x = self.feature(x)

        # for v in x:
        #     print(v.shape)

        f4 = x[3]
        f3 = F.interpolate(
             x[2], size=f4.shape[2:], mode='bilinear',
            align_corners=True)
        f5 = F.interpolate(
            x[4], size=f4.shape[2:], mode='bilinear',
            align_corners=True)

        # 16x downsample
        f = self.f3_unify(f3) + self.f4_unify(f4) + self.f5_unify(f5)
        f = F.relu(self.f_down(f))
        FH, FW = f.shape[2:]

        x = self.crop(f, c)
        x = self.regress(x)

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


class GenCropC:

    def __init__(self, use_subject, in_dim,
                 arch='resnet50', device='cuda', debug=False,
                 subject_boundary_loss=False):
        self.use_subject = use_subject
        self.use_sbl = subject_boundary_loss
        in_channels = 3
        if use_subject:
            in_channels += 1

        backbone = CroppingModel(in_channels, in_dim, arch=arch, debug=debug)
        backbone.to(device)
        self.backbone = backbone
        self.device = device

    def predict(self, img, style_cond, subject_xywh, subject=None):
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
            pred, _, weight = self.backbone(cond, style_cond.to(self.device),
                                            subject_xyxy)

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
                style_cond = batch['cond']

                B, C, H, W = img.shape
                img = img.half().to(self.device)

                assert H == W
                crop_xyxy_scaled = crop_xywh.float() / H
                crop_xyxy_scaled[:, 2] += crop_xyxy_scaled[:, 0]
                crop_xyxy_scaled[:, 3] += crop_xyxy_scaled[:, 1]
                crop_xyxy_scaled = crop_xyxy_scaled.to(self.device)

                valid_xyxy_scaled = valid_xywh.float() / H
                valid_xyxy_scaled[:, 2] += valid_xyxy_scaled[:, 0]
                valid_xyxy_scaled[:, 3] += valid_xyxy_scaled[:, 1]
                valid_xyxy_scaled = valid_xyxy_scaled.to(self.device)

                # if self.use_depth or self.use_canny:
                #     img = gaussian_blur(img, (21, 21), (7, 7))

                cond = [img]
                if self.use_subject:
                    cond.append(batch['subject'].half().to(self.device))
                cond = torch.cat(cond, dim=1) if len(cond) > 1 else cond[0]

                with torch.autocast(self.device, dtype=torch.float16):
                    pred, pred_all, _ = self.backbone(
                        cond, style_cond.to(self.device), subject_xyxy)

                    # loss against ground truth
                    loss = F.l1_loss(pred, crop_xyxy_scaled)

                    loss += 0.1 * F.l1_loss(
                        pred_all, crop_xyxy_scaled.unsqueeze(1).repeat(
                            1, pred_all.shape[1], 1))

                    # loss against subject edge
                    if self.subject_boundary_loss:
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
    bad_img_file = os.path.join(args.dataset, 'bad.json')

    train_dataset = ConditionalDatasetWrapper(OutpaintImageCropDataset(
        os.path.join(args.dataset, '..', 'train.json'),
        os.path.join(args.dataset, 'images'),
        det_dict,
        args.dim, augment=True,
        bad_img_file=bad_img_file, mask_file=mask_file,
        invert_prob=0 if args.no_invert else 0.2))
    train_dataset.print_info()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=num_workers,
        shuffle=True)

    val_dataset = ConditionalDatasetWrapper(OutpaintImageCropDataset(
        os.path.join(args.dataset, '..', 'val.json'),
        os.path.join(args.dataset, 'images'),
        det_dict,
        args.dim, bad_img_file=bad_img_file, mask_file=mask_file,
        invert_prob=0 if args.no_invert else 0.2))
    val_dataset.print_info()
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=num_workers)
    del det_dict

    model = GenCropC(
        use_subject=args.use_subject, in_dim=args.dim,
        asymmetric_loss=args.asym, subject_boundary_loss=not args.no_sbl)
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=0.0001)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * args.num_epochs))
    scaler = torch.cuda.amp.GradScaler()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        store_json(os.path.join(args.save_dir, 'config.json'), {
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'img_dim': args.dim,
            'use_subject': args.use_subject,
            'use_subject_boundary_loss': not args.no_sbl,
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