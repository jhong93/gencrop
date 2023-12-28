#!/usr/bin/env python3

import os
import argparse
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import segmentation_models_pytorch as smp
from tqdm import tqdm

from util.io import load_gz_json, store_json
from util.dataset import OutpaintImageCropDataset

cv2.setNumThreads(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to the outpainted dataset')
    parser.add_argument('--dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_epoch_len', type=int)
    parser.add_argument('--use_subject', action='store_true',
                        help='Use the subject mask as input')
    parser.add_argument('--no_invert', action='store_true')
    parser.add_argument('-s', '--save_dir')
    return parser.parse_args()


class GenCropU:

    def __init__(self, use_subject, device='cuda'):
        self.use_subject = use_subject
        in_channel = 3
        if use_subject:
            in_channel += 1

        backbone = smp.Unet(
            encoder_name='resnet50', encoder_weights='imagenet',
            in_channels=in_channel, classes=1)
        backbone.to(device)
        self.backbone = backbone
        self.device = device

    def predict(self, img, subject=None):
        self.backbone.eval()

        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16):
            img = img.half().to(self.device)

            cond = [img]
            if self.use_subject:
                cond.append(subject.half().to(self.device))
            cond = torch.cat(cond, dim=1) if len(cond) > 1 else cond[0]

            pred = F.sigmoid(self.backbone(cond))
            return pred.cpu()

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

        with tqdm(loader) as pbar:
            for batch in pbar:
                img, valid, crop = \
                    batch['img'], batch['valid'], batch['crop']

                B = img.shape[0]
                img = img.half().to(self.device)
                valid = valid.half().to(self.device)
                crop = crop.half().to(self.device)

                cond = [img]
                if self.use_subject:
                    cond.append(batch['subject'].half().to(self.device))
                cond = torch.cat(cond, dim=1) if len(cond) > 1 else cond[0]

                with torch.autocast(self.device, dtype=torch.float16):
                    pred = F.sigmoid(self.backbone(cond))
                    loss = F.mse_loss(pred * valid, crop)

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

    print(os.path.join(args.dataset, '..', 'train.json'))
    print(os.path.exists(os.path.join(args.dataset, '..', 'train.json')))

    train_dataset = OutpaintImageCropDataset(
        os.path.join(args.dataset, '..', 'train.json'),
        os.path.join(args.dataset, 'images'),
        det_dict,
        args.dim, augment=True,
        bad_img_file=bad_img_file, crop_and_valid_mask=True,
        mask_file=mask_file, invert_prob=0 if args.no_invert else 0.2)
    train_dataset.print_info()

    val_dataset = OutpaintImageCropDataset(
        os.path.join(args.dataset, '..', 'val.json'),
        os.path.join(args.dataset, 'images'),
        det_dict,
        args.dim, bad_img_file=bad_img_file,
        crop_and_valid_mask=True, mask_file=mask_file,
        invert_prob=0 if args.no_invert else 0.2)
    val_dataset.print_info()
    del det_dict

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=num_workers,
        shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=num_workers)

    model = GenCropU(use_subject=args.use_subject)
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=0.0001)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=(len(train_loader) * args.num_epochs))
    scaler = torch.cuda.amp.GradScaler()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        store_json(os.path.join(args.save_dir, 'config.json'), {
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'img_dim': args.dim,
            'use_subject': args.use_subject
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