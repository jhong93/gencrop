#!/usr/bin/env python3

"""
Binary outpaint quality classifier.

The image directory should contain two subdirectories (good and bad) which contain the images of the respective label.
"""

import os
import argparse
import random
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
import timm
from sklearn.metrics import classification_report
from tqdm import tqdm

from util.io import store_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--arch', required=True, choices=['resnet50'])
    parser.add_argument('-s', '--save_dir')
    return parser.parse_args()


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageDataset(Dataset):

    def __init__(self, img_files, img_dim, label_dict={}, augment=False,
                 transform=None):
        self.imgs = img_files
        self.label_dict = label_dict
        self.augment = augment

        if transform is None:
            T = []
            if augment:
                T.extend([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
                ])
            T.extend([
                transforms.Resize((img_dim, img_dim)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

            self.transform = transforms.Compose(T)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label, img_file = self.imgs[idx]
        return self.label_dict.get(label, -1), \
            self.transform(Image.open(img_file)), img_file

    def print_info(self):
        print('Dataset: {} (augment={})'.format(len(self.imgs), self.augment))


class QualityClassifier:

    def __init__(self, arch, device='cuda'):
        self.device = device

        self.preprocess = lambda x: x
        self.model = timm.create_model(
            arch, num_classes=2, pretrained=True)
        self.model.to(device)

    def predict(self, img):
        self.model.eval()
        with torch.no_grad():
            return F.softmax(self.model(img.to(self.device)), dim=1)

    def epoch(self, loader, optimizer=None, lr_scheduler=None, scaler=None,
              weight=None):
        if optimizer is None:
            self.model.eval()
        else:
            self.model.train()
            optimizer.zero_grad()

        epoch_pred = []
        epoch_gt = []
        epoch_loss = 0.
        N = 0

        weight = torch.tensor(weight, device=self.device) if weight else None
        for label, img, _ in tqdm(loader):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                pred = self.model(self.preprocess(img.to(self.device)))
                loss = F.cross_entropy(
                    pred, label.to(self.device),
                    label_smoothing=0.1, weight=weight)

            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_pred.extend(pred.argmax(dim=1).detach().cpu().tolist())
            epoch_gt.extend(label.detach().cpu().tolist())
            epoch_loss += loss.item()
            N += img.shape[0]

        epoch_pred = np.array(epoch_pred)
        epoch_gt = np.array(epoch_gt)

        epoch_acc = np.sum(epoch_pred == epoch_gt) / N
        return epoch_loss / N, epoch_acc, (epoch_pred, epoch_gt)


def load_datasets(img_dir, img_dim):
    label_dict = {'good': 0, 'bad': 1}

    img_files = []
    for label in sorted(os.listdir(img_dir)):
        label_dir = os.path.join(img_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for img_file in os.listdir(label_dir):
            if img_file.endswith('.jpg'):
                img_files.append((label, os.path.join(label_dir, img_file)))

    random.shuffle(img_files)
    val_count = 500
    train_dataset = ImageDataset(img_files[val_count:], img_dim, label_dict,
                                 augment=True)
    train_dataset.print_info()
    val_dataset = ImageDataset(img_files[:val_count], img_dim, label_dict)
    val_dataset.print_info()
    return train_dataset, val_dataset


def main(args):
    trainer = QualityClassifier(arch=args.arch)

    train_dataset, val_dataset = load_datasets(args.img_dir, args.dim)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=8)

    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=3 * len(train_loader),
        num_training_steps=(len(train_loader) * args.num_epochs))

    weight = [1., 10.]
    for epoch in range(args.num_epochs):
        train_loss, train_acc, _ = trainer.epoch(
            train_loader, optimizer, scheduler, scaler, weight)
        val_loss, val_acc, (val_pred, val_gt) = trainer.epoch(
            val_loader, weight=weight)
        print('Epoch: {} Train loss: {:.4f} Train acc: {:.4f} '
              'Val loss: {:.4f} Val acc: {:.4f}'.format(
                  epoch, train_loss, train_acc, val_loss, val_acc))

        print(classification_report(
            val_gt, val_pred, target_names=['good', 'bad']))

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'config.json'), {
                'arch': args.arch,
                'dim': args.dim,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'weight': weight,
                'classes': train_dataset.label_dict
            }, indent=2)
            torch.save(trainer.model.state_dict(),
                       os.path.join(args.save_dir, 'model.pt'))

    print('Done!')


if __name__ == '__main__':
    main(get_args())