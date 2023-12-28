import os
import random
from collections import defaultdict
from typing import NamedTuple
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .io import load_json, load_json, list_images
from .box import Box
from .outpaint import load_outpaint_split


V_ASPECT_RATIOS = [1, 4/5, 3/4, 5/7, 2/3]
H_ASPECT_RATIOS = [1, 4/5, 3/4, 5/7, 2/3, 9/16]


def hflip_xywh(xywh, x_max):
    return torch.tensor((x_max - xywh[0] - xywh[2], xywh[1], xywh[2], xywh[3]),
                        dtype=xywh.dtype)


def xywh_to_mask(W, H, xywh, device='cpu'):
    if len(xywh.shape) == 2:
        mask = torch.zeros((xywh.shape[0], 1, W, H), dtype=torch.bool,
                           device=device)
        for i in range(xywh.shape[0]):
            mask[i, :, xywh[i, 1]:xywh[i, 1] + xywh[i, 3],
                 xywh[i, 0]:xywh[i, 0] + xywh[i, 2]] = 1.
    else:
        mask = torch.zeros((1, W, H), dtype=torch.bool, device=device)
        mask[0, xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1.
    return mask


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


ImageNetUnnormalize = transforms.Normalize(
    mean=[-IMAGENET_MEAN[i] / IMAGENET_STD[i] for i in range(3)],
    std=[1 / IMAGENET_STD[i] for i in range(3)])


class PadCropLoader:

    def __init__(self, dim, augment, same_elastic=False):
        self.dim = dim

        T = [transforms.ToTensor()]
        if augment:
            T.append(transforms.ColorJitter(
                brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05))
            T.append(transforms.GaussianBlur(
                kernel_size=(5, 5), sigma=(0.1, 5)))
            T.append(transforms.RandomGrayscale(p=0.05))
            if not same_elastic:
                T.append(transforms.RandomApply([
                    transforms.ElasticTransform()]))
        T.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.img_transform = transforms.Compose(T)

        self.subject_transform = None
        if augment and not same_elastic:
            self.subject_transform = transforms.RandomApply(
                [transforms.ElasticTransform()])

        self.elastic_transform = None
        if augment and same_elastic:
            self.elastic_transform = transforms.RandomApply(
                [transforms.ElasticTransform()])

    def xywh_to_mask(self, xywh):
        return xywh_to_mask(self.dim, self.dim, xywh)

    def scale_and_pad(self, img, img_data, x, y, w, h,
                      subject_mask=None):
        # Pad longest side to dim, then pad other side with 0s
        if w > h:
            scale = self.dim / w
            new_w = self.dim
            new_h = round(scale * h)
            pad_x = 0
            pad_y = (self.dim - new_h) // 2
        else:
            scale = self.dim / h
            new_h = self.dim
            new_w = round(scale * w)
            pad_y = 0
            pad_x = (self.dim - new_w) // 2
        img = cv2.resize(img, (new_w, new_h))

        # Convert img to tensor and pad
        img = self.img_transform(img)
        img = F.pad(img, (
            pad_x, self.dim - pad_x - new_w, pad_y, self.dim - pad_y - new_h))
        assert img.shape == (3, self.dim, self.dim), img.shape

        # Apply same transform to subject mask
        if subject_mask is not None:
            subject_mask = torch.tensor(cv2.resize(
                subject_mask, (new_w, new_h)) > 0, dtype=float).unsqueeze(0)
            subject_mask = F.pad(
                subject_mask, (pad_x, self.dim - pad_x - new_w,
                               pad_y, self.dim - pad_y - new_h))
            if self.subject_transform is not None:
                subject_mask = self.subject_transform(subject_mask)

        # Calculate gt crop after scaling and padding
        if isinstance(img_data.crop, list):
            crop_xywh = []
            for crop in img_data.crop:
                crop_x = round(scale * (crop.x - x)) + pad_x
                crop_y = round(scale * (crop.y - y)) + pad_y
                crop_w = round(scale * crop.w)
                crop_h = round(scale * crop.h)
                crop_xywh.append(torch.tensor(
                    (crop_x, crop_y, crop_w, crop_h), dtype=torch.int16))
            crop_xywh = torch.stack(crop_xywh)
        else:
            crop_x = round(scale * (img_data.crop.x - x)) + pad_x
            crop_y = round(scale * (img_data.crop.y - y)) + pad_y
            crop_w = round(scale * img_data.crop.w)
            crop_h = round(scale * img_data.crop.h)
            crop_xywh = torch.tensor(
                (crop_x, crop_y, crop_w, crop_h), dtype=torch.int16)

        # Calculate subject bbox after scaling and padding
        subj_xywh = None
        if img_data.subject is not None:
            subj_x = round(scale * (img_data.subject.x - x)) + pad_x
            subj_y = round(scale * (img_data.subject.y - y)) + pad_y
            subj_x2 = subj_x + round(scale * img_data.subject.w)
            subj_y2 = subj_y + round(scale * img_data.subject.h)
            subj_x = max(pad_x, subj_x)
            subj_y = max(pad_y, subj_y)
            subj_w = min(pad_x + new_w, subj_x2) - subj_x
            subj_h = min(pad_y + new_h, subj_y2) - subj_y
            del subj_x2, subj_y2
            subj_xywh = torch.tensor(
                (subj_x, subj_y, subj_w, subj_h), dtype=torch.int16)

        # Calculate valid bbox after scaling and padding
        valid_xywh = torch.tensor((pad_x, pad_y, new_w, new_h),
                                  dtype=torch.int16)

        if self.elastic_transform is not None:
            if subject_mask is not None:
                tmp = torch.cat((img, subject_mask), dim=0)
                tmp = self.elastic_transform(tmp)
                img = tmp[:3]
                subject_mask = tmp[3:4]
            else:
                img = self.elastic_transform(img)

        ret = {'img': img,
               'valid_xywh': valid_xywh,
               'crop_xywh': crop_xywh,
               'scale': scale}
        if subj_xywh is not None:
            ret['subject_xywh'] = subj_xywh
        if subject_mask is not None:
            ret['subject'] = subject_mask
        return ret


def sample_ex_uniform(e_p=0.25):
    if random.random() < e_p:
        return int(random.random() < 0.5)
    else:
        return random.uniform(0, 1)


def sample_crop(img_data, min_scale, max_scale, invert_prob=0.2,
                sampler=random.random):
    is_portrait = img_data.crop.h > img_data.crop.w

    i = 0
    while True:
        if not is_portrait or random.random() >= invert_prob:
            eff_max_scale = max_scale
            aspect = random.choice(
                V_ASPECT_RATIOS if is_portrait else H_ASPECT_RATIOS)
        else:
            # Invert orientation if portrait
            eff_max_scale = 1.1 * min_scale
            aspect = 1. / random.choice(H_ASPECT_RATIOS)

        # Always sample the long side first
        if is_portrait:
            h = int(random.uniform(
                min_scale * img_data.crop.h,
                min(eff_max_scale * img_data.crop.h, img_data.height)))
            w = int(aspect * h)
        else:
            w = int(random.uniform(
                min_scale * img_data.crop.w,
                min(eff_max_scale * img_data.crop.w, img_data.width)))
            h = int(aspect * w)

        if (h >= img_data.crop.h and w >= img_data.crop.w
            and h <= img_data.height and w <= img_data.width
        ):
            break

        i += 1
        if i % 1000 == 0:
            print('Unable to find valid crop! it={} crop={}'.format(
                i, img_data.crop))

    # print(h, w, img_data.crop.x, img_data.crop.y,
    #     img_data.crop.h, img_data.crop.w)

    # The crop is a valid one
    x_min = max(0, img_data.crop.x2 - w)
    x_max = min(img_data.width - w, img_data.crop.x)
    if x_min != x_max:
        x = round(sampler() * (x_max - x_min)) + x_min
    else:
        x = x_min

    y_min = max(0, img_data.crop.y2 - h)
    y_max = min(img_data.height - h, img_data.crop.y)
    if y_min != y_max:
        y = round(sampler() * (y_max - y_min)) + y_min
    else:
        y = y_min
    return x, y, w, h


def augment_item(ret, dim):
    # Jitter subject bbox
    if 'subject_xywh' in ret:
        sx, sy, sw, sh = ret['subject_xywh'].tolist()
        sx2, sy2 = sx + sw, sy + sh
        sx += int(random.uniform(-0.1, 0.1) * sw)
        sx2 += int(random.uniform(-0.1, 0.1) * sw)
        sy += int(random.uniform(-0.1, 0.1) * sh)
        sy2 += int(random.uniform(-0.1, 0.1) * sh)
        vx, vy, vw, vh = ret['valid_xywh'].tolist()
        sx = max(vx, sx)
        sy = max(vy, sy)
        ret['subject_xywh'] = torch.tensor(
            [sx, sy, min(vx + vw, sx2) - sx, min(vy + vh, sy2) - sy],
            dtype=torch.int16)

    # Horizontal flip
    if random.random() > 0.5:
        ret['img'] = torch.flip(ret['img'], dims=(2,))
        if 'subject' in ret:
            ret['subject'] = torch.flip(ret['subject'], dims=(2,))

        ret['valid_xywh'] = hflip_xywh(ret['valid_xywh'], dim)
        assert ret['crop_xywh'].shape == (4,)
        ret['crop_xywh'] = hflip_xywh(ret['crop_xywh'], dim)

        if 'subject_xywh' in ret:
            ret['subject_xywh'] = hflip_xywh(ret['subject_xywh'], dim)
    return ret


class OutpaintImageCropDataset(Dataset):

    def __init__(self, split_file, img_dir, det_dict, dim,
                 augment=False,
                 stratify=True,
                 min_scale=1., max_scale=2,
                 crop_and_valid_mask=False,
                 bad_img_file=None,
                 mask_file=None,
                 invert_prob=0.2,
                 heuristic_filter=True
    ):
        self.loader = PadCropLoader(dim, augment)

        self.split_file = split_file
        self.img_dir = img_dir
        self.invert_prob = invert_prob

        # HACK: lazy load in each worker
        self.mask_file = mask_file
        self.mask_data = None

        img_files = list_images(img_dir)
        if bad_img_file is not None:
            bad_imgs = set(load_json(bad_img_file))
            img_files = list(filter(lambda x: x not in bad_imgs, img_files))

        imgs = list(load_outpaint_split(
            load_json(split_file), img_files, det_dict,
            filter_with_heuristics=heuristic_filter))
        if stratify:
            tmp = defaultdict(list)
            for img in imgs:
                tmp[img.id].append(img)
            self.imgs = list(tmp.values())
        else:
            self.imgs = [[x] for x in imgs]

        self.augment = augment

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_and_valid_mask = crop_and_valid_mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_data = random.choice(self.imgs[index])
        x, y, w, h = sample_crop(
            img_data, self.min_scale, self.max_scale,
            sampler=sample_ex_uniform, invert_prob=self.invert_prob)

        img = cv2.imread(os.path.join(self.img_dir, img_data.img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img.shape[:2] == (img_data.height, img_data.width)

        subject = None
        if img_data.subject is not None:
            subject = np.zeros((img_data.height, img_data.width),
                               dtype=np.uint8)
            if self.mask_data is None:
                # Lazy loading
                self.mask_data = np.load(self.mask_file, mmap_mode='r')

            cv2.fillPoly(
                subject,
                [self.mask_data[str(img_data.subject.payload)].astype(int)], 1)
            # Apply crop
            subject = subject[y:y + h, x:x + w]

        ret = self.loader.scale_and_pad(
            img[y:y + h, x:x + w], img_data, x, y, w, h, subject_mask=subject)
        if self.augment:
            ret = augment_item(ret, self.loader.dim)

        if self.crop_and_valid_mask:
            ret['valid'] = self.loader.xywh_to_mask(ret['valid_xywh'])
            ret['crop'] = self.loader.xywh_to_mask(ret['crop_xywh'])

        if 'subject' in ret:
            non_zero = torch.nonzero(ret['subject'])
            if len(non_zero) > 0:
                smx1 = non_zero[:, 2].min()
                smy1 = non_zero[:, 1].min()
                ret['subject_mask_xywh'] = torch.tensor([
                    smx1, smy1, non_zero[:, 2].max() - smx1,
                    non_zero[:, 1].max() - smy1])
            else:
                ret['subject_mask_xywh'] = ret['valid_xywh']

        ret['file'] = img_data.img_file
        return ret

    def print_info(self):
        print('{}: {} images ({} variations)'.format(
            self.split_file, len(self.imgs), sum(len(x) for x in self.imgs)))


class BenchmarkDatasetBase(Dataset):

    class Image(NamedTuple):
        name: str
        img_dir: str
        subject: Box
        crop: Box

        subject_dir: str = None

    def __init__(self, imgs, dim, augment=False,
                 crop_and_valid_mask=False, mask_ext='.png'):
        self.augment = augment
        self.loader = PadCropLoader(dim, augment)
        self.mask_ext = mask_ext

        self.imgs = imgs

        self.crop_and_valid_mask = crop_and_valid_mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index, orig_size_crop=False):
        img_data = self.imgs[index]
        img_path = os.path.join(img_data.img_dir, img_data.name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if img_data.subject_dir is not None:
            subject = np.array(Image.open(os.path.join(
                img_data.subject_dir,
                os.path.splitext(img_data.name)[0] + self.mask_ext)))
            subject = (subject > 127).astype(np.uint8)
        elif img_data.subject is not None:
            assert len(img_data.subject.payload.shape) == 2
            if img_data.subject.payload.shape[1] == 2:
                # Polygon form
                subject = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(
                    subject, [img_data.subject.payload.astype(int)], 1)
            else:
                # Mask form
                subject = img_data.subject.payload
        else:
            subject = None

        ret = self.loader.scale_and_pad(
            img, img_data, 0, 0, w, h, subject_mask=subject)

        if self.augment:
            ret = augment_item(ret, self.loader.dim)

        if self.crop_and_valid_mask:
            ret['valid'] = self.loader.xywh_to_mask(ret['valid_xywh'])
            if len(ret['crop_xywh'].shape) == 1:
                # HACK: only for single crop
                ret['crop'] = self.loader.xywh_to_mask(ret['crop_xywh'])

        if orig_size_crop:
            # Get other crops, for evaluation purposes
            if isinstance(img_data.crop, list):
                orig_crop_xywh = []
                orig_crop_score = []
                for crop in img_data.crop:
                    orig_crop_xywh.append(crop.xywh)
                    orig_crop_score.append(crop.score)
                ret['orig_crop_score'] = torch.tensor(orig_crop_score)
            else:
                orig_crop_xywh = [img_data.crop.x, img_data.crop.y,
                                  img_data.crop.w, img_data.crop.h]
            ret['orig_crop_xywh'] = torch.tensor(orig_crop_xywh)
            ret['orig_w'] = w
            ret['orig_h'] = h
        ret['file'] = img_path
        return ret