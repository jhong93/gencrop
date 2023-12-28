import os
import json
import cv2
import numpy as np

from .io import load_json, load_gz_json, decode_png, list_images
from .dataset import BenchmarkDatasetBase
from .box import Box, xyxy2xywh

from config import BenchmarkConfig, UNSPLASH_IMAGE_DIR


class BenchmarkUnsplashDataset(BenchmarkDatasetBase):

    def __init__(self, label_dir, dim):
        self.img_dir = UNSPLASH_IMAGE_DIR

        assert os.path.isdir(label_dir), \
            'Dataset should be the directory containing testeval_sha256.json.gz'
        self.label_file = os.path.join(label_dir, 'testeval_sha256.json.gz')
        labels = load_gz_json(self.label_file)

        id_to_hash = {x['hash_id']: x['id']
                      for x in load_json(os.path.join(label_dir, 'test.json'))}

        imgs = []
        for hash_id in sorted(labels):
            img_id = id_to_hash.get(hash_id)
            if img_id is None:
                print('Unable to find image for hash_id: {}'.format(hash_id))

            data_img = labels[hash_id]

            img_file = 'img{}.jpg'.format(img_id)
            H, W = cv2.imread(os.path.join(self.img_dir, img_file)).shape[:2]

            crop = data_img['crop_xywh']
            if isinstance(crop[0], list):
                crop = [Box(x[0] * W, x[1] * H, x[2] * W, x[3] * H)
                        for x in crop]
            else:
                crop = Box(crop[0] * W, crop[1] * H, crop[2] * W, crop[3] * H)

            img = BenchmarkDatasetBase.Image(
                img_file, self.img_dir,
                Box(*data_img['subject_xywh'],
                    payload=decode_png(data_img['subject_mask']))
                    if data_img['subject_xywh'] is not None else None,
                crop)
            imgs.append(img)
        super().__init__(imgs, dim, crop_and_valid_mask=True)

    def __getitem__(self, idx):
        return super().__getitem__(idx, orig_size_crop=True)

    def print_info(self):
        print('{}: {} images'.format(self.label_file, len(self)))


class BenchmarkHumanDataset(BenchmarkDatasetBase):

    def __init__(self, datasets, dim, human_only):
        self.datasets = []

        imgs = []
        for dataset_name in datasets:
            # Annoyingly, each dataset has slightly different formats
            dataset_name = dataset_name.upper()

            bboxes = None
            if dataset_name == 'CPC':
                if human_only:
                    bboxes = load_json(os.path.join(
                        BenchmarkConfig.CPC_DIR, 'human_bboxes.json'))
                img_dir = os.path.join(BenchmarkConfig.CPC_DIR, 'images')
                crops = load_json(os.path.join(
                    BenchmarkConfig.CPC_DIR, 'image_crop.json'))
                det_file = os.path.join(
                    BenchmarkConfig.CPC_DIR, 'detect.json.gz')
                mask_file = os.path.join(
                    BenchmarkConfig.CPC_DIR, 'mask.npz')

                def process(fname):
                    if bboxes is not None and fname not in bboxes:
                        return None
                    return BenchmarkDatasetBase.Image(
                        fname, img_dir,
                        Box(*xyxy2xywh(bboxes[fname]['bbox'])
                            if bboxes is not None else None),
                        Box(*xyxy2xywh(crops[fname]['bboxes'][0])))

            elif dataset_name == 'GAICD':
                if human_only:
                    bboxes = load_json(os.path.join(
                        BenchmarkConfig.GAICD_DIR, 'human_bboxes.json'))
                    split = load_json(os.path.join(
                        BenchmarkConfig.GAICD_DIR, 'human_data_split.json'))
                else:
                    split = load_json(os.path.join(
                        BenchmarkConfig.GAICD_DIR, 'original_data_split.json'))

                img_dir = os.path.join(BenchmarkConfig.GAICD_DIR, 'images')
                crops = load_json(os.path.join(
                    BenchmarkConfig.GAICD_DIR, 'image_crop.json'))

                det_file = os.path.join(
                    BenchmarkConfig.GAICD_DIR, 'detect.json.gz')
                mask_file = os.path.join(
                    BenchmarkConfig.GAICD_DIR, 'mask.npz')

                # ids = set(split['train'] + split['test'])
                ids = set(split['test'])

                def process(fname):
                    if fname not in ids:
                        return None
                    return BenchmarkDatasetBase.Image(
                        fname, img_dir,
                        Box(*xyxy2xywh(bboxes[fname]))
                            if bboxes is not None else None,
                        [Box(*xyxy2xywh(x), score=s) for x, s in
                         zip(crops[fname]['bbox'], crops[fname]['score'])])

            elif dataset_name == 'FLMS':
                if human_only:
                    bboxes = load_json(os.path.join(
                        BenchmarkConfig.FLMS_DIR, 'human_bboxes.json'))

                img_dir = os.path.join(BenchmarkConfig.FLMS_DIR, 'image')
                crops = load_json(os.path.join(
                    BenchmarkConfig.FLMS_DIR, 'image_crop.json'))

                det_file = os.path.join(
                    BenchmarkConfig.FLMS_DIR, 'detect.json.gz')
                mask_file = os.path.join(
                    BenchmarkConfig.FLMS_DIR, 'mask.npz')

                def process(fname):
                    if bboxes is not None and fname not in bboxes:
                        return None
                    return BenchmarkDatasetBase.Image(
                        fname, img_dir,
                        Box(*xyxy2xywh(bboxes[fname]))
                            if bboxes is not None else None,
                        [Box(*xyxy2xywh(b)) for b in crops[fname]])

            elif dataset_name == 'FCDB':
                if human_only:
                    bboxes = load_json(os.path.join(
                        BenchmarkConfig.FCDB_DIR, 'human_bboxes.json'))
                img_dir = os.path.join(BenchmarkConfig.FCDB_DIR, 'data')
                crops = load_json(os.path.join(
                    BenchmarkConfig.FCDB_DIR, 'image_crop.json'))

                split = load_json(os.path.join(
                    BenchmarkConfig.FCDB_DIR, 'data_split.json'))
                if human_only:
                    ids = set(split['train'] + split['test'])
                else:
                    ids = set(split['test'])

                det_file = os.path.join(
                    BenchmarkConfig.FCDB_DIR, 'detect.json.gz')
                mask_file = os.path.join(
                    BenchmarkConfig.FCDB_DIR, 'mask.npz')

                def process(fname):
                    if bboxes is not None and fname not in bboxes:
                        return None
                    if fname not in crops:
                        return None
                    if fname not in ids:
                        return None
                    return BenchmarkDatasetBase.Image(
                        fname, img_dir,
                        Box(*xyxy2xywh(bboxes[fname]))
                            if bboxes is not None else None,
                        Box(*crops[fname]))

            else:
                raise NotImplementedError(dataset_name)

            det_dict = load_gz_json(det_file)
            mask_data = np.load(mask_file, mmap_mode='r')

            for img_file in list_images(img_dir):
                img = process(img_file)

                if img is not None:
                    if img.subject is not None:
                        # Fill in subject mask
                        best_iou = 0
                        best_det = None
                        for d in det_dict[os.path.splitext(img.name)[0]]:
                            d = Box(*d['xywh'],
                                    payload=mask_data[str(d['mask_id'])])
                            iou = d.iou(img.subject)
                            if iou > best_iou:
                                best_iou = iou
                                best_det = d
                        if best_det is None:
                            print('No subject found!')
                            img.subject._payload = np.array([
                                (img.subject.x, img.subject.y),
                                (img.subject.x, img.subject.y2),
                                (img.subject.x2, img.subject.y2),
                                (img.subject.x2, img.subject.y),
                            ])
                        else:
                            img.subject._payload = best_det.payload
                    imgs.append(img)
            self.datasets.append(dataset_name)

        super().__init__(imgs, dim, crop_and_valid_mask=True)

    def __getitem__(self, idx):
        return super().__getitem__(idx, orig_size_crop=True)

    def print_info(self):
        print('{}: {} images'.format(' + '.join(self.datasets), len(self)))


class BenchmarkSACDDataset(BenchmarkDatasetBase):

    def __init__(self, dim):
        img_dir = os.path.join(BenchmarkConfig.SACD_DIR, 'images-sym')
        subj_dir = os.path.join(BenchmarkConfig.SACD_DIR, 'masks')
        split_data = load_json(
            os.path.join(BenchmarkConfig.SACD_DIR, 'test.json'))

        imgs = []
        for annot_file in split_data['jsons']:
            annot_data = load_json(
                os.path.join(BenchmarkConfig.SACD_DIR, annot_file))

            img_file = annot_file.split('/', 1)[-1].rsplit('.', 1)[0] + '.jpg'
            subj = json.loads(annot_data['mask_box'])
            imgs.append(BenchmarkDatasetBase.Image(
                img_file, img_dir,
                Box(*subj), # why is this one already xywh?
                [Box(*xyxy2xywh(annot_data['best_box'][0]))] +
                [Box(*xyxy2xywh(b)) for b in annot_data['crop_box']],
                subject_dir=subj_dir))

        super().__init__(imgs, dim,
                         crop_and_valid_mask=True, mask_ext='.jpg')

    def __getitem__(self, idx):
        return super().__getitem__(idx, orig_size_crop=True)

    def print_info(self):
        print('SACD: {} images'.format(len(self)))