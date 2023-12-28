import os
from typing import NamedTuple
from tqdm import tqdm

from .box import Box


class OutpaintImage(NamedTuple):
    id: str
    img_file: str
    crop: Box
    subject: Box
    subject_orig: Box
    height: int = 512
    width: int = 512


def parse_outpaint_image_name(img_file):
    tmp = img_file[3:-4]
    img_id, x, y, w, h = tmp.rsplit('_', 4)
    return img_id, int(x), int(y), int(w), int(h)


def load_outpaint_split(split_data, img_files, det_dict=None,
                        filter_with_heuristics=True):
    split_dict = {x['id']: x for x in split_data}

    for img_file in tqdm(img_files):
        img_name = os.path.splitext(img_file)[0]
        img_id, x, y, w, h = parse_outpaint_image_name(img_file)
        img_data = split_dict.get(img_id)
        if img_data is None:
            continue
        crop = Box(x, y, w, h)

        # Calculate size of the subject in the outpainted dataset
        orig_subject_xywh = img_data['subject_xywh']
        if orig_subject_xywh is not None:
            orig_subject_scaled = Box(
                x + orig_subject_xywh[0] * w,
                y + orig_subject_xywh[1] * h,
                orig_subject_xywh[2] * w,
                orig_subject_xywh[3] * h,
                1.)

            # Find largest IoU detection with the subject
            new_subject = None
            new_subject_iou = 0.25
            largest_hallucinated = None
            det_boxes = [Box(*d['xywh'], payload=d['mask_id'])
                         for d in det_dict[img_name]]
            for det in det_boxes:
                iou = det.iou(orig_subject_scaled)
                if iou > new_subject_iou:
                    new_subject = det
                    new_subject_iou = iou

                # Filter images where a hallucinated subject is of similar size or larger than the original subject
                if (det.iou(crop) < 0.01 and (
                    largest_hallucinated is None
                    or det.area > largest_hallucinated.area)
                ):
                    largest_hallucinated = det

            if new_subject is None:
                continue
            if filter_with_heuristics:
                if (largest_hallucinated is not None
                    and largest_hallucinated.area >
                        0.25 * orig_subject_scaled.area):
                    continue
                if max(d.area for d in det_boxes) > 2 * new_subject.area:
                    continue
        else:
            orig_subject_scaled = None
            new_subject = None
        yield OutpaintImage(img_id, img_file, crop, new_subject,
                            orig_subject_scaled)
