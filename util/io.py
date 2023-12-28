import os
from io import BytesIO
import json
import base64
import gzip
import numpy as np
from PIL import Image


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def store_json(fpath, obj, indent=None):
    kwargs = {}
    if indent is not None:
        kwargs['indent'] = indent
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj, indent=None):
    kwargs = {}
    if indent is not None:
        kwargs['indent'] = indent
    with gzip.open(fpath, 'wt', encoding='ascii') as fp:
        json.dump(obj, fp, **kwargs)


def load_text(fpath):
    with open(fpath, 'r') as fp:
        return fp.read().strip()


def store_text(fpath, s):
    with open(fpath, 'w') as fp:
        fp.write(s)


def list_images(img_dir, exts=('.jpg', '.png')):
    return [x for x in os.listdir(img_dir) if x.endswith(exts)]


def decode_png(data):
    if isinstance(data, str):
        data = base64.decodebytes(data.encode())
    else:
        assert isinstance(data, bytes)
    fstream = BytesIO(data)
    im = Image.open(fstream)
    return np.array(im)