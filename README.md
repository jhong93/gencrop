# Learning Subject-Aware Cropping by Outpainting Professional Photos

This repository contains code for our paper:

*Learning Subject-Aware Cropping by Outpainting Professional Photos*\
In AAAI 2024\
James Hong, Lu Yuan, Michael Gharbi, Matthew Fisher, Kayvon Fatahalian

Links: [project](https://jhong93.github.io/projects/crop.html), [arXiv](https://arxiv.org/abs/2312.12080)

```bibtex
@inproceedings{gencrop_aaai24,
    author={Hong, James and Yuan, Lu and Gharbi, Micha\"{e}l and Fisher, Matthew and Fatahalian, Kayvon},
    title={Learning Subject-Aware Cropping by Outpainting Professional Photos},
    booktitle={AAAI},
    year={2024}
}
```

This code is released under the BSD-3 [LICENSE](/LICENSE).

## Stock Image Dataset

We use unsplash in the paper. To download the dataset, see [Unsplash](#unsplash).

### Unsplash

To use the Unsplash dataset, you must request access from [Unsplash](https://unsplash.com/data). Once you have access, download the dataset and extract the files.

We provide SHA256 hashes of the image ids used in our paper. To convert these back to the original image ids, use the following command:
```python3 prepare_unsplash.py <unsplash-dataset-path>```
This will create files in the `data/` directory.

For example, after this step, your `data/portrait` directory should contain the following files:
```
ids_sha256.json  test.json         train.json         val.json
images.json      test_sha256.json  train_sha256.json  val_sha256.json
```

`images.json` contains the list of image ids. `<split>.json` contains a list of objects:
```
[{
    "id": image_id,                   # image id
    "subject_xywh": [x, y, w, h],     # subject bounding box (in 0 - 1)
    "hash_id": hash_id                # SHA256 hash of image id
}, ...]
```

### Use your own dataset

To use your own dataset, you should create a directory with the following structure:
```
dataset/
    train.json
    val.json
    test.json
```

Each json file should contain a list of objects:
```
[{
    "id": image_id,                   # image id
    "subject_xywh": [x, y, w, h],     # subject bounding box (in 0 - 1)
}, ...]
```

The images should be in a directory. We prefix each image id in Unsplash with `img` in order to avoid odd file names starting with dashes. You can do the same, or modify the code to remove this prefix. An easy solution that does not require renaming files is to make a directory of symbolic links to the images.

## Generating an outpainted dataset

Once you have a directory of images in the format described above, we can proceed with the following steps. We assume that you have at least one GPU as the steps will take a very long time on a CPU.

1. Run the image captioner to generate captions for each image. To run [BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2):

    ```python3 run_blip.py <img_dir> -o <blip_output_dir>```

    For every image, this will write a file `blip<IMAGEID>.txt`, where `<IMAGEID>` is replaced by the image id, in `<blip_output_dir>` that contains the estimated caption.

    Look at the `--part` keyword argument to run this in parallel.

2. Run the diffusion-inpainting model to generate outpainted images. To run [StableDiffusionV2](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting):

    ```python3 run_outpaint.py <img_dir> -o <outpaint_output_dir>```

    Look at the `--part` keyword argument to run this in parallel.

    Given an image, say img<IMAGEID>.jpg, in `<img_dir>`, this will create several outpainted variations of the image in `<outpaint_output_dir>`.

    These images will have the name format `img<IMAGEID>_X_Y_W_H.jpg` where X, Y, W, H are the original image's bounding box coordinates.

3. To detect bad outpainting results with a pretrained classifier, run: ```python3 run_quality.py <img_dir> -o <bad_img_file>```

    This will write a JSON file `<bad_img_file>` that contains a list of image files from `<img_dir>` that are likely to be bad outpainting results.

    Note that if the images in `<img_dir>` change, you will need to re-run this step.

4. Compute instance segmentations for the outpainted images. To run [YOLOv8](https://github.com/ultralytics/ultralytics):

    ```python3 run_yolox.py <outpaint_output_dir> -o <yolo_output_dir>```

    If you are outpainting non-human images, you will need to pass the `--cls` argument to this script.

    This will create two files in `<yolo_output_dir>`, ```detect.json.gz``` and ```mask.npz```, which contain the bounding boxes and instance segmentations for all of the outpainted images in `<outpaint_output_dir>`.

    You can use ```view_data.py``` understand the format of these binary files.

### Recommended directory structure

We recommend the following directory structures.

For the stock images:
```
unsplash/
    images/
        img<IMAGEID>.jpg
        ...
    blip/                           # step 1
        blip<IMAGEID>.txt
        ...
```

For a cropping dataset:
```
data/portrait/
    images.json
    train.json
    val.json
    test.json
    outpaint/
        images/                     # step 2
            img<IMAGEID>_X_Y_W_H.jpg
            ...
        bad.json                    # step 3
        detect.json.gz              # step 4
        mask.npz                    # step 4
```

## Cropping model

Once you have generated an outpainted dataset, we are ready to train a cropping model.

### Setup

You will need to download and build the RoDAlign library. See [rod_align/README.md](rod_align/README.md) for instructions. The code is from [GAIC](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch).

### Training

We provide scripts for training: ```train_gencrop.py```. This trains the generic cropping model. Use ```-h``` flag to see the arguments. You want to pass the directory of the outpainted images as the ```dataset``` argument: for example, ```data/portrait/outpaint```.

### Inference

#### Getting the datasets from HCIC

Follow the instructions in [HCIC](https://github.com/bcmi/Human-Centric-Image-Cropping/tree/main/human_bboxes). For each dataset, you should have a directory with the following structure:
```
images/
    ...
detect.json.gz      # generated by run_yolox.py
mask.npz            # ^ same ^
human_bboxes.json   # copy from HCIC /human_bboxes directory
image_crop.json     # ^ same ^
... (any other files from HCIC)
```

Use the ```run_yolox.py``` script to generate the ```detect.json.gz``` and ```mask.npz``` files. Also, copy all of the files in /human_bboxes from HCIC to the appropriate directory.

Update ```config.py``` to point to the correct directories.

#### Getting SACD

Download it from the [SACD](https://cg.cs.tsinghua.edu.cn/SACD/) page.
Since each image may have multiple examples (depending on subject), it easiest to make a directory of symbolic links mapping examples to images (in the same directory as the unzipped dataset; for example, ```images-sym```). See ```sym_link_sacd.py```.

Update ```config.py``` to point to the correct directory.

### Running inference

We provide scripts (e.g., ```test_gencrop.py```) to run inference on the trained models. Use the ```-h``` flag to see the arguments.

These scripts assume that you have updated ```config.py``` to point to the correct directories.

For bechmarks, use the ```--benchmark``` flag. For the Unsplash datasets, point the ```--dataset_dir``` flag to a dataset in ```/data``` (e.g., ```/data/portrait```). If no option is passed, the script will run inference on the test split of the generated images.

Note that earlier epochs tend to perform better on the existing datasets, while later epochs tend to perform better on the Unsplash datasets. We believe that this is because of the difference in the crop distributions between these datasets.

### Pretrained models

Coming soon!
