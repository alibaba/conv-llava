# Data

We use the following hyperparameters for training ConvLLaVA.

| Hyperparameters | Stage 1 | Stage 2 | Stage 3 |
| --------------- | ------- | ------- | ------- |
| Learning Rate   | 3e-4    | 2e-5    | 2e-5    |
| Batch Size      | 256     | 256     | 128     |
| Epochs          | 1       | 1       | 1       |
| Warmup Ratio    | 0.03    | 0.03    | 0.03    |
| Weight Decay    | 0       | 0       | 0       |
| Optimizer       | AdamW   | AdamW   | AdamW   |

## Projector Initialzation

We use captions from ShareGPT4V-PT, ShareGPT4V, ALLAVA.

## Vision Language Pretraining

We use ShareGPT4V-PT, ShareGPT4V, ALLAVA and a part of VFLAN.

## Instrcution Tuning

We use LLaVA-1.5 sft 665k dataset. We would update the results when LLaVA-NExT released.

## Prepare Images

First, download all images and instrcution files.

- ALLaVA: [images](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- LLaVA: [llava](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- WebData: [images](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Only for academic usage.
- SAM: [images](https://ai.meta.com/datasets/segment-anything-downloads/). We only use 000000~000050.tar for now. If you find it is slow for you to donnload in China, please refer to [opendatalab](https://opendatalab.com/OpenDataLab/SA-1B) to download it.
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- TextVQA: [trainvalimages](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- vflan: [vflan](https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k)

Then, organize the data as follows:

```none
ShareGPT4V
├── ...
├── data
│   ├── allava
│   │   ├── allava_laion
│   │   │   ├── images
│   │   │   ├── ALLaVA-Caption-LAION-4V.json
│   │   │   ├── ALLaVA-Instruct-LAION-4V.json
│   │   ├── allava_vflan
│   │   │   ├── ALLaVA-Caption-VFLAN-4V.json
│   │   │   ├── ALLaVA-Instruct-VFLAN-4V.json
│   ├── coco
│   │   ├── train2017
│   ├── llava
│   │   ├── llava_v1_5_mix665k.json
│   ├── sam
│   │   ├── images
│   ├── gqa
│   │   ├── images
│   ├── ocr_vqa
│   │   ├── images
│   ├── textvqa
│   │   ├── train_images
│   ├── vg
│   │   ├── VG_100K
│   │   ├── VG_100K_2
│   ├── vflan
│   │   ├── images_191task_1k
│   │   ├── annotation_191-task_1k.json
│   ├── sharegpt4v
│   │   ├── share-captioner_coco_lcs_sam_1246k_1107.json
│   │   ├── sharegpt4v_instruct_gpt4-vision_cap100k.json
│   ├── share_textvqa
│   │   ├── images
│   ├── web-celebrity
│   │   ├── images
│   ├── web-landmark
│   │   ├── images
│   ├── wikiart
│   │   ├── images
├── ...
```

If you find download ocrvqa images slow. You could refer to this [issue](https://github.com/haotian-liu/LLaVA/issues/931).
Use multiprocessing to speed up:

```python
import concurrent.futures
def download_image(k):
    ext = os.path.splitext(data[k]['imageURL'])[1]
    outputFile = 'images/%s%s' % (k, ext)

    # Only download the image if it doesn't exist
    if not os.path.exists(outputFile):
        ureq.urlretrieve(data[k]['imageURL'], outputFile)


if download == 1:
    # Create the directory if it doesn't exist
    if not os.path.exists('./images'):
        os.mkdir('./images')

    # Create a thread pool and download the images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_image, data.keys())
```

For ocrvqa, some git images should be transfered to jpg. You could follow bwloe code:

```python
import os
from PIL import Image

def convert_gif_to_jpg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.gif'):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                jpg_filename = os.path.splitext(filename)[0] + '.jpg'  
                jpg_path = os.path.join(folder_path, jpg_filename)
                img.convert('RGB').save(jpg_path, 'JPEG', quality=95)
                print(f'Converted {filename} to {jpg_filename}')

folder_path = 'path_to_your_folder'
convert_gif_to_jpg(folder_path)
```

## Data Configuration

You could modify the file [data.py](conv-llava/llava/data/data_blending.py) to add the datasets. Replace with the true path:

```python
def build_sharegpt4v(tokenizer, data_args):
    data_path = 'path_to_sharegpt4v_pt.json'
    image_folder = 'folder_to_sharegpt4v_pt'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset
```
