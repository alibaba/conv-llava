## Data

### Projector Initialzation

We use captions from ShareGPT4V-PT, ShareGPT4V, ALLAVA.

### Vision Language Pretraining

We use ShareGPT4V-PT, ShareGPT4V, ALLAVA and a part of VFLAN.

### Instrcution Tuning

We use LLaVA-1.5 sft 665k dataset. We would release the results when LLaVA-NExT released.

### Prepare Images

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

### Data Configuration

You could modify the file [data.py](conv-llava/llava/data/data_blending.py) to add the datasets. Replace with the true path:

```python
def build_sharegpt4v(tokenizer, data_args):
    data_path = 'path_to_sharegpt4v_pt.json'
    image_folder = 'folder_to_sharegpt4v_pt'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset
```
