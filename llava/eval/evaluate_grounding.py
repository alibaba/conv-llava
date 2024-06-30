import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from PIL import Image
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def build_transform(is_train, input_size, pad2square=False):
    if is_train:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.),
                                interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        if pad2square is False:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    return transform


ds_collections = {
    'refcoco_val': 'data/refcoco/refcoco_val.jsonl',
    'refcoco_testA': 'data/refcoco/refcoco_testA.jsonl',
    'refcoco_testB': 'data/refcoco/refcoco_testB.jsonl',
    'refcoco+_val': 'data/refcoco/refcoco+_val.jsonl',
    'refcoco+_testA': 'data/refcoco/refcoco+_testA.jsonl',
    'refcoco+_testB': 'data/refcoco/refcoco+_testB.jsonl',
    'refcocog_val': 'data/refcoco/refcocog_val.jsonl',
    'refcocog_test': 'data/refcoco/refcocog_test.jsonl',
}


def reserve_square_bbox(box, w, h):
    if w == h:
        return box
    box = box.tolist()[0]
    if w > h:
        x1, y1, x2, y2 = box
        y1 -= (w - h) // 2
        y2 -= (w - h) // 2
        box = [[x1, y1, x2, y2]]
        return torch.tensor(box).resize(1, 4)
    else:
        x1, y1, x2, y2 = box
        x1 -= (h - w) // 2
        x2 -= (h - w) // 2
        box = [[x1, y1, x2, y2]]
        return torch.tensor(box).resize(1, 4)


class ModelConfig:
    def __init__(self, image_aspect_ratio=None):
        self.image_aspect_ratio = image_aspect_ratio

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def collate_fn(batch, tokenzier=None):
    input_ids, image_tensors, bbox, hw, image_path, text = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, bbox, hw, image_path, text


class RefCOCODataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size=224, pad2square=False, image_processor=None, model_cfg=None, tokenizer=None):
        self.datas = open(test).readlines()
        self.prompt = prompt
        self.transform = build_transform(is_train=False, input_size=input_size, pad2square=pad2square)
        self.image_processor = image_processor
        self.model_config = model_cfg
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        image_path = data['image']
        text = data['sent']
        bbox = data['bbox']

        w, h = data['width'], data['height']
        image = os.path.join('/mnt/thuair/gcjtcl/InternVL', image_path)

        image = Image.open(image).convert('RGB')
        # pixel_values = self.transform(image).unsqueeze(0)
        pixel_values = process_images([image], self.image_processor, self.model_config)[0]
        prompt = self.prompt.format(text)
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, pixel_values, bbox, (h, w), image_path, text


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    print('prompt:', prompt)
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        dataset = RefCOCODataset(
            test=os.path.join("/mnt/thuair/gcjtcl/InternVL", ds_collections[ds_name]),
            prompt=prompt,
            input_size=image_size,
            pad2square=pad2square,
            image_processor=image_processor,
            model_cfg=model_cfg,
            tokenizer=tokenizer
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn),
        )

        outputs = []
        for _, (questions, pixel_values, bboxes, hws, image_path, text) in enumerate(tqdm(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            output_ids = model.generate(
                questions.to(device='cuda', non_blocking=True),
                images=pixel_values.to(dtype=torch.float16),
                do_sample=False,
                temperature=0,
                max_new_tokens=100)

            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            answers = [pred]

            for bbox, hw, answer in zip(bboxes, hws, answers):
                outputs.append({
                    'image_path': image_path,
                    'text': text,
                    'answer': answer,
                    'gt_bbox': bbox,
                    'hw': hw,
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            # with open("/mnt/thuair/gcjtcl/InternVL/internvl_chat/conv768/refcocog_val_240419174233.json", 'r') as f:
            #     merged_outputs = json.load(f)

            correct = total_cnt = 0
            for i, output in enumerate(merged_outputs):
                predict_bbox = re.findall(PATTERN, output['answer'])
                try:
                    predict_bbox = (float(predict_bbox[0][0]), float(predict_bbox[0][1]), float(predict_bbox[0][2]),
                                    float(predict_bbox[0][3]))
                except:
                    predict_bbox = (0., 0., 0., 0.)
                target_bbox = torch.tensor(output['gt_bbox'],
                                           dtype=torch.float32).view(-1, 4)
                predict_bbox = torch.tensor(predict_bbox,
                                            dtype=torch.float32).view(-1, 4)
                if predict_bbox.sum() >= 4:
                    predict_bbox = predict_bbox / 1000
                predict_bbox *= max(output['hw'])
                w, h = output['hw'][1], output['hw'][0]
                predict_bbox = reserve_square_bbox(predict_bbox, w, h)
                # print(predict_bbox)
                # predict_bbox[:, 0::2] *= output['hw'][1]
                # predict_bbox[:, 1::2] *= output['hw'][0]
                iou, _ = box_iou(predict_bbox, target_bbox)
                iou = iou.item()
                total_cnt += 1
                if iou >= 0.5:
                    correct += 1

            print(f'Evaluating {ds_name} ...')
            print(f'Precision @ 1: {correct / total_cnt} \n')
            summaries.append([args.checkpoint, ds_name, f'Precision @ 1: {correct / total_cnt} \n'])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='refcoco_val,refcoco_testA,refcoco_testB,'
                                                        'refcoco+_val,refcoco+_testA,refcoco+_testB,'
                                                        'refcocog_val,refcocog_test')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()


    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    rank = torch.distributed.get_rank()
    if rank == 0:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    torch.cuda.set_device(int(os.getenv('RANK', 0)))
    print(f"rank: {int(os.getenv('RANK', 0))}")
    device = torch.device(int(os.getenv('RANK', 0)))

    # tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')

    # 创建一个model_cfg对象
    model_cfg = ModelConfig(image_aspect_ratio="pad")


    # device = 
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.checkpoint, None, "llava",  device='cpu', device_map='cpu')
    model = model.to(device).eval()
    vision_tower = model.get_vision_tower().to(device)
    model.get_model().mm_projector.to(device)
    # model.cuda()
    # for p in model.parameters():
    #     p.cuda()
    image_size = 336
    pad2square = True
    prompt = 'Please provide the bounding box coordinate of the region this sentence describes: {}.'

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 30:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] pad2square: {pad2square}')
    print(f'[test] template: v1')

    evaluate_chat_model()
