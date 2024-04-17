# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import subprocess
from packaging import version
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import numpy as np

from torch.utils.data import ConcatDataset
from tqdm import tqdm
import torch

import transformers
from transformers.utils import logging
import tokenizers
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
from datasets import interleave_datasets
from transformers import CLIPImageProcessor

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(
    tokenizers.__version__) >= version.parse('0.14')


def decode_tokens(tokenizer, token_sequences, tokens_to_skip):
    # Decode each sublist in the lists, include original IDs for tokens that cannot be decoded
    decoded_sequences = [
        [
            tokenizer.decode([token], skip_special_tokens=True) if token not in tokens_to_skip
            else f"{token}"
            for token in ids
        ]
        for ids in token_sequences.tolist()
    ]
    return decoded_sequences


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + \
                    '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    debug = False
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    tokens_to_skip = [-100, -200, 0, 1, 2, 32000, 32001]

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if debug:
                print(f'decode rounds: {rou}')

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # legacy = getattr(tokenizer, "legacy", False)
            if i != 0 and IS_TOKENIZER_GREATER_THAN_0_14:
                # print("reduce")
                # if i != 0 and not legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        if debug:
            print(
                f'decode inputs: {decode_tokens(tokenizer, input_ids, tokens_to_skip)}')
            print(
                f'decode targets: {decode_tokens(tokenizer, targets, tokens_to_skip)}')

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + \
            conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(
            source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print(conversation_lib.default_conversation.version)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len(
                [header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = self.data_args.image_folder

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(
                self.image_folder, image_file)).convert('RGB')

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255)
                                      for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            do_center_crop = getattr(
                self.data_args.image_processor, 'do_center_crop', False)
            if do_center_crop:
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            data_dict['image'] = torch.zeros(
                3, crop_size['height'], crop_size['width'])
        return data_dict


class SampleDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 image_folder: str,
                 sample_size: int = 0,
                 shuffle: bool = False):
        super(SampleDataset, self).__init__(data_path, tokenizer, data_args)
        if shuffle:
            np.random.shuffle(self.list_data_dict)
        if sample_size != 0:
            if sample_size < len(self.list_data_dict):
                self.list_data_dict = self.list_data_dict[:sample_size]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder


class CombinedLazySupervisedDataset(Dataset):
    def __init__(self, datasets: List[LazySupervisedDataset]):
        super(CombinedLazySupervisedDataset, self).__init__()
        self.datasets = datasets
        self.total_length = sum(len(dataset) for dataset in datasets)

        # Combine list_data_dict and image_folder
        self.combined_list_data_dict = []
        self.combined_image_folders = []
        current_length = 0
        for dataset in datasets:
            dataset_length = len(dataset)
            self.combined_list_data_dict.extend(dataset.list_data_dict)
            self.combined_image_folders.extend(
                [dataset.image_folder] * dataset_length)
            current_length += dataset_length

    @property
    def lengths(self):
        length_list = []
        for sample in self.combined_list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(
                sum(len(conv['value'].split())
                    for conv in sample['conversations']) + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.combined_list_data_dict:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return self.total_length

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Find the dataset index and the index within that dataset
        dataset_index = 0
        while i >= len(self.datasets[dataset_index]):
            i -= len(self.datasets[dataset_index])
            dataset_index += 1

        # Get item from the corresponding dataset
        data_dict = self.datasets[dataset_index].__getitem__(i)

        return data_dict


def build_sharegpt4v(tokenizer, data_args):
    data_path = 'path_to_sharegpt4v_pt.json'
    image_folder = 'folder_to_sharegpt4v_pt'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_sharegpt4v100k(tokenizer, data_args):
    data_path = 'path_to_sharegpt4v.json'
    image_folder = 'folder_to_sharegpt4v'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_lcs(tokenizer, data_args):
    data_path = 'path_to_lcs'
    image_folder = 'folder_to_lcs'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_llava665k(tokenizer, data_args):
    data_path = 'path_to_llava665k_dataset'
    image_folder = 'folder_to_llava665k_images'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_vflan(tokenizer, data_args):
    data_path = 'path_to_vflan_dataset'
    image_folder = 'folder_to_vflan_images'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_allava_laion_caption(tokenizer, data_args):
    data_path = 'path_to_allava_laion_caption_dataset'
    image_folder = 'folder_to_allava_laion_caption_images'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_allava_vflan_caption(tokenizer, data_args):
    data_path = 'path_to_allava_vflan_caption_dataset'
    image_folder = 'folder_to_allava_vflan_caption_images'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_allava_vflan_instruct(tokenizer, data_args):
    data_path = 'path_to_allava_vflan_instruct_dataset'
    image_folder = 'folder_to_allava_vflan_instruct_images'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset


def build_allava_laion_instruct(tokenizer, data_args):
    data_path = 'path_to_allava_laion_instruct_dataset'
    image_folder = 'folder_to_allava_laion_instruct_images'
    dataset = SampleDataset(data_path, tokenizer, data_args,
                            image_folder)
    return dataset



def build_interleaved_dataset(tokenizer: transformers.PreTrainedTokenizer,
                              data_args,
                              dataset_names: str):

    data_list = []
    print(dataset_names)
    for dataset_name in dataset_names:
        if dataset_name == 'lcs':
            dataset = build_lcs(tokenizer, data_args)
        elif dataset_name == 'sharegpt4vpretrain':
            dataset = build_sharegpt4v(
                tokenizer, data_args)
        elif dataset_name == 'sharegpt4v100k':
            dataset = build_sharegpt4v100k(
                tokenizer, data_args)
        elif dataset_name == 'llava665k':
            dataset = build_llava665k(
                tokenizer, data_args)
        elif dataset_name == 'vflan':
            dataset = build_vflan(
                tokenizer, data_args)
        elif dataset_name == 'allava_laion_caption':
            dataset = build_allava_laion_caption(
                tokenizer, data_args)
        elif dataset_name == 'allava_vflan_caption':
            dataset = build_allava_vflan_caption(
                tokenizer, data_args)
        elif dataset_name == 'allava_laion_instruct':
            dataset = build_allava_laion_instruct(
                tokenizer, data_args)
        elif dataset_name == 'allava_vflan_instruct':
            dataset = build_allava_vflan_instruct(
                tokenizer, data_args)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        data_list.append(dataset)

    if len(data_list) > 1:
        datasets = CombinedLazySupervisedDataset(data_list)
    else:
        datasets = data_list[0]

    return datasets


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer = None,

        data_args=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = build_interleaved_dataset(
        tokenizer=tokenizer,
        data_args=data_args,
        dataset_names=data_args.dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
