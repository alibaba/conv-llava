import torch
import torch.nn as nn
import argparse
from transformers import CLIPImageProcessor
from transformers import PretrainedConfig
import os
from .unireplknet.unireplknet_encoder import unireplknet_l_plus


class LKNetConfig(PretrainedConfig):
    model_type = "lknet"

    def __init__(
        self,
        in_chans=3,
        image_size=256,
        num_classes=1000,
        depths=(3, 3, 27, 3, 6),
        dims=(192, 384, 768, 1536, 3072),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        kernel_sizes=None,
        deploy=False,
        with_cp=True,
        init_cfg=None,
        attempt_use_lk_impl=True,
        use_sync_bn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_chans = in_chans
        self.image_size = image_size
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.head_init_scale = head_init_scale
        self.kernel_sizes = kernel_sizes
        self.deploy = deploy
        self.with_cp = with_cp
        self.init_cfg = init_cfg
        self.attempt_use_lk_impl = attempt_use_lk_impl
        self.use_sync_bn = use_sync_bn


unireplknet_l_plus_config = {
    "depths": (3, 3, 27, 3, 6),
    "kernel_sizes": (
        (3, 3, 3),
        (13, 13, 13),
        (
            13,
            3,
            3,
            13,
            3,
            3,
            13,
            3,
            3,
            13,
            3,
            3,
            13,
            3,
            3,
            13,
            3,
            3,
            13,
            3,
            3,
            13,
            3,
            3,
            13,
            3,
            3,
        ),
        (13, 13, 13),
        (3, 3, 3, 3, 3, 3),
    ),
}


class LKNetCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.update_resolution = getattr(args, "mm_vision_resolution", 256)
        self.cfg_only = LKNetConfig.from_dict(unireplknet_l_plus_config)

        if not delay_load:
            self.load_model()

    def load_model(self):
        print(f"entering load model, load {self.vision_tower_name}")
        self.image_processor = CLIPImageProcessor.from_pretrained(
            os.path.dirname(self.vision_tower_name)
        )
        self.vision_tower = unireplknet_l_plus()
        self.vision_tower.config = LKNetConfig.from_dict(unireplknet_l_plus_config)
        ckpt = torch.load(self.vision_tower_name)
        del ckpt["norm.weight"]
        del ckpt["norm.bias"]
        missing_keys, unexpected_keys = self.vision_tower.load_state_dict(
            ckpt, strict=False
        )
        print("Loaded CLIP Pretrained Models")
        print(
            f"missing keys are {missing_keys}\n unexpected keys are {unexpected_keys}"
        )

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        if self.update_resolution > 256:
            self.set_crop_size(self.update_resolution)
            print(
                f"Crop size changed to {self.update_resolution}x{self.update_resolution}"
            )
        self.make_layers_trainable_after_stage(4)

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                x = image
                for stage_idx in range(5):
                    x = self.vision_tower.downsample_layers[stage_idx](x)
                    x = self.vision_tower.stages[stage_idx](x)
                image_features = x.permute(0, 2, 3, 1)
                image_features = image_features.reshape(x.shape[0], -1, x.shape[1]).to(image.dtype)
                image_features.append(x)
        else:
            x = images
            for stage_idx in range(5):
                x = self.vision_tower.downsample_layers[stage_idx](x)
                x = self.vision_tower.stages[stage_idx](x)
            image_features = x.permute(0, 2, 3, 1)
            image_features = image_features.reshape(x.shape[0], -1, x.shape[1]).to(images.dtype)
        return image_features

    def make_layers_trainable_after_stage(self, stage_index):
        for i, stage in enumerate(self.vision_tower.stages):
            if i >= stage_index:
                for param in stage.parameters():
                    param.requires_grad = True
        for i, stage in enumerate(self.vision_tower.downsample_layers):
            if i >= stage_index:
                for param in stage.parameters():
                    param.requires_grad = True
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        print("Trainable status of each stage:")
        for i, stage in enumerate(self.vision_tower.stages):
            trainable = all(param.requires_grad for param in stage.parameters())
            print(f"Stage {i}: {'Trainable' if trainable else 'Not Trainable'}")

        print("\nTrainable status of each downsampling layer:")
        for i, downsample_layer in enumerate(self.vision_tower.downsample_layers):
            trainable = all(param.requires_grad for param in downsample_layer.parameters())
            print(f"Downsampling Layer {i}: {'Trainable' if trainable else 'Not Trainable'}")

    def set_crop_size(self, new_size):
        size_dict = {"height": new_size, "width": new_size}
        self.image_processor.crop_size = size_dict
        self.image_processor.size = {"shortest_edge": new_size}
        self.vision_tower.config.image_size = new_size
        self.config.image_size = new_size

    def save_config(self, path):
        self.config.save_pretrained(path)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.downsample_layers[0][0].weight.dtype

    @property
    def device(self):
        return self.vision_tower.downsample_layers[0][0].weight.device

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.dims[-1]

    @property
    def num_patches_per_side(self):
        return self.config.image_size // 2 ** (len(self.config.depths) + 1)

    @property
    def num_patches(self):
        return (self.config.image_size // 2 ** (len(self.config.depths) + 1)) ** 2

    @property
    def crop_size(self):
        return self.image_processor.crop_size

