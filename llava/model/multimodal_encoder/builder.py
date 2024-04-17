import os
from .clip_encoder import CLIPVisionTower
from .convnext_encoder import ConvNeXtCLIPVisionTower
from .siglip_encoder import SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print(f"now we are building vision tower, the model is {vision_tower}")
    if 'siglip' in vision_tower:
        print(f'building SiglipVisionTower')
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if vision_tower.startswith("openai") or 'clip-vit' in vision_tower:
        print(f'building CLIPVisionTower')
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if 'convnext' in vision_tower:
        print(f'building ConvNeXtCLIPVisionTower')
        return ConvNeXtCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    return ConvNeXtCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

