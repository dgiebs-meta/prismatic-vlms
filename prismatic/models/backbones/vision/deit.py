"""
clip_vit.py
"""
from prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported CLIP Vision Backbones (from TIMM)
DEIT_VISION_BACKBONES = {
    "deit3-huge": "deit3_huge_patch14_224.fb_in22k_ft_in1k",
}


# [IMPORTANT] By Default, TIMM initialized OpenAI CLIP models with the standard GELU activation from PyTorch.
#             HOWEVER =>> Original OpenAI models were trained with the quick_gelu *approximation* -- while it's
#                         a decent approximation, the resulting features are *worse*; this was a super tricky bug
#                         to identify, but luckily there's an easy fix (`override_act_layer`)
class DeitBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            DEIT_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
