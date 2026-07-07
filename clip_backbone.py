import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPVisionWrapper(nn.Module):
    """
    Wraps the vision tower of a HuggingFace CLIPModel so it is a drop-in
    replacement for the DINO ViT backbones used elsewhere in this repo:
    forward(x) -> [B, 768] pooled features (pre-projection), the same
    contract vit_threeHeads_v2 / vit_twoHeads_v2 expect from `self.backbone(x)`.

    The text tower is deliberately NOT stored on this module. It lives on
    the plain CLIPModel returned alongside this wrapper by build_clip_backbone
    (see args.clip_model in train_seal.py), so it never enters this module's
    parameter tree and can never be swept up by get_params_groups()/the
    optimizer, and student.train()/.eval() calls on the SEAL model never
    touch it.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.embed_dim = clip_model.config.vision_config.hidden_size
        self.proj_dim = clip_model.config.projection_dim

    def forward(self, x, return_proj=False):
        feat = self.vision_model(pixel_values=x).pooler_output
        if return_proj:
            proj = self.visual_projection(feat)
            return feat, proj
        return feat


def build_clip_backbone(clip_model_name):
    """
    Loads a HuggingFace CLIP checkpoint once and returns:
      - backbone: CLIPVisionWrapper, usable anywhere the DINO backbone was
      - clip_model: the full CLIPModel (vision + text), kept alive so the
        text encoder can be used for prototype building elsewhere
    """
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    backbone = CLIPVisionWrapper(clip_model)
    return backbone, clip_model
