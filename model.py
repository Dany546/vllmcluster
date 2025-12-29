"""
Defines the model classes for DINO and CLIP models.
DINO: gives attention maps and attention weights (by heads) for the last transformer block, computes weighted averages of the tokens using both weights and maps
Projection: projects visual embeddings from DINO's into text embedding space using similarity with CLIP's text encoder
CLIP: provides text embeddings from the captions, projects visual embeddings from DINO's into text embedding space
"""

import os
from typing import Literal as Choice

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import device, model_cache_path
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class ProjectionLayer(nn.Module):
    def __init__(self, in_dim=768, proj_dim=128, out_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.Tanh(),
            nn.Linear(proj_dim, out_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale.requires_grad = False

    def contrastive_loss(self, scores):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()  # logit_scale *
        logits_per_image = scores * logit_scale
        logits_per_text = logits_per_image.t()

        # compute bidirectional CE loss
        num_logits = logits_per_image.shape[0]
        labels = torch.arange(
            num_logits, device=logits_per_image.device, dtype=torch.long
        )
        loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return loss  # / scores.shape[0] ** 2  # normalization by the batch s

    def forward(self, visual_embedding, textual_embedding):
        projected = self.projection(visual_embedding)  # still NxHxD
        if self.training:
            sims = torch.einsum(
                "ik,ijk->ij", textual_embedding, projected
            )  # similarities matrices for each head
            sims = sims.softmax(dim=-1)  # normalizes
            # visual_embedding == weighted averaged by the text similarities
            projected = (projected * sims.unsqueeze(dim=-1)).sum(dim=1)
            sims = textual_embedding @ projected.transpose(1, 0)

            return projected, self.contrastive_loss(sims)
        else:
            return projected


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.encoder = CLIPModel.from_pretrained(f"openai/clip-vit-base-patch16")

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        pixel_values = inputs["pixel_values"].to(images.device)
        projected = self.encoder.get_image_features(pixel_values=pixel_values)
        return projected


class DINO(nn.Module):
    def __init__(self, attention_pooling=False):
        super().__init__()
        # Load DINO ViT model
        if os.path.exists(os.path.join(model_cache_path, "dinov2.pth")):
            self.dino = torch.load(
                os.path.join(model_cache_path, "dinov2.pth"), weights_only=False
            )["model"]
        else:
            self.dino = timm.create_model(f"vit_base_patch14_dinov2", pretrained=True)
            torch.save(
                {"model": self.dino.cpu()}, os.path.join(model_cache_path, "dinov2.pth")
            )

        self.attention_pooling = attention_pooling 
        # self.projection_layer = ProjectionLayer()

    def attention_pooling_fn(self, images):
        """
        Pool patch tokens using CLS attention weights from last layer
        """ 
        # Forward pass through patch embedding and blocks
        x = self.dino.patch_embed(images)
        x = self.dino._pos_embed(x)
        x = self.dino.norm_pre(x)
        
        # Pass through all blocks except we need attention from last one
        for i, block in enumerate(self.dino.blocks):
            if i < len(self.dino.blocks) - 1:
                x = block(x)
            else:
                # Last block - extract attention weights
                B, N, C = x.shape
                qkv = block.attn.qkv(block.norm1(x))
                qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                
                # Get CLS token attention to patches (average across heads)
                cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # (B, num_patches)
                
                # Complete the block
                x = block(x)
        
        x = self.dino.norm(x)
        
        # Attention pooling: weight patch tokens by CLS attention
        patch_tokens = x[:, 1:]  # (B, num_patches, dim)
        pooled = (patch_tokens * cls_attn.unsqueeze(-1)).sum(dim=1)  # (B, dim)
            
        return pooled
    
    def forward(self, image, labels=None, text_embeddings=None):
        # Forward pass through DINO model
        if self.attention_pooling:
            return self.attention_pooling_fn(image) 
    
        dino_features = self.dino.forward_features(image)
        return dino_features[:, 0]  # , [[0, 0, 0]]

        dino_features = (dino_features.unsqueeze(1) * self.attention_weights).mean(
            dim=2
        )  # NxH visual representations

        # Project DINO features into CLIP text embedding space
        return self.projection_layer(dino_features, text_embeddings.squeeze(1))


def get_dino_attention_map(model, image_tensor, head=0):
    with torch.no_grad():
        outputs = model.forward_features(image_tensor)
        attn = model.blocks[-1].attn.get_attention_map()
        cls_attn = attn[:, head, 0, 1:]  # CLS token attention to patches
        return cls_attn.reshape(1, 14, 14)  # Assuming 14x14 patch grid
