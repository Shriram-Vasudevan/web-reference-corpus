"""CLIP embedding extraction for website screenshots."""

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

import config


class CLIPEmbedder:
    """Extract CLIP embeddings from images and text."""

    def __init__(self, device: str | None = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)

    @torch.no_grad()
    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Embed a single image, returning an L2-normalized 512-dim vector."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def embed_images_batch(self, image_paths: list[str | Path],
                           batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
        """Embed a batch of images, returning (N, 512) L2-normalized matrix."""
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self.preprocess(img))

            batch = torch.stack(tensors).to(self.device)
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        result = np.concatenate(all_features, axis=0).astype(np.float32)
        return result

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query, returning an L2-normalized 512-dim vector."""
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)
