"""CLIP image/text embedder backed by ``open_clip`` (install with ``pip install .[clip]``).

Images are fetched from their ``image_uri`` through the storage layer, so they can live on S3
or local disk. Frames whose image cannot be loaded get a zero vector and are reported via
``last_failures`` — downstream similarity treats them as maximally dissimilar to everything.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pyarrow as pa

from datamine.embeddings.base import Embedder, l2_normalize
from datamine.storage import resolve

log = logging.getLogger(__name__)


class ClipEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        batch_size: int = 64,
        device: str | None = None,
        crop_to_boxes: bool = False,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("ClipEmbedder requires the [clip] extra: pip install 'datamine[clip]'") from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = batch_size
        self.crop_to_boxes = crop_to_boxes
        self.name = f"clip-{model_name}-{pretrained}"
        with torch.no_grad():
            probe = self.model.encode_text(self.tokenizer(["probe"]).to(self.device))
        self.dim = int(probe.shape[-1])
        self.last_failures: list[str] = []

    def _load_image(self, uri: str):
        from PIL import Image

        loc = resolve(uri)
        with loc.filesystem.open_input_file(loc.path) as fh:
            return Image.open(io.BytesIO(fh.read())).convert("RGB")

    def _union_crop(self, image, labels):
        if not labels:
            return image
        x1 = min(l["x1"] for l in labels)
        y1 = min(l["y1"] for l in labels)
        x2 = max(l["x2"] for l in labels)
        y2 = max(l["y2"] for l in labels)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return image
        return image.crop((x1, y1, x2, y2))

    def embed_samples(self, samples: pa.Table) -> np.ndarray:
        torch = self._torch
        rows = samples.select(["sample_id", "image_uri", "labels"]).to_pylist()
        out = np.zeros((len(rows), self.dim), dtype=np.float32)
        self.last_failures = []

        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            tensors, positions = [], []
            for offset, row in enumerate(batch):
                try:
                    img = self._load_image(row["image_uri"])
                    if self.crop_to_boxes:
                        img = self._union_crop(img, row.get("labels") or [])
                    tensors.append(self.preprocess(img))
                    positions.append(start + offset)
                except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the batch
                    log.warning("failed to load %s: %s", row.get("image_uri"), exc)
                    self.last_failures.append(row["sample_id"])
            if not tensors:
                continue
            with torch.no_grad():
                feats = self.model.encode_image(torch.stack(tensors).to(self.device)).float().cpu().numpy()
            out[positions] = l2_normalize(feats)
        return out

    def embed_text(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(tokens).float().cpu().numpy()
        return l2_normalize(feats)
