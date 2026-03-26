from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class ImageNetClassifier:
    def __init__(self) -> None:
        self.weights = ResNet50_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]
        self.transforms = self.weights.transforms()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=self.weights)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[Prediction]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            scores, indices = torch.topk(probabilities, k=top_k)

        return [
            Prediction(
                label=self.categories[index],
                confidence=float(score.item()),
            )
            for score, index in zip(scores, indices.tolist(), strict=False)
        ]
