from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class DeepReIDStatus:
    enabled: bool
    backend: str
    message: str


class DeepPersonReID:
    """
    Anonymous full-body person ReID embedding extractor.

    Preferred backend:
      - Torchreid OSNet, using torchreid.reid.utils.FeatureExtractor or torchreid.utils.FeatureExtractor, depending on the installed torchreid package layout.

    Optional fallback backend:
      - Torchvision MobileNetV3 deep visual embedding.

    This version defaults to strict OSNet mode from configs/tracking_memory.yaml:
    backend=torchreid, allow_torchvision_fallback=false, require_backend=true.
    That means the app will not silently pretend that a generic fallback is OSNet.

    This class does not do face recognition and does not assign real identities.
    It only produces anonymous full-body embeddings for track continuity.
    """

    def __init__(
        self,
        enabled: bool = True,
        backend: str = "torchreid",
        model_name: str = "osnet_x0_25",
        model_path: str = "",
        device: Optional[str] = None,
        image_height: int = 256,
        image_width: int = 128,
        allow_torchvision_fallback: bool = False,
        require_backend: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.backend_request = str(backend or "torchreid").lower()
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or "auto"
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.allow_torchvision_fallback = bool(allow_torchvision_fallback)
        self.require_backend = bool(require_backend)

        self.backend = "none"
        self.status = DeepReIDStatus(False, "none", "Deep ReID disabled")
        self._extractor = None
        self._torch = None
        self._torchvision_model = None
        self._torchvision_weights = None

        if not self.enabled:
            return

        requested = self.backend_request
        wants_torchreid = requested in {"auto", "torchreid", "osnet"}
        wants_torchvision = requested in {"auto", "torchvision", "mobilenet"}

        if wants_torchreid and self._try_init_torchreid():
            return

        if requested in {"torchreid", "osnet"}:
            previous_message = self.status.message if self.status and self.status.message else ""
            message = previous_message or (
                "Torchreid/OSNet backend requested but unavailable. "
                "Install requirements_reid_optional.txt or set person_reid.allow_torchvision_fallback=true."
            )
            self.status = DeepReIDStatus(False, "none", message)
            if self.require_backend:
                print(f"[ReID WARNING] {message}")
            return

        if wants_torchvision and self.allow_torchvision_fallback:
            if self._try_init_torchvision():
                return

        if wants_torchvision and not self.allow_torchvision_fallback:
            message = "Torchvision fallback is disabled; no OSNet backend was available."
        else:
            message = "No deep ReID backend available; HSV descriptors will be used only as a last-resort descriptor."
        self.status = DeepReIDStatus(False, "none", message)
        if self.require_backend:
            print(f"[ReID WARNING] {message}")

    def _select_device(self) -> str:
        try:
            import torch

            if self.device and self.device != "auto":
                return self.device
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _try_init_torchreid(self) -> bool:
        try:
            FeatureExtractor = _import_torchreid_feature_extractor()
        except Exception as exc:
            self.status = DeepReIDStatus(False, "torchreid", f"Torchreid import failed: {type(exc).__name__}: {exc}")
            return False

        try:
            device = self._select_device()
            # FeatureExtractor accepts numpy RGB images and returns feature tensors.
            self._extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=self.model_path,
                image_size=(self.image_height, self.image_width),
                device=device,
                verbose=False,
            )
            self.device = device
            self.backend = "torchreid"
            self.status = DeepReIDStatus(True, "torchreid", f"Using Torchreid/OSNet backend: {self.model_name} on {device}")
            return True
        except Exception as exc:
            self.status = DeepReIDStatus(False, "torchreid", f"Torchreid initialization failed: {type(exc).__name__}: {exc}")
            return False

    def _try_init_torchvision(self) -> bool:
        try:
            import torch
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except Exception as exc:
            self.status = DeepReIDStatus(False, "torchreid", f"Torchreid import failed: {type(exc).__name__}: {exc}")
            return False

        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.eval()
            device = self._select_device()
            model.to(device)
            self._torch = torch
            self._torchvision_model = model
            self._torchvision_weights = weights
            self.device = device
            self.backend = "torchvision"
            self.status = DeepReIDStatus(True, "torchvision", f"Using Torchvision MobileNetV3 fallback on {device}")
            return True
        except Exception as exc:
            self.status = DeepReIDStatus(False, "torchvision", f"Torchvision initialization failed: {exc}")
            return False

    def is_available(self) -> bool:
        return bool(self.status.enabled)

    def describe(self) -> str:
        return self.status.message

    def extract(self, crop_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0 or not self.is_available():
            return None
        if crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 10:
            return None

        if self.backend == "torchreid":
            return self._extract_torchreid(crop_bgr)
        if self.backend == "torchvision":
            return self._extract_torchvision(crop_bgr)
        return None

    def _extract_torchreid(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._extractor is None:
            return None
        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_rgb = cv2.resize(crop_rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
            features = self._extractor([crop_rgb])
            if hasattr(features, "detach"):
                features = features.detach().cpu().numpy()
            features_np = np.asarray(features, dtype=np.float32)
            vector = features_np[0] if features_np.ndim == 2 else features_np.reshape(-1)
            return _l2_normalize(vector)
        except Exception:
            return None

    def _extract_torchvision(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._torch is None or self._torchvision_model is None or self._torchvision_weights is None:
            return None
        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            preprocess = self._torchvision_weights.transforms()
            tensor = self._torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
            tensor = preprocess(tensor).unsqueeze(0).to(self.device)
            with self._torch.inference_mode():
                x = self._torchvision_model.features(tensor)
                x = self._torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            vector = x.detach().cpu().numpy().reshape(-1).astype(np.float32)
            return _l2_normalize(vector)
        except Exception:
            return None


def _l2_normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def _import_torchreid_feature_extractor():
    """Import FeatureExtractor across different torchreid package layouts.

    PyPI torchreid==0.2.5 exposes FeatureExtractor under
    torchreid.reid.utils, while some documentation/examples use
    torchreid.utils. Supporting both keeps the project portable.
    """
    try:
        from torchreid.utils import FeatureExtractor  # type: ignore

        return FeatureExtractor
    except ModuleNotFoundError as first_exc:
        # PyPI torchreid 0.2.5 commonly has no top-level torchreid.utils.
        if "torchreid.utils" not in str(first_exc):
            raise
        from torchreid.reid.utils import FeatureExtractor  # type: ignore

        return FeatureExtractor
