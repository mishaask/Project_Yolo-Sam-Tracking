from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from screening_ai.deep_reid import DeepPersonReID


def import_feature_extractor_for_diagnostic():
    try:
        from torchreid.utils import FeatureExtractor  # type: ignore

        print("Imported FeatureExtractor from torchreid.utils")
        return FeatureExtractor
    except ModuleNotFoundError as first_exc:
        if "torchreid.utils" not in str(first_exc):
            raise
        from torchreid.reid.utils import FeatureExtractor  # type: ignore

        print("Imported FeatureExtractor from torchreid.reid.utils")
        return FeatureExtractor


def main() -> None:
    print("Checking Torchreid/OSNet backend...")
    extractor = DeepPersonReID(
        enabled=True,
        backend="torchreid",
        model_name="osnet_x0_25",
        allow_torchvision_fallback=False,
        require_backend=True,
    )

    print(extractor.describe())

    if extractor.is_available():
        print("OSNet is available.")
        return

    print("\nDirect Torchreid FeatureExtractor diagnostic:")

    try:
        FeatureExtractor = import_feature_extractor_for_diagnostic()
        FeatureExtractor(
            model_name="osnet_x0_25",
            model_path="",
            image_size=(256, 128),
            device="cpu",
            verbose=False,
        )
        print("Direct Torchreid FeatureExtractor construction succeeded.")
    except Exception:
        traceback.print_exc()

    raise SystemExit(
        "\nOSNet is not available. Most common fixes: use Python 3.12, "
        "pin torch/torchvision to the stable ReID stack, keep gdown==4.7.3, "
        "and install tensorboard."
    )


if __name__ == "__main__":
    main()
