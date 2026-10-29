"""Google Colab helpers for running the desktop pipeline without its GUI."""

from __future__ import annotations

import threading
import sys
from pathlib import Path

from mosaicdiff.fetch import _REMOTE
from mosaicdiff.paths import BUNDLED_NAMES, models_dir
from mosaicdiff.pipeline import process_video
from mosaicdiff.settings import Settings


def colab_settings(
    comfy_root: str | Path = "/content/ComfyUI",
    model_directory: str | Path | None = None,
    output_directory: str | Path = "/content/MosaicDiff/Output",
    h3_seconds: int = 5,
    h3_resolution: int = 800,
    compare: bool = False,
) -> Settings:
    """Create settings suitable for the Linux Python runtime in Colab."""
    root = Path(comfy_root).expanduser().resolve()
    model_root = Path(model_directory or models_dir()).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    settings = Settings(
        output_dir=str(output),
        h3_seconds=max(1, min(15, int(h3_seconds))),
        h3_resolution=max(512, min(1280, int(h3_resolution) // 32 * 32)),
        compare=bool(compare),
        paths={
            "comfy_root": str(root),
            "comfy_python": sys.executable,
        },
    )
    for key, filename in BUNDLED_NAMES.items():
        settings.paths[key] = str(model_root / filename)
    output.mkdir(parents=True, exist_ok=True)
    return settings


def process_colab_video(
    source: str | Path,
    destination: str | Path,
    settings: Settings,
) -> Path:
    """Process one video with the same pipeline used by the desktop app."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    cancel = threading.Event()

    def log(message: str) -> None:
        print(message, flush=True)

    def progress(fraction: float, message: str) -> None:
        print(f"[{max(0.0, min(1.0, float(fraction))) * 100:5.1f}%] {message}", flush=True)

    return process_video(
        source_path,
        destination_path,
        settings,
        log,
        progress,
        cancel,
    )


def process_colab_videos(
    sources: list[str | Path],
    settings: Settings,
) -> list[Path]:
    """Process videos in order and return their output paths."""
    results = []
    output = Path(settings.output_dir)
    for source in sources:
        source_path = Path(source)
        destination = output / f"{source_path.stem}_mosaicdiff.mp4"
        results.append(process_colab_video(source_path, destination, settings))
    return results


def download_huggingface_models(
    model_directory: str | Path,
    token: str = "",
    detector_repo: str = "",
    detector_file: str = "rfdetr.onnx",
    lora_repo: str = "",
    lora_file: str = "lora.safetensors",
    log=print,
) -> Path:
    """Download all six model files from Hugging Face into one local folder.

    The detector and LoRA sources are configurable because this repository does
    not publish an official source for either file.
    """
    from huggingface_hub import hf_hub_download, login

    if token.strip():
        login(token=token.strip(), add_to_git_credential=False)
    model_root = Path(model_directory).expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    sources = dict(_REMOTE)
    if detector_repo.strip():
        sources["detector"] = (detector_repo.strip(), detector_file.strip(), None)
    if lora_repo.strip():
        sources["lora"] = (lora_repo.strip(), lora_file.strip(), None)
    missing_sources = [key for key in ("detector", "lora") if key not in sources]
    if missing_sources:
        raise ValueError(
            "Set detector_repo and lora_repo to Hugging Face repositories "
            "containing the official model files before downloading."
        )
    for key in BUNDLED_NAMES:
        repo, filename, revision = sources[key]
        destination = model_root / BUNDLED_NAMES[key]
        if destination.is_file():
            log(f"Already present: {destination.name}")
            continue
        log(f"Downloading {destination.name} from {repo}")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_dir=str(model_root),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != destination.resolve():
            downloaded_path.replace(destination)
        log(f"Saved {destination}")
    return model_root
