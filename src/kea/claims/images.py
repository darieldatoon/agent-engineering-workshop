"""Photo helpers: files and bytes to data URLs, and the photo on the LangSmith trace."""

from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import get_current_run_tree

from kea.config import DATA_DIR

PHOTO_DIR = DATA_DIR / "bag-photos"
IMAGE_ATTACHMENT = "image"
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def mime_for(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "image/jpeg")


def bytes_to_data_url(data: bytes, mime_type: str = "image/jpeg") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def image_to_data_url(path: Path) -> str:
    return bytes_to_data_url(path.read_bytes(), mime_for(path))


def parse_data_url(data_url: str) -> tuple[str, bytes]:
    header, _, payload = data_url.partition(",")
    return header.removeprefix("data:").removesuffix(";base64"), base64.b64decode(payload)


def resolve_photo(photo: str) -> str:
    """A data URL passes through; a bare file name loads from `data/bag-photos` (handy in Studio)."""
    if photo.startswith("data:"):
        return photo
    return image_to_data_url(PHOTO_DIR / Path(photo).name)


def attach_image_to_root_run(config: RunnableConfig, data_url: str) -> None:
    """Put the photo on the trace's root run so it shows in the UI and in add-to-dataset."""
    current = get_current_run_tree()
    callbacks = config.get("callbacks")
    if current is None or callbacks is None:
        return
    mime_type, data = parse_data_url(data_url)
    handlers = callbacks if isinstance(callbacks, list) else callbacks.handlers
    for handler in handlers:
        if isinstance(handler, LangChainTracer) and (
            root := handler.run_map.get(str(current.trace_id))
        ):
            # Only the root run's tracer record uploads attachments with the run.
            root.attachments[IMAGE_ATTACHMENT] = (mime_type, data)  # ty: ignore[invalid-assignment]
