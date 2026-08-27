"""CPU만 사용해 사진 속 얼굴을 검출하고 모자이크 처리한다."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any

from pillow_heif import register_heif_opener


register_heif_opener(thumbnails=False)


class FaceMosaicError(RuntimeError):
    status_code = 422
    error_code = "FACE_MOSAIC_FAILED"


class FaceMosaicConfigurationError(FaceMosaicError):
    status_code = 503
    error_code = "FACE_MOSAIC_NOT_CONFIGURED"


class FaceImageError(FaceMosaicError):
    status_code = 422
    error_code = "INVALID_FACE_IMAGE"


class FaceNotDetectedError(FaceMosaicError):
    """원본을 그대로 공개하는 일을 막기 위해 검출 0건을 실패로 처리한다."""

    status_code = 422
    error_code = "FACE_NOT_DETECTED"


@dataclass(frozen=True)
class FaceMosaicResult:
    image_bytes: bytes
    content_type: str
    face_count: int
    width: int
    height: int


_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)
_DETECTOR_LOCK = Lock()


def _integer_environment(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise FaceMosaicConfigurationError(f"{name} 설정은 정수여야 합니다.") from exc
    if value < minimum or value > maximum:
        raise FaceMosaicConfigurationError(
            f"{name} 설정은 {minimum}~{maximum} 범위여야 합니다."
        )
    return value


def _float_environment(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise FaceMosaicConfigurationError(f"{name} 설정은 숫자여야 합니다.") from exc
    if value < minimum or value > maximum:
        raise FaceMosaicConfigurationError(
            f"{name} 설정은 {minimum}~{maximum} 범위여야 합니다."
        )
    return value


@lru_cache(maxsize=1)
def _processing_semaphore() -> BoundedSemaphore:
    # t3.small에서 여러 고해상도 이미지를 동시에 풀지 않게 한다.
    concurrent_jobs = _integer_environment(
        "FACE_MAX_CONCURRENT_JOBS",
        1,
        1,
        2,
    )
    return BoundedSemaphore(concurrent_jobs)


@lru_cache(maxsize=1)
def configured_face_detector() -> Any:
    """YuNet 모델을 한 번만 로드해 t3.small의 메모리 사용량을 줄인다."""
    try:
        import cv2
    except ImportError as exc:
        raise FaceMosaicConfigurationError(
            "얼굴 모자이크에 필요한 OpenCV가 설치되지 않았습니다."
        ) from exc

    model_path = Path(os.environ.get("FACE_DETECTION_MODEL_PATH", _DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        raise FaceMosaicConfigurationError(
            f"YuNet 얼굴 검출 모델을 찾지 못했습니다: {model_path}"
        )

    # 요청 스레드마다 OpenCV가 CPU 스레드를 많이 만들지 않게 한다.
    cv2.setNumThreads(_integer_environment("FACE_OPENCV_THREADS", 1, 1, 2))
    score_threshold = _float_environment(
        "FACE_DETECTION_SCORE_THRESHOLD", 0.75, 0.1, 0.99
    )
    return cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (320, 320),
        score_threshold,
        0.3,
        500,
    )


def _decode_image(image_bytes: bytes) -> Any:
    if not image_bytes:
        raise FaceImageError("이미지가 비어 있습니다.")

    try:
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(BytesIO(image_bytes)) as source:
            if source.width * source.height > _integer_environment(
                "FACE_MAX_IMAGE_PIXELS", 24_000_000, 1_000_000, 50_000_000
            ):
                raise FaceImageError("이미지 해상도가 너무 큽니다.")
            image = ImageOps.exif_transpose(source).convert("RGB")

        max_output_edge = _integer_environment(
            "FACE_OUTPUT_MAX_EDGE", 2560, 640, 4096
        )
        if max(image.size) > max_output_edge:
            ratio = max_output_edge / max(image.size)
            resized_size = (
                max(1, round(image.width * ratio)),
                max(1, round(image.height * ratio)),
            )
            image = image.resize(resized_size, Image.Resampling.LANCZOS)

        # Pillow RGB -> OpenCV BGR. copy() 없이는 음수 stride로 인해 DNN이 실패할 수 있다.
        return np.asarray(image, dtype=np.uint8)[:, :, ::-1].copy()
    except FaceMosaicError:
        raise
    except Exception as exc:
        raise FaceImageError(
            "JPEG, PNG, WebP, HEIC/HEIF 이미지를 읽지 못했습니다."
        ) from exc


def _detect_face_boxes(image: Any, detector: Any) -> list[tuple[int, int, int, int]]:
    import cv2

    height, width = image.shape[:2]
    detection_edge = _integer_environment(
        "FACE_DETECTION_MAX_EDGE", 1280, 320, 1920
    )
    scale = min(1.0, detection_edge / max(width, height))
    if scale < 1.0:
        detect_width = max(1, round(width * scale))
        detect_height = max(1, round(height * scale))
        detection_image = cv2.resize(
            image,
            (detect_width, detect_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detect_width, detect_height = width, height
        detection_image = image

    # FaceDetectorYN은 setInputSize/detect가 같은 인스턴스에서 thread-safe하지 않다.
    with _DETECTOR_LOCK:
        detector.setInputSize((detect_width, detect_height))
        _, faces = detector.detect(detection_image)

    if faces is None:
        return []

    padding_ratio = _float_environment("FACE_BOX_PADDING", 0.18, 0.0, 0.5)
    inverse_scale = 1.0 / scale
    boxes: list[tuple[int, int, int, int]] = []
    for face in faces:
        x, y, box_width, box_height = (
            float(face[0]) * inverse_scale,
            float(face[1]) * inverse_scale,
            float(face[2]) * inverse_scale,
            float(face[3]) * inverse_scale,
        )
        pad_x = box_width * padding_ratio
        pad_y = box_height * padding_ratio
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(width, int(x + box_width + pad_x))
        y2 = min(height, int(y + box_height + pad_y))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _blur_faces(image: Any, boxes: list[tuple[int, int, int, int]]) -> None:
    import cv2

    # 예시처럼 사각형 블록 없이 자연스럽게 얼굴 전체를 흐리게 한다.
    # 얼굴의 짧은 변을 기준으로 계산해 가까이 찍힌 얼굴과 작은 얼굴에 모두 대응한다.
    blur_ratio = _float_environment("FACE_BLUR_SIGMA_RATIO", 0.09, 0.02, 0.20)
    for x1, y1, x2, y2 in boxes:
        face = image[y1:y2, x1:x2]
        height, width = face.shape[:2]
        sigma = max(1.0, min(width, height) * blur_ratio)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(
            face,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )


def mosaic_face_image_bytes(
    image_bytes: bytes,
    content_type: str = "application/octet-stream",
    *,
    detector: Any | None = None,
) -> FaceMosaicResult:
    """얼굴을 검출해 모자이크한 JPEG 이미지를 만든다."""
    with _processing_semaphore():
        return _mosaic_face_image_bytes(image_bytes, content_type, detector=detector)


def _mosaic_face_image_bytes(
    image_bytes: bytes,
    content_type: str,
    *,
    detector: Any | None,
) -> FaceMosaicResult:
    del content_type  # 실제 형식은 파일 내용을 Pillow로 판별한다.
    image = _decode_image(image_bytes)
    selected_detector = detector or configured_face_detector()
    boxes = _detect_face_boxes(image, selected_detector)
    if not boxes:
        raise FaceNotDetectedError(
            "얼굴을 검출하지 못해 원본 이미지를 저장하지 않았습니다."
        )

    _blur_faces(image, boxes)
    try:
        import cv2

        encoded, output = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    except Exception as exc:
        raise FaceMosaicError("모자이크 이미지 인코딩에 실패했습니다.") from exc
    if not encoded:
        raise FaceMosaicError("모자이크 이미지 인코딩에 실패했습니다.")

    height, width = image.shape[:2]
    return FaceMosaicResult(
        image_bytes=output.tobytes(),
        content_type="image/jpeg",
        face_count=len(boxes),
        width=width,
        height=height,
    )
