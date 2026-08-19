"""S3에서 원본 이미지를 읽고 모자이크 결과를 저장하는 기능."""

from __future__ import annotations

from dataclasses import replace
import os
from typing import Any
from uuid import uuid4

from recommendation_api.face_mosaic import FaceMosaicResult
from recommendation_api.receipt_s3 import (
    S3ReceiptConfigurationError,
    S3ReceiptObject,
    S3ReceiptSettings,
    configured_s3_client,
    load_s3_receipt_object,
)


class S3MosaicUploadError(RuntimeError):
    status_code = 502
    error_code = "S3_MOSAIC_UPLOAD_FAILED"


def _environment_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _normalized_prefix(value: str | None, default: str) -> str:
    prefix = (value or default).strip().strip("/")
    if (
        not prefix
        or prefix.startswith(("s3:", "http:", "https:"))
        or any(part in {".", ".."} for part in prefix.split("/"))
    ):
        raise S3ReceiptConfigurationError("S3 모자이크 저장 경로가 올바르지 않습니다.")
    return prefix + "/"


def load_s3_face_image(
    s3_key: str,
    *,
    max_bytes: int,
    client: Any | None = None,
) -> S3ReceiptObject:
    """모자이크할 이미지를 설정된 단일 버킷에서 읽는다."""
    base_settings = S3ReceiptSettings.from_environment()
    image_prefix = _environment_value(
        "AWS_S3_IMAGE_PREFIX",
        "aws.s3.image-prefix",
    )
    settings = replace(base_settings, allowed_prefix=image_prefix)
    return load_s3_receipt_object(
        s3_key,
        max_bytes=max_bytes,
        client=client,
        settings=settings,
    )


def store_s3_mosaic_image(
    result: FaceMosaicResult,
    *,
    client: Any | None = None,
) -> str:
    """모자이크 결과를 새 객체로 저장하고 객체 키를 반환한다."""
    settings = S3ReceiptSettings.from_environment()
    output_prefix = _normalized_prefix(
        _environment_value(
            "AWS_S3_MOSAIC_PREFIX",
            "aws.s3.mosaic-prefix",
        ),
        "mosaics",
    )
    output_key = f"{output_prefix}{uuid4().hex}-mosaic.jpg"
    arguments: dict[str, Any] = {
        "Bucket": settings.bucket,
        "Key": output_key,
        "Body": result.image_bytes,
        "ContentType": result.content_type,
    }
    if settings.expected_bucket_owner:
        arguments["ExpectedBucketOwner"] = settings.expected_bucket_owner

    try:
        (client or configured_s3_client()).put_object(**arguments)
    except Exception as exc:
        try:
            from botocore.exceptions import NoCredentialsError
        except ImportError as import_error:
            raise S3ReceiptConfigurationError(
                "S3 연동에 필요한 boto3 패키지가 설치되지 않았습니다."
            ) from import_error
        if isinstance(exc, NoCredentialsError):
            raise S3ReceiptConfigurationError(
                "S3 접근 자격 증명을 찾지 못했습니다."
            ) from exc
        raise S3MosaicUploadError(
            "모자이크 이미지를 S3에 저장하지 못했습니다."
        ) from exc
    return output_key
