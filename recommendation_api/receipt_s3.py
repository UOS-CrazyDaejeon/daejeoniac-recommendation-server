"""S3에 저장된 영수증 이미지를 제한된 크기로 읽는 기능."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any


class S3ReceiptError(RuntimeError):
    """S3 영수증 객체를 가져오지 못했을 때 사용하는 기본 오류."""

    status_code = 502
    error_code = "S3_RECEIPT_ACCESS_FAILED"


class S3ReceiptConfigurationError(S3ReceiptError):
    status_code = 503
    error_code = "S3_RECEIPT_NOT_CONFIGURED"


class S3ReceiptKeyError(S3ReceiptError):
    status_code = 400
    error_code = "INVALID_S3_RECEIPT_KEY"


class S3ReceiptNotFoundError(S3ReceiptError):
    status_code = 404
    error_code = "S3_RECEIPT_NOT_FOUND"


class S3ReceiptTooLargeError(S3ReceiptError):
    status_code = 413
    error_code = "RECEIPT_IMAGE_TOO_LARGE"


class S3ReceiptMediaTypeError(S3ReceiptError):
    status_code = 415
    error_code = "UNSUPPORTED_RECEIPT_MEDIA_TYPE"


class S3ReceiptAccessError(S3ReceiptError):
    status_code = 502
    error_code = "S3_RECEIPT_ACCESS_FAILED"


def _first_environment_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class S3ReceiptSettings:
    region: str
    bucket: str
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    allowed_prefix: str | None = None
    expected_bucket_owner: str | None = None

    @classmethod
    def from_environment(cls) -> "S3ReceiptSettings":
        region = _first_environment_value(
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "aws.region",
        )
        bucket = _first_environment_value("AWS_S3_BUCKET", "aws.s3.bucket")
        if not region or not bucket:
            raise S3ReceiptConfigurationError(
                "S3 영수증 분석에는 AWS 리전과 S3 버킷 설정이 필요합니다."
            )

        access_key_id = _first_environment_value(
            "AWS_ACCESS_KEY_ID",
            "aws.access-key",
        )
        secret_access_key = _first_environment_value(
            "AWS_SECRET_ACCESS_KEY",
            "aws.secret-key",
        )
        if bool(access_key_id) != bool(secret_access_key):
            raise S3ReceiptConfigurationError(
                "AWS 액세스 키와 시크릿 키는 함께 설정해야 합니다."
            )

        return cls(
            region=region,
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=_first_environment_value(
                "AWS_SESSION_TOKEN",
                "aws.session-token",
            ),
            allowed_prefix=_first_environment_value(
                "AWS_S3_RECEIPT_PREFIX",
                "aws.s3.receipt-prefix",
            ),
            expected_bucket_owner=_first_environment_value(
                "AWS_S3_EXPECTED_BUCKET_OWNER",
                "aws.s3.expected-bucket-owner",
            ),
        )


@dataclass(frozen=True)
class S3ReceiptObject:
    key: str
    content_type: str
    image_bytes: bytes


def _validate_object_key(key: str, allowed_prefix: str | None) -> str:
    normalized = key.strip()
    if not normalized:
        raise S3ReceiptKeyError("s3Key는 비어 있을 수 없습니다.")
    if normalized.startswith(("s3://", "http://", "https://")):
        raise S3ReceiptKeyError(
            "S3 URL이 아니라 버킷 내부 객체 키만 s3Key로 보내야 합니다."
        )
    if "\x00" in normalized or len(normalized.encode("utf-8")) > 1024:
        raise S3ReceiptKeyError("s3Key 형식이 올바르지 않습니다.")

    if allowed_prefix:
        prefix = allowed_prefix.strip().lstrip("/")
        candidate = normalized.lstrip("/")
        normalized_prefix = prefix.rstrip("/") + "/"
        if not candidate.startswith(normalized_prefix):
            raise S3ReceiptKeyError(
                f"s3Key는 허용된 경로({normalized_prefix}) 안에 있어야 합니다."
            )
        return candidate
    return normalized.lstrip("/")


@lru_cache(maxsize=1)
def configured_s3_client() -> Any:
    settings = S3ReceiptSettings.from_environment()

    import boto3
    from botocore.config import Config

    arguments: dict[str, Any] = {
        "region_name": settings.region,
        "config": Config(
            connect_timeout=3,
            read_timeout=15,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    }
    if settings.access_key_id and settings.secret_access_key:
        arguments["aws_access_key_id"] = settings.access_key_id
        arguments["aws_secret_access_key"] = settings.secret_access_key
        if settings.session_token:
            arguments["aws_session_token"] = settings.session_token
    return boto3.client("s3", **arguments)


def load_s3_receipt_object(
    s3_key: str,
    *,
    max_bytes: int,
    client: Any | None = None,
    settings: S3ReceiptSettings | None = None,
) -> S3ReceiptObject:
    """설정된 단일 버킷에서 영수증 이미지 객체를 안전하게 읽는다."""
    selected_settings = settings or S3ReceiptSettings.from_environment()
    object_key = _validate_object_key(s3_key, selected_settings.allowed_prefix)
    get_arguments: dict[str, Any] = {
        "Bucket": selected_settings.bucket,
        "Key": object_key,
    }
    if selected_settings.expected_bucket_owner:
        get_arguments["ExpectedBucketOwner"] = (
            selected_settings.expected_bucket_owner
        )

    try:
        response = (client or configured_s3_client()).get_object(**get_arguments)
    except S3ReceiptError:
        raise
    except Exception as exc:
        try:
            from botocore.exceptions import ClientError, NoCredentialsError
        except ImportError as import_error:
            raise S3ReceiptConfigurationError(
                "S3 연동에 필요한 boto3 패키지가 설치되지 않았습니다."
            ) from import_error

        if isinstance(exc, NoCredentialsError):
            raise S3ReceiptConfigurationError(
                "S3 접근 자격 증명을 찾지 못했습니다."
            ) from exc
        if isinstance(exc, ClientError):
            error = exc.response.get("Error", {})
            error_code = str(error.get("Code", ""))
            status = int(
                exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
                or 0
            )
            if error_code in {"NoSuchKey", "NotFound", "404"} or status == 404:
                raise S3ReceiptNotFoundError(
                    "S3에서 영수증 이미지를 찾지 못했습니다."
                ) from exc
        raise S3ReceiptAccessError(
            "S3에서 영수증 이미지를 가져오지 못했습니다."
        ) from exc

    body = response.get("Body")
    if body is None:
        raise S3ReceiptAccessError("S3 객체 응답에 이미지 본문이 없습니다.")

    try:
        content_length = int(response.get("ContentLength", 0) or 0)
        if content_length > max_bytes:
            raise S3ReceiptTooLargeError(
                f"영수증 이미지는 {max_bytes // (1024 * 1024)}MB 이하여야 합니다."
            )

        content_type = str(
            response.get("ContentType", "application/octet-stream")
            or "application/octet-stream"
        ).lower()
        if not (
            content_type.startswith("image/")
            or content_type == "application/octet-stream"
        ):
            raise S3ReceiptMediaTypeError(
                "S3 객체가 지원되는 이미지 형식이 아닙니다."
            )

        image_bytes = body.read(max_bytes + 1)
        if len(image_bytes) > max_bytes:
            raise S3ReceiptTooLargeError(
                f"영수증 이미지는 {max_bytes // (1024 * 1024)}MB 이하여야 합니다."
            )
        if not image_bytes:
            raise S3ReceiptMediaTypeError("S3 영수증 이미지가 비어 있습니다.")
        return S3ReceiptObject(
            key=object_key,
            content_type=content_type,
            image_bytes=image_bytes,
        )
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
