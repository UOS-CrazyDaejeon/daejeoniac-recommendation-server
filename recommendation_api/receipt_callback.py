"""OCR 완료 결과를 Spring 서버에 전달하는 콜백 기능."""

from __future__ import annotations

from datetime import date, datetime, time
import os
from typing import Any


class ReceiptOcrCallbackError(RuntimeError):
    status_code = 502
    error_code = "OCR_RESULT_CALLBACK_FAILED"


class ReceiptOcrCallbackConfigurationError(ReceiptOcrCallbackError):
    status_code = 503
    error_code = "OCR_RESULT_CALLBACK_NOT_CONFIGURED"


def _required_callback_url() -> str:
    callback_url = os.environ.get("SPRING_OCR_CALLBACK_URL", "").strip()
    if not callback_url:
        raise ReceiptOcrCallbackConfigurationError(
            "SPRING_OCR_CALLBACK_URL 설정이 필요합니다."
        )
    if not callback_url.startswith(("http://", "https://")):
        raise ReceiptOcrCallbackConfigurationError(
            "SPRING_OCR_CALLBACK_URL은 http 또는 https URL이어야 합니다."
        )
    return callback_url


def _callback_timeout_seconds() -> float:
    try:
        timeout = float(os.environ.get("SPRING_OCR_CALLBACK_TIMEOUT_SECONDS", "5"))
    except ValueError as exc:
        raise ReceiptOcrCallbackConfigurationError(
            "SPRING_OCR_CALLBACK_TIMEOUT_SECONDS는 숫자여야 합니다."
        ) from exc
    if not 1 <= timeout <= 30:
        raise ReceiptOcrCallbackConfigurationError(
            "SPRING_OCR_CALLBACK_TIMEOUT_SECONDS는 1~30초여야 합니다."
        )
    return timeout


def _ocr_paid_at(result: dict[str, Any]) -> str | None:
    raw_date = result.get("transactionDate")
    raw_time = result.get("transactionTime")
    if not isinstance(raw_date, str) or not isinstance(raw_time, str):
        return None

    try:
        paid_date = date.fromisoformat(raw_date)
        paid_time = time.fromisoformat(raw_time)
    except ValueError:
        return None
    return datetime.combine(paid_date, paid_time).isoformat(timespec="seconds")


def _ocr_success_status() -> str:
    """Spring Receipt.OcrStatus enum과 같은 완료 상태를 사용한다."""
    return os.environ.get("SPRING_OCR_SUCCESS_STATUS", "SUCCESS").strip() or "SUCCESS"


def build_ocr_result_fields(result: dict[str, Any]) -> dict[str, str | None]:
    """외부에 노출할 영수증 OCR 핵심 결과 네 필드만 만든다."""
    place_name = result.get("merchantName")
    place_address = result.get("address")
    return {
        "ocrStatus": _ocr_success_status(),
        "ocrPlaceName": str(place_name) if place_name is not None else None,
        "ocrPlaceAddress": str(place_address) if place_address is not None else None,
        "ocrPaidAt": _ocr_paid_at(result),
    }


def build_ocr_result_callback_payload(
    receipt_uuid: str,
    result: dict[str, Any],
) -> dict[str, str | None]:
    """Spring의 /api/v1/receipts/ocr-result 요청 계약에 맞춰 결과를 변환한다."""
    return {"receiptUuid": receipt_uuid, **build_ocr_result_fields(result)}


def post_ocr_result_callback(
    payload: dict[str, str | None],
    *,
    client: Any | None = None,
) -> None:
    """Spring OCR 결과 API가 2xx를 반환할 때만 정상 처리한다."""
    callback_url = _required_callback_url()
    headers = {"Content-Type": "application/json"}
    authorization = os.environ.get("SPRING_OCR_CALLBACK_AUTHORIZATION", "").strip()
    if authorization:
        headers["Authorization"] = authorization

    try:
        if client is not None:
            response = client.post(callback_url, json=payload, headers=headers)
            response.raise_for_status()
            return

        import httpx

        with httpx.Client(timeout=_callback_timeout_seconds()) as http_client:
            response = http_client.post(callback_url, json=payload, headers=headers)
            response.raise_for_status()
    except ReceiptOcrCallbackError:
        raise
    except Exception as exc:
        raise ReceiptOcrCallbackError(
            "Spring OCR 결과 콜백 요청에 실패했습니다."
        ) from exc
