from collections.abc import Callable
from datetime import datetime, timezone
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from recommend_llm import process_spring_recommendation_request
from recommendation_api.receipt_gpt import (
    ReceiptVisionConfigurationError,
    ReceiptVisionUpstreamError,
    analyze_receipt_image_with_gpt,
)
from recommendation_api.receipt_ocr import (
    ReceiptDocumentError,
    analyze_receipt_image_bytes,
)
from recommendation_api.schemas import (
    GptReceiptAnalysisResponse,
    ReceiptAnalysisResponse,
    RecommendationRequest,
    RecommendationResponse,
)


RecommendationProcessor = Callable[[dict[str, Any]], dict[str, Any]]
ReceiptProcessor = Callable[[bytes, str], dict[str, Any]]
GptReceiptProcessor = Callable[[bytes, str], dict[str, Any]]
logger = logging.getLogger(__name__)

MAX_RECEIPT_IMAGE_BYTES = int(
    os.environ.get("MAX_RECEIPT_IMAGE_BYTES", 10 * 1024 * 1024)
)
OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "kor+eng").strip() or "kor+eng"


app = FastAPI(title="Daejeon Internal API", version="1.2.0")


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def _request_id_from_body(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    request_id = body.get("request_id")
    return str(request_id) if request_id is not None else None


@app.exception_handler(RequestValidationError)
def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        status_code=422,
        code="INVALID_REQUEST_BODY",
        message="요청 본문이 올바르지 않습니다",
        request_id=_request_id_from_body(exc.body),
    )


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_recommendation_processor() -> RecommendationProcessor:
    return process_spring_recommendation_request


def get_receipt_processor() -> ReceiptProcessor:
    return lambda image_bytes, language: analyze_receipt_image_bytes(
        image_bytes,
        language=language,
    )


def get_gpt_receipt_processor() -> GptReceiptProcessor:
    return analyze_receipt_image_with_gpt


@app.post(
    "/api/v1/recommendations",
    response_model=RecommendationResponse,
    tags=["recommendations"],
)
def create_recommendations(
    request: RecommendationRequest,
    processor: RecommendationProcessor = Depends(get_recommendation_processor),
) -> dict[str, Any] | JSONResponse:
    try:
        return processor(request.model_dump(mode="json"))
    except ValueError as exc:
        return _error_response(
            status_code=400,
            code="INVALID_RECOMMENDATION_REQUEST",
            message=str(exc),
            request_id=request.request_id,
        )
    except Exception:
        logger.exception(
            "Unexpected recommendation failure request_id=%s",
            request.request_id,
        )
        return _error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="추천 처리 중 오류가 발생했습니다",
            request_id=request.request_id,
        )


@app.post(
    "/api/v1/receipts/analyze",
    response_model=ReceiptAnalysisResponse,
    tags=["receipts"],
)
def analyze_receipt(
    image: UploadFile = File(..., description="OCR 처리할 영수증 이미지"),
    requestId: str | None = Form(default=None),
    documentId: str | None = Form(default=None),
    userId: int | None = Form(default=None),
    processor: ReceiptProcessor = Depends(get_receipt_processor),
) -> dict[str, Any] | JSONResponse:
    """Analyze one uploaded receipt image and return structured receipt data."""
    if image.content_type and not (
        image.content_type.startswith("image/")
        or image.content_type == "application/octet-stream"
    ):
        return _error_response(
            status_code=415,
            code="UNSUPPORTED_RECEIPT_MEDIA_TYPE",
            message="영수증 이미지 파일만 업로드할 수 있습니다",
            request_id=requestId,
        )

    image_bytes = image.file.read(MAX_RECEIPT_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_RECEIPT_IMAGE_BYTES:
        return _error_response(
            status_code=413,
            code="RECEIPT_IMAGE_TOO_LARGE",
            message=f"영수증 이미지는 {MAX_RECEIPT_IMAGE_BYTES // (1024 * 1024)}MB 이하여야 합니다",
            request_id=requestId,
        )

    try:
        analysis = processor(image_bytes, OCR_LANGUAGE)
        result = analysis["result"]
        raw_ocr_text = analysis.get("rawOcrText", "")
        return {
            "requestId": requestId,
            "documentId": documentId,
            "userId": userId,
            "documentType": "RECEIPT",
            "status": "COMPLETED",
            "result": result,
            "warnings": result.get("warnings", []),
            "processedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            # OCR 원문에는 개인정보가 포함될 수 있어 길이만 반환한다.
            "rawOcrCharCount": len(raw_ocr_text),
        }
    except ReceiptDocumentError as exc:
        return _error_response(
            status_code=422,
            code="RECEIPT_ANALYSIS_FAILED",
            message=str(exc),
            request_id=requestId,
        )
    except Exception:
        logger.exception(
            "Unexpected receipt analysis failure request_id=%s document_id=%s",
            requestId,
            documentId,
        )
        return _error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="영수증 처리 중 오류가 발생했습니다",
            request_id=requestId,
        )


@app.post(
    "/api/v1/receipts/analyze-gpt-mini",
    response_model=GptReceiptAnalysisResponse,
    tags=["receipts"],
)
def analyze_receipt_with_gpt_mini(
    image: UploadFile = File(..., description="GPT로 분석할 영수증 이미지"),
    requestId: str | None = Form(default=None),
    documentId: str | None = Form(default=None),
    userId: int | None = Form(default=None),
    processor: GptReceiptProcessor = Depends(get_gpt_receipt_processor),
) -> dict[str, Any] | JSONResponse:
    """Analyze one receipt image using the configured gpt-5-mini model."""
    if image.content_type and not (
        image.content_type.startswith("image/")
        or image.content_type == "application/octet-stream"
    ):
        return _error_response(
            status_code=415,
            code="UNSUPPORTED_RECEIPT_MEDIA_TYPE",
            message="영수증 이미지 파일만 업로드할 수 있습니다",
            request_id=requestId,
        )

    image_bytes = image.file.read(MAX_RECEIPT_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_RECEIPT_IMAGE_BYTES:
        return _error_response(
            status_code=413,
            code="RECEIPT_IMAGE_TOO_LARGE",
            message=f"영수증 이미지는 {MAX_RECEIPT_IMAGE_BYTES // (1024 * 1024)}MB 이하여야 합니다",
            request_id=requestId,
        )

    try:
        analysis = processor(
            image_bytes,
            image.content_type or "application/octet-stream",
        )
        result = analysis["result"]
        return {
            "requestId": requestId,
            "documentId": documentId,
            "userId": userId,
            "documentType": "RECEIPT",
            "status": "COMPLETED",
            "model": analysis["model"],
            "result": result,
            "warnings": result.get("warnings", []),
            "processedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "processingTimeMs": analysis["processingTimeMs"],
            "usage": analysis.get("usage"),
        }
    except ReceiptVisionConfigurationError as exc:
        return _error_response(
            status_code=503,
            code="RECEIPT_VISION_NOT_CONFIGURED",
            message=str(exc),
            request_id=requestId,
        )
    except ReceiptDocumentError as exc:
        return _error_response(
            status_code=422,
            code="RECEIPT_ANALYSIS_FAILED",
            message=str(exc),
            request_id=requestId,
        )
    except ReceiptVisionUpstreamError as exc:
        logger.warning(
            "GPT receipt upstream failure request_id=%s document_id=%s: %s",
            requestId,
            documentId,
            exc,
        )
        return _error_response(
            status_code=502,
            code="RECEIPT_VISION_UPSTREAM_ERROR",
            message=str(exc),
            request_id=requestId,
        )
    except Exception:
        logger.exception(
            "Unexpected GPT receipt failure request_id=%s document_id=%s",
            requestId,
            documentId,
        )
        return _error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="GPT 영수증 처리 중 오류가 발생했습니다",
            request_id=requestId,
        )
