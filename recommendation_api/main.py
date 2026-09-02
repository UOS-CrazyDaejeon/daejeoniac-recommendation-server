from collections.abc import Callable
from datetime import datetime, timezone
import json
import logging
import os
from time import perf_counter
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from recommend_llm import (
    process_spring_next_places_request,
    process_spring_recommendation_request,
    process_spring_similar_places_request,
)
from recommendation_api.face_mosaic import (
    FaceMosaicError,
    FaceMosaicResult,
    mosaic_face_image_bytes,
)
from recommendation_api.face_mosaic_s3 import (
    S3MosaicUploadError,
    load_s3_face_image,
    store_s3_mosaic_image,
)
from recommendation_api.receipt_gpt import (
    ReceiptVisionConfigurationError,
    ReceiptVisionUpstreamError,
    analyze_receipt_image_with_gpt,
)
from recommendation_api.receipt_callback import (
    build_ocr_result_fields,
    build_ocr_result_callback_payload,
)
from recommendation_api.receipt_ocr import (
    ReceiptDocumentError,
    analyze_receipt_image_bytes,
)
from recommendation_api.receipt_s3 import (
    S3ReceiptError,
    S3ReceiptObjectList,
    S3ReceiptObject,
    list_s3_receipt_objects,
    load_s3_receipt_object,
)
from recommendation_api.schemas import (
    FaceMosaicResponse,
    NextPlacesRequest,
    NextPlacesResponse,
    ReceiptOcrFieldsResponse,
    ReceiptOcrRequest,
    ReceiptSpringOcrResponse,
    RecommendationRequest,
    RecommendationResponse,
    SimilarPlacesRequest,
    SimilarPlacesResponse,
    S3FaceMosaicRequest,
    S3ReceiptAnalysisRequest,
    S3ReceiptObjectListResponse,
    S3ReceiptReadCheckRequest,
    S3ReceiptReadCheckResponse,
)


RecommendationProcessor = Callable[[dict[str, Any]], dict[str, Any]]
ReceiptProcessor = Callable[[bytes, str], dict[str, Any]]
GptReceiptProcessor = Callable[[bytes, str], dict[str, Any]]
S3ImageLoader = Callable[[str], S3ReceiptObject]
S3ReceiptObjectLister = Callable[[int], S3ReceiptObjectList]
FaceMosaicProcessor = Callable[[bytes, str], FaceMosaicResult]
S3MosaicUploader = Callable[[FaceMosaicResult], str]
logger = logging.getLogger(__name__)

MAX_RECEIPT_IMAGE_BYTES = int(
    os.environ.get("MAX_RECEIPT_IMAGE_BYTES", 10 * 1024 * 1024)
)
OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "kor+eng").strip() or "kor+eng"
MAX_FACE_IMAGE_BYTES = int(
    os.environ.get("MAX_FACE_IMAGE_BYTES", 10 * 1024 * 1024)
)


app = FastAPI(
    title="대전 관광 추천 내부 API",
    description=(
        "Spring 서버에서 전달한 장소 후보를 바탕으로 유사 장소와 다음 장소를 "
        "추천하고, 영수증 구조화와 사진의 얼굴 모자이크를 "
        "처리하는 내부 API입니다."
    ),
    version="1.5.0",
    openapi_tags=[
        {
            "name": "장소 추천",
            "description": "선택 장소와 비슷한 장소 또는 다음 이동 장소를 추천합니다.",
        },
        {
            "name": "영수증 분석",
            "description": "영수증 이미지를 OCR 또는 GPT Vision으로 분석합니다.",
        },
        {
            "name": "이미지 비식별화",
            "description": "사진에서 얼굴을 찾아 모자이크 처리합니다.",
        },
    ],
)

RECOMMENDATION_ERROR_RESPONSES = {
    400: {"description": "추천 요청 데이터 오류"},
    422: {"description": "요청 형식 검증 오류"},
    500: {"description": "추천 처리 서버 오류"},
}
RECEIPT_ERROR_RESPONSES = {
    413: {"description": "이미지 용량 제한 초과"},
    415: {"description": "지원하지 않는 파일 형식"},
    422: {"description": "영수증 판독 실패 또는 요청 형식 검증 오류"},
    500: {"description": "영수증 처리 서버 오류"},
}
GPT_RECEIPT_ERROR_RESPONSES = {
    **RECEIPT_ERROR_RESPONSES,
    502: {"description": "OpenAI API 호출 또는 응답 오류"},
    503: {"description": "OpenAI API 설정 누락"},
}
S3_IMAGE_ERROR_RESPONSES = {
    400: {"description": "S3 객체 키 오류"},
    404: {"description": "S3 이미지를 찾을 수 없음"},
    413: {"description": "이미지 용량 제한 초과"},
    415: {"description": "지원하지 않는 파일 형식"},
    422: {"description": "얼굴 검출 또는 이미지 처리 실패"},
    502: {"description": "S3 읽기 또는 저장 실패"},
    503: {"description": "S3 또는 얼굴 검출 설정 누락"},
}
LOCAL_FACE_ERROR_RESPONSES = {
    413: {"description": "이미지 용량 제한 초과"},
    415: {"description": "지원하지 않는 파일 형식"},
    422: {"description": "얼굴 검출 또는 이미지 처리 실패"},
    503: {"description": "얼굴 검출 설정 누락"},
}


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
    request_id = (
        body.get("request_id")
        or body.get("requestId")
        or body.get("receiptUuid")
    )
    return str(request_id) if request_id is not None else None


def _log_recommendation_result(endpoint: str, response: dict[str, Any]) -> None:
    """추천 결과를 컨테이너 로그에서 바로 확인할 수 있게 남긴다."""
    logger.info(
        "Recommendation result endpoint=%s response=%s",
        endpoint,
        json.dumps(response, ensure_ascii=False, default=str),
    )


def _face_s3_error_code(error_code: str) -> str:
    return {
        "S3_RECEIPT_NOT_CONFIGURED": "S3_IMAGE_NOT_CONFIGURED",
        "INVALID_S3_RECEIPT_KEY": "INVALID_S3_IMAGE_KEY",
        "S3_RECEIPT_NOT_FOUND": "S3_IMAGE_NOT_FOUND",
        "RECEIPT_IMAGE_TOO_LARGE": "FACE_IMAGE_TOO_LARGE",
        "UNSUPPORTED_RECEIPT_MEDIA_TYPE": "UNSUPPORTED_IMAGE_MEDIA_TYPE",
        "S3_RECEIPT_ACCESS_FAILED": "S3_IMAGE_ACCESS_FAILED",
    }.get(error_code, error_code)


@app.exception_handler(RequestValidationError)
def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.warning(
        "Invalid request body endpoint=%s request_id=%s errors=%s",
        request.url.path,
        _request_id_from_body(exc.body),
        json.dumps(exc.errors(), ensure_ascii=False, default=str),
    )
    return _error_response(
        status_code=422,
        code="INVALID_REQUEST_BODY",
        message="요청 본문이 올바르지 않습니다",
        request_id=_request_id_from_body(exc.body),
    )


def get_recommendation_processor() -> RecommendationProcessor:
    return process_spring_recommendation_request


def get_similar_places_processor() -> RecommendationProcessor:
    return process_spring_similar_places_request


def get_next_places_processor() -> RecommendationProcessor:
    return process_spring_next_places_request


def get_receipt_processor() -> ReceiptProcessor:
    return lambda image_bytes, language: analyze_receipt_image_bytes(
        image_bytes,
        language=language,
    )


def get_gpt_receipt_processor() -> GptReceiptProcessor:
    return analyze_receipt_image_with_gpt


def get_s3_receipt_loader() -> S3ImageLoader:
    return lambda s3_key: load_s3_receipt_object(
        s3_key,
        max_bytes=MAX_RECEIPT_IMAGE_BYTES,
    )


def get_s3_receipt_object_lister() -> S3ReceiptObjectLister:
    return lambda max_keys: list_s3_receipt_objects(max_keys=max_keys)


def get_s3_face_image_loader() -> S3ImageLoader:
    return lambda s3_key: load_s3_face_image(
        s3_key,
        max_bytes=MAX_FACE_IMAGE_BYTES,
    )


def get_face_mosaic_processor() -> FaceMosaicProcessor:
    return mosaic_face_image_bytes


def get_s3_mosaic_uploader() -> S3MosaicUploader:
    return lambda result: store_s3_mosaic_image(result)


@app.post(
    "/api/v1/recommendations",
    response_model=RecommendationResponse,
    tags=["장소 추천"],
    summary="통합 장소 추천(기존 API)",
    description=(
        "선택 장소와 비슷한 장소 5개와 다음 이동 장소 5개를 한 번에 계산합니다. "
        "기존 Spring 클라이언트 호환용이며 신규 연동은 분리된 추천 API를 사용합니다."
    ),
    response_description="유사 장소와 다음 장소 통합 추천 결과",
    responses=RECOMMENDATION_ERROR_RESPONSES,
    deprecated=True,
)
def create_recommendations(
    request: RecommendationRequest,
    processor: RecommendationProcessor = Depends(get_recommendation_processor),
) -> dict[str, Any] | JSONResponse:
    try:
        response = processor(request.model_dump(mode="json"))
        _log_recommendation_result("/api/v1/recommendations", response)
        return response
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
    "/api/v1/recommendations/similar-places",
    response_model=SimilarPlacesResponse,
    tags=["장소 추천"],
    summary="선택 장소와 비슷한 장소 추천",
    description=(
        "사용자가 선택한 장소를 기준으로 카테고리, 태그, 설명과 거리가 "
        "비슷한 후보 장소를 최대 5개 추천합니다. 다음 장소 추천은 실행하지 않습니다."
    ),
    response_description="선택 장소와 비슷한 장소(최대 5개)",
    responses=RECOMMENDATION_ERROR_RESPONSES,
)
def create_similar_place_recommendations(
    request: SimilarPlacesRequest,
    processor: RecommendationProcessor = Depends(get_similar_places_processor),
) -> dict[str, Any] | JSONResponse:
    try:
        response = processor(request.to_processor_request())
        _log_recommendation_result(
            "/api/v1/recommendations/similar-places",
            response,
        )
        return response
    except ValueError as exc:
        return _error_response(
            status_code=400,
            code="INVALID_SIMILAR_PLACES_REQUEST",
            message=str(exc),
            request_id=f"similar-{request.selectedPlace.id}",
        )
    except Exception:
        logger.exception(
            "Unexpected similar-place failure selected_place_id=%s",
            request.selectedPlace.id,
        )
        return _error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="비슷한 장소 추천 처리 중 오류가 발생했습니다",
            request_id=f"similar-{request.selectedPlace.id}",
        )


@app.post(
    "/api/v1/recommendations/next-places",
    response_model=NextPlacesResponse,
    tags=["장소 추천"],
    summary="다음 이동 장소 추천",
    description=(
        "현재 장소, 최근 선택 이력, 시간과 날씨를 바탕으로 다음 이동에 적합한 "
        "장소를 최대 5개 추천합니다. 유사 장소 추천은 실행하지 않습니다."
    ),
    response_description="다음 이동 장소(최대 5개)와 추천 로그",
    responses=RECOMMENDATION_ERROR_RESPONSES,
)
def create_next_place_recommendations(
    request: NextPlacesRequest,
    processor: RecommendationProcessor = Depends(get_next_places_processor),
) -> dict[str, Any] | JSONResponse:
    try:
        response = processor(request.to_processor_request())
        _log_recommendation_result(
            "/api/v1/recommendations/next-places",
            response,
        )
        return response
    except ValueError as exc:
        return _error_response(
            status_code=400,
            code="INVALID_NEXT_PLACES_REQUEST",
            message=str(exc),
            request_id=f"next-{request.selectedPlace.id}",
        )
    except Exception:
        logger.exception(
            "Unexpected next-place failure selected_place_id=%s",
            request.selectedPlace.id,
        )
        return _error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="다음 장소 추천 처리 중 오류가 발생했습니다",
            request_id=f"next-{request.selectedPlace.id}",
        )


@app.post(
    "/api/v1/receipts/analyze",
    response_model=ReceiptOcrFieldsResponse,
    tags=["영수증 분석"],
    summary="영수증 OCR 분석",
    description="업로드한 영수증 이미지에서 상호명, 주소, 결제 일시를 추출합니다.",
    response_description="OCR 핵심 추출 결과",
    responses=RECEIPT_ERROR_RESPONSES,
)
def analyze_receipt(
    image: UploadFile = File(..., description="OCR 처리할 영수증 이미지"),
    requestId: str | None = Form(default=None, description="요청 추적 ID"),
    documentId: str | None = Form(default=None, description="영수증 문서 ID"),
    userId: int | None = Form(default=None, description="사용자 ID"),
    processor: ReceiptProcessor = Depends(get_receipt_processor),
) -> dict[str, Any] | JSONResponse:
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

    started_at = perf_counter()
    try:
        analysis = processor(image_bytes, OCR_LANGUAGE)
        result = analysis["result"]
        logger.info(
            "Receipt OCR completed request_id=%s document_id=%s elapsed_ms=%d",
            requestId,
            documentId,
            (perf_counter() - started_at) * 1000,
        )
        return build_ocr_result_fields(result)
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
    response_model=ReceiptOcrFieldsResponse,
    tags=["영수증 분석"],
    summary="GPT-5 Mini 영수증 분석",
    description="업로드한 영수증 이미지에서 GPT Vision으로 상호명, 주소, 결제 일시를 추출합니다.",
    response_description="OCR 핵심 추출 결과",
    responses=GPT_RECEIPT_ERROR_RESPONSES,
)
def analyze_receipt_with_gpt_mini(
    image: UploadFile = File(..., description="GPT로 분석할 영수증 이미지"),
    requestId: str | None = Form(default=None, description="요청 추적 ID"),
    documentId: str | None = Form(default=None, description="영수증 문서 ID"),
    userId: int | None = Form(default=None, description="사용자 ID"),
    processor: GptReceiptProcessor = Depends(get_gpt_receipt_processor),
) -> dict[str, Any] | JSONResponse:
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
        return build_ocr_result_fields(result)
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


def _completed_ocr_receipt_response(
    analysis: dict[str, Any],
    request: S3ReceiptAnalysisRequest,
) -> dict[str, Any]:
    del request
    return build_ocr_result_fields(analysis["result"])


def _completed_gpt_receipt_response(
    analysis: dict[str, Any],
    request: S3ReceiptAnalysisRequest,
) -> dict[str, Any]:
    del request
    return build_ocr_result_fields(analysis["result"])


@app.post(
    "/api/v1/ocr",
    response_model=ReceiptSpringOcrResponse,
    tags=["영수증 분석"],
    summary="S3 영수증 OCR 분석(Spring 연동용)",
    description=(
        "Spring이 receiptUuid와 S3 objectKey를 JSON으로 전달하면, 서버가 "
        "설정된 S3 버킷에서 이미지를 읽어 OCR 분석한 뒤 결과를 HTTP 응답으로 반환합니다."
    ),
    response_description="Spring이 receiptId에 저장할 OCR 핵심 추출 결과",
    responses=S3_IMAGE_ERROR_RESPONSES,
)
def analyze_receipt_from_spring_ocr_request(
    request: ReceiptOcrRequest,
    loader: S3ImageLoader = Depends(get_s3_receipt_loader),
    processor: ReceiptProcessor = Depends(get_receipt_processor),
) -> dict[str, Any] | JSONResponse:
    try:
        source = loader(request.objectKey)
        analysis = processor(source.image_bytes, OCR_LANGUAGE)
        result = analysis["result"]
        response_payload = build_ocr_result_callback_payload(request.receiptUuid, result)
        logger.info(
            "OCR analysis completed receipt_uuid=%s result=%s",
            request.receiptUuid,
            result,
        )
        logger.info(
            "Returning Spring OCR response receipt_uuid=%s payload=%s",
            request.receiptUuid,
            response_payload,
        )
        print(
            f"SPRING_OCR_RESPONSE {response_payload}",
            flush=True,
        )
        return response_payload
    except S3ReceiptError as exc:
        logger.warning(
            "S3 OCR failure receipt_uuid=%s code=%s: %s",
            request.receiptUuid,
            exc.error_code,
            exc,
        )
        return _error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            request.receiptUuid,
        )
    except ReceiptDocumentError as exc:
        logger.warning(
            "Receipt OCR analysis failure receipt_uuid=%s: %s",
            request.receiptUuid,
            exc,
        )
        return _error_response(
            422,
            "RECEIPT_ANALYSIS_FAILED",
            str(exc),
            request.receiptUuid,
        )
    except Exception:
        logger.exception(
            "Unexpected Spring OCR failure receipt_uuid=%s",
            request.receiptUuid,
        )
        return _error_response(
            500,
            "INTERNAL_SERVER_ERROR",
            "S3 영수증 처리 중 오류가 발생했습니다",
            request.receiptUuid,
        )


@app.post(
    "/api/v1/receipts/s3-read-check",
    response_model=S3ReceiptReadCheckResponse,
    tags=["영수증 분석"],
    summary="S3 영수증 객체 읽기 확인",
    description=(
        "OCR을 수행하지 않고, 전달한 objectKey의 영수증 이미지 객체를 "
        "설정된 S3 버킷에서 실제로 읽을 수 있는지 확인합니다. "
        "이미지 원본은 응답에 포함하지 않습니다."
    ),
    response_description="S3에서 읽은 영수증 이미지 객체의 메타데이터",
    responses=S3_IMAGE_ERROR_RESPONSES,
)
def check_s3_receipt_read_access(
    request: S3ReceiptReadCheckRequest,
    loader: S3ImageLoader = Depends(get_s3_receipt_loader),
) -> dict[str, Any] | JSONResponse:
    try:
        source = loader(request.objectKey)
        return {
            "requestId": request.requestId,
            "status": "AVAILABLE",
            "objectKey": source.key,
            "contentType": source.content_type,
            "sizeBytes": len(source.image_bytes),
        }
    except S3ReceiptError as exc:
        logger.warning(
            "S3 receipt read check failure request_id=%s code=%s: %s",
            request.requestId,
            exc.error_code,
            exc,
        )
        return _error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            request.requestId,
        )


@app.get(
    "/api/v1/receipts/s3-objects",
    response_model=S3ReceiptObjectListResponse,
    tags=["영수증 분석"],
    summary="S3 영수증 객체 목록 조회",
    description=(
        "설정된 영수증 S3 경로(기본값: receipts/)에서 객체 메타데이터만 "
        "최대 50개까지 조회합니다. 이미지 원본은 반환하지 않습니다."
    ),
    response_description="S3 영수증 객체 목록",
    responses=S3_IMAGE_ERROR_RESPONSES,
)
def list_s3_receipt_objects_for_check(
    maxKeys: int = Query(
        default=20,
        ge=1,
        le=50,
        description="반환할 최대 객체 수(1~50)",
    ),
    lister: S3ReceiptObjectLister = Depends(get_s3_receipt_object_lister),
) -> dict[str, Any] | JSONResponse:
    try:
        result = lister(maxKeys)
        return {
            "status": "AVAILABLE",
            "prefix": result.prefix,
            "objectCount": len(result.objects),
            "objects": [
                {
                    "objectKey": item.key,
                    "sizeBytes": item.size_bytes,
                    "lastModified": (
                        item.last_modified.isoformat()
                        if item.last_modified is not None
                        else None
                    ),
                }
                for item in result.objects
            ],
        }
    except S3ReceiptError as exc:
        logger.warning("S3 receipt list failure code=%s: %s", exc.error_code, exc)
        return _error_response(exc.status_code, exc.error_code, str(exc), None)


@app.post(
    "/api/v1/receipts/analyze-from-s3",
    response_model=ReceiptOcrFieldsResponse,
    tags=["영수증 분석"],
    summary="S3 영수증 OCR 분석",
    description=(
        "Spring이 전달한 S3 객체 키로 영수증 이미지를 읽어 "
        "Tesseract OCR로 분석합니다."
    ),
    response_description="S3 이미지에서 추출한 OCR 핵심 결과",
    responses=S3_IMAGE_ERROR_RESPONSES,
)
def analyze_receipt_from_s3(
    request: S3ReceiptAnalysisRequest,
    loader: S3ImageLoader = Depends(get_s3_receipt_loader),
    processor: ReceiptProcessor = Depends(get_receipt_processor),
) -> dict[str, Any] | JSONResponse:
    try:
        source = loader(request.s3Key)
        analysis = processor(source.image_bytes, OCR_LANGUAGE)
        return _completed_ocr_receipt_response(analysis, request)
    except S3ReceiptError as exc:
        return _error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            request.requestId,
        )
    except ReceiptDocumentError as exc:
        return _error_response(
            422,
            "RECEIPT_ANALYSIS_FAILED",
            str(exc),
            request.requestId,
        )
    except Exception:
        logger.exception("Unexpected S3 receipt failure request_id=%s", request.requestId)
        return _error_response(
            500,
            "INTERNAL_SERVER_ERROR",
            "S3 영수증 처리 중 오류가 발생했습니다",
            request.requestId,
        )


@app.post(
    "/api/v1/receipts/analyze-gpt-mini-from-s3",
    response_model=ReceiptOcrFieldsResponse,
    tags=["영수증 분석"],
    summary="S3 영수증 GPT-5 Mini 분석",
    description=(
        "Spring이 전달한 S3 객체 키로 영수증 이미지를 읽어 "
        "GPT-5 Mini Vision으로 분석합니다."
    ),
    response_description="S3 이미지에서 추출한 OCR 핵심 결과",
    responses=S3_IMAGE_ERROR_RESPONSES,
)
def analyze_receipt_with_gpt_mini_from_s3(
    request: S3ReceiptAnalysisRequest,
    loader: S3ImageLoader = Depends(get_s3_receipt_loader),
    processor: GptReceiptProcessor = Depends(get_gpt_receipt_processor),
) -> dict[str, Any] | JSONResponse:
    try:
        source = loader(request.s3Key)
        analysis = processor(source.image_bytes, source.content_type)
        return _completed_gpt_receipt_response(analysis, request)
    except S3ReceiptError as exc:
        return _error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            request.requestId,
        )
    except ReceiptVisionConfigurationError as exc:
        return _error_response(
            503,
            "RECEIPT_VISION_NOT_CONFIGURED",
            str(exc),
            request.requestId,
        )
    except ReceiptDocumentError as exc:
        return _error_response(
            422,
            "RECEIPT_ANALYSIS_FAILED",
            str(exc),
            request.requestId,
        )
    except ReceiptVisionUpstreamError as exc:
        return _error_response(
            502,
            "RECEIPT_VISION_UPSTREAM_ERROR",
            str(exc),
            request.requestId,
        )
    except Exception:
        logger.exception(
            "Unexpected GPT S3 receipt failure request_id=%s",
            request.requestId,
        )
        return _error_response(
            500,
            "INTERNAL_SERVER_ERROR",
            "S3 GPT 영수증 처리 중 오류가 발생했습니다",
            request.requestId,
        )


@app.post(
    "/api/v1/images/face-mosaic",
    response_model=FaceMosaicResponse,
    tags=["이미지 비식별화"],
    summary="S3 이미지 얼굴 모자이크",
    description=(
        "Spring이 전달한 S3 객체 키로 원본을 읽고, YuNet으로 얼굴을 "
        "검출해 모자이크한 JPEG을 S3의 새 객체로 저장합니다."
    ),
    response_description="모자이크 결과의 S3 객체 키와 검출한 얼굴 수",
    responses=S3_IMAGE_ERROR_RESPONSES,
)
def mosaic_s3_image_faces(
    request: S3FaceMosaicRequest,
    loader: S3ImageLoader = Depends(get_s3_face_image_loader),
    processor: FaceMosaicProcessor = Depends(get_face_mosaic_processor),
    uploader: S3MosaicUploader = Depends(get_s3_mosaic_uploader),
) -> dict[str, Any] | JSONResponse:
    try:
        source = loader(request.s3Key)
        result = processor(source.image_bytes, source.content_type)
        output_key = uploader(result)
        return {
            "requestId": request.requestId,
            "userId": request.userId,
            "status": "COMPLETED",
            "sourceS3Key": source.key,
            "outputS3Key": output_key,
            "contentType": result.content_type,
            "faceCount": result.face_count,
            "width": result.width,
            "height": result.height,
            "processedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    except S3ReceiptError as exc:
        return _error_response(
            exc.status_code,
            _face_s3_error_code(exc.error_code),
            str(exc),
            request.requestId,
        )
    except S3MosaicUploadError as exc:
        return _error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            request.requestId,
        )
    except FaceMosaicError as exc:
        return _error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            request.requestId,
        )
    except Exception:
        logger.exception("Unexpected face mosaic failure request_id=%s", request.requestId)
        return _error_response(
            500,
            "INTERNAL_SERVER_ERROR",
            "얼굴 모자이크 처리 중 오류가 발생했습니다",
            request.requestId,
        )


@app.post(
    "/api/v1/images/face-mosaic-local",
    response_class=Response,
    tags=["이미지 비식별화"],
    summary="로컬 이미지 얼굴 모자이크",
    description=(
        "로컬에서 multipart/form-data로 올린 이미지의 얼굴을 모자이크하고 "
        "JPEG 파일을 바로 반환합니다. S3는 사용하지 않습니다."
    ),
    response_description="얼굴이 모자이크된 JPEG 이미지",
    responses=LOCAL_FACE_ERROR_RESPONSES,
)
def mosaic_local_image_faces(
    image: UploadFile = File(..., description="얼굴을 모자이크할 이미지"),
    processor: FaceMosaicProcessor = Depends(get_face_mosaic_processor),
) -> Response:
    if image.content_type and not (
        image.content_type.startswith("image/")
        or image.content_type == "application/octet-stream"
    ):
        return _error_response(
            415,
            "UNSUPPORTED_IMAGE_MEDIA_TYPE",
            "이미지 파일만 업로드할 수 있습니다",
            None,
        )

    image_bytes = image.file.read(MAX_FACE_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_FACE_IMAGE_BYTES:
        return _error_response(
            413,
            "FACE_IMAGE_TOO_LARGE",
            f"이미지는 {MAX_FACE_IMAGE_BYTES // (1024 * 1024)}MB 이하여야 합니다",
            None,
        )

    try:
        result = processor(
            image_bytes,
            image.content_type or "application/octet-stream",
        )
        return Response(
            content=result.image_bytes,
            media_type=result.content_type,
            headers={
                "Content-Disposition": 'attachment; filename="face-mosaic.jpg"',
                "X-Face-Count": str(result.face_count),
                "X-Image-Width": str(result.width),
                "X-Image-Height": str(result.height),
            },
        )
    except FaceMosaicError as exc:
        return _error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            None,
        )
    except Exception:
        logger.exception("Unexpected local face mosaic failure")
        return _error_response(
            500,
            "INTERNAL_SERVER_ERROR",
            "얼굴 모자이크 처리 중 오류가 발생했습니다",
            None,
        )
