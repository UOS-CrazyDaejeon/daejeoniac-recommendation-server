from collections.abc import Callable
import logging
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from recommend_llm import process_spring_recommendation_request
from recommendation_api.schemas import RecommendationRequest, RecommendationResponse


RecommendationProcessor = Callable[[dict[str, Any]], dict[str, Any]]
logger = logging.getLogger(__name__)


app = FastAPI(title="Daejeon Recommendation API", version="1.0.0")


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
