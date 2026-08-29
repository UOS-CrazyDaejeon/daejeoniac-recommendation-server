from datetime import datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


def _unwrap_spring_page_content(value: Any) -> Any:
    """Spring Page 응답은 메타데이터를 무시하고 content 배열만 사용한다."""
    if isinstance(value, dict) and "content" in value:
        content = value["content"]
        if not isinstance(content, list):
            raise ValueError("Spring Page의 content는 배열이어야 합니다")
        return content
    return value


class Place(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int = Field(
        validation_alias=AliasChoices("place_id", "placeId", "id")
    )
    name: str = Field(
        validation_alias=AliasChoices("place_name", "placeName", "name")
    )
    latitude: float
    longitude: float
    congestion: float = 50.0
    monthly_visitors: int = 0
    selected_count: int = 0
    category: str = "unknown"
    description: str = ""
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_spring_place(cls, value: Any) -> Any:
        """Spring Place DTO를 추천 엔진이 사용하는 장소 형태로 보정한다."""
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if "id" not in normalized:
            raw_id = normalized.get("place_id", normalized.get("placeId"))
            if raw_id is not None:
                normalized["id"] = int(raw_id)
        if "name" not in normalized:
            normalized["name"] = normalized.get(
                "place_name", normalized.get("placeName")
            )
        if "congestion" not in normalized:
            raw_congestion = normalized.get(
                "congestion_rate", normalized.get("congestionRate")
            )
            if raw_congestion is not None:
                normalized["congestion"] = raw_congestion
        if "monthly_visitors" not in normalized:
            raw_visitor_count = normalized.get(
                "visitor_count", normalized.get("visitorCount")
            )
            if raw_visitor_count is not None:
                normalized["monthly_visitors"] = raw_visitor_count
        if "selected_count" not in normalized and normalized.get("selectedCount") is not None:
            normalized["selected_count"] = normalized["selectedCount"]
        if "description" not in normalized:
            normalized["description"] = normalized.get("placeDescription") or ""

        for camel_name, snake_name in (
            ("categoryLarge", "category_large"),
            ("categoryMedium", "category_medium"),
            ("categorySmall", "category_small"),
            ("visitedAt", "visited_at"),
        ):
            if camel_name not in normalized and normalized.get(snake_name) is not None:
                normalized[camel_name] = normalized[snake_name]

        if "category" not in normalized:
            normalized["category"] = (
                normalized.get("categorySmall")
                or normalized.get("category_small")
                or normalized.get("categoryMedium")
                or normalized.get("category_medium")
                or normalized.get("categoryLarge")
                or normalized.get("category_large")
                or "unknown"
            )

        if "tags" not in normalized:
            raw_tag = normalized.get("tag")
            if isinstance(raw_tag, str):
                normalized["tags"] = [
                    tag.strip() for tag in raw_tag.split(",") if tag.strip()
                ]
            elif isinstance(raw_tag, list):
                normalized["tags"] = [str(tag).strip() for tag in raw_tag if str(tag).strip()]
            else:
                normalized["tags"] = list(
                    dict.fromkeys(
                        str(item)
                        for item in (
                            normalized.get("categoryLarge")
                            or normalized.get("category_large"),
                            normalized.get("categoryMedium")
                            or normalized.get("category_medium"),
                            normalized.get("categorySmall")
                            or normalized.get("category_small"),
                            normalized.get("gu"),
                            normalized.get("dong"),
                        )
                        if item
                    )
                )
        return normalized


class RecommendationContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    current_time: str
    weather: str | None = None
    user_preferences: str = ""
    radius_m: float = Field(gt=0, le=1000)
    similar_top_k: Literal[5]
    next_top_k: Literal[5]


RecentPlaces = Annotated[list[Place], Field(max_length=4)]
# Spring Page 객체({"content": [...]})와 페이징 제거 후의 배열을 모두 받는다.
Candidates = Annotated[
    list[Place],
    BeforeValidator(_unwrap_spring_page_content),
    Field(min_length=1),
]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str
    current_place: Place
    recent_places: RecentPlaces
    visited_place_ids: list[int]
    candidates: Candidates
    context: RecommendationContext


class RecommendationResponse(BaseModel):
    request_id: str
    session_id: str
    generated_at: str
    current_place_id: int
    similar_places: list[dict[str, Any]]
    next_places: list[dict[str, Any]]
    recommendation_log: dict[str, Any]


class SimilarPlacesContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    radius_m: float = Field(gt=0, le=1000)
    top_k: Literal[5] = 5


class SimilarPlacesRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "selected_place": {
                        "place_id": 2,
                        "place_name": "중앙로 지하상가",
                        "description": "대전 원도심 쇼핑 공간",
                        "tag": "쇼핑,원도심,대전",
                        "place_address": "대전광역시 중구 중앙로",
                        "latitude": 36.3281,
                        "longitude": 127.4265,
                        "gu": "중구",
                        "dong": "은행동",
                        "category_large": "쇼핑",
                        "category_medium": "상가",
                        "category_small": "지하상가",
                        "congestion_rate": 61.0,
                        "visitor_count": 980,
                    },
                    "nearby_places": [
                        {
                            "place_id": 3,
                            "place_name": "으능정이문화의거리",
                            "description": "대전 원도심 대표 문화 거리",
                            "tag": "관광지,중구,문화거리",
                            "place_address": "대전광역시 중구 중앙로164번길",
                            "latitude": 36.3277,
                            "longitude": 127.4281,
                            "gu": "중구",
                            "dong": "은행동",
                            "category_large": "관광지",
                            "category_medium": "거리",
                            "category_small": "문화거리",
                            "congestion_rate": 55.0,
                            "visitor_count": 720,
                        }
                    ],
                    "visited_place_ids": [],
                    "radius_m": 1000,
                }
            ]
        },
    )

    selectedPlace: Place = Field(
        validation_alias=AliasChoices("selected_place", "selectedPlace"),
        description="사용자가 선택한 기준 장소",
    )
    nearbyPlaces: Candidates = Field(
        validation_alias=AliasChoices("nearby_places", "nearbyPlaces", "candidates"),
        description="Spring이 조회한 인근 장소 목록 또는 Page 객체",
    )
    visitedPlaceIds: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("visited_place_ids", "visitedPlaceIds"),
    )
    radiusM: float = Field(
        default=1000,
        gt=0,
        le=1000,
        validation_alias=AliasChoices("radius_m", "radiusM"),
        description="후보 반경(m)",
    )

    @model_validator(mode="before")
    @classmethod
    def adapt_legacy_request(cls, value: Any) -> Any:
        """기존 snake_case 계약도 새 Spring DTO 계약으로 변환한다."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        # 이전 호출 클라이언트가 보내던 추적용 값은 추천 계약에서 사용하지 않는다.
        normalized.pop("request_id", None)
        normalized.pop("requestId", None)
        normalized.pop("session_id", None)
        normalized.pop("sessionId", None)
        legacy_context = normalized.pop("context", None)
        if isinstance(legacy_context, dict) and "radiusM" not in normalized:
            normalized["radiusM"] = legacy_context.get("radius_m", 1000)
        return normalized

    def to_processor_request(self) -> dict[str, Any]:
        request_id = f"similar-{self.selectedPlace.id}"
        session_id = request_id
        return {
            "request_id": request_id,
            "session_id": session_id,
            "selected_place": self.selectedPlace.model_dump(mode="json"),
            "visited_place_ids": self.visitedPlaceIds,
            "candidates": [place.model_dump(mode="json") for place in self.nearbyPlaces],
            "context": {"radius_m": self.radiusM, "top_k": 5},
        }


class SimilarPlacesResponse(BaseModel):
    generated_at: str
    selected_place_id: int
    similar_places: list[dict[str, Any]]


class NextPlacesContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_time: str
    weather: str | None = None
    user_preferences: str = ""
    radius_m: float = Field(gt=0, le=1000)
    top_k: Literal[5] = 5


class NextPlacesRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "selected_place": {
                        "place_id": 2,
                        "place_name": "중앙로 지하상가",
                        "description": "대전 원도심 쇼핑 공간",
                        "tag": "쇼핑,원도심,대전",
                        "latitude": 36.3281,
                        "longitude": 127.4265,
                        "place_address": "대전광역시 중구 중앙로",
                        "gu": "중구",
                        "dong": "은행동",
                        "category_large": "쇼핑",
                        "category_medium": "상가",
                        "category_small": "지하상가",
                        "congestion_rate": 61.0,
                        "visitor_count": 980,
                    },
                    "nearby_places": [
                        {
                            "place_id": 3,
                            "place_name": "으능정이문화의거리",
                            "description": "대전 원도심 대표 문화 거리",
                            "tag": "관광지,중구,문화거리",
                            "latitude": 36.3277,
                            "longitude": 127.4281,
                            "place_address": "대전광역시 중구 중앙로164번길",
                            "gu": "중구",
                            "dong": "은행동",
                            "category_large": "관광지",
                            "category_medium": "거리",
                            "category_small": "문화거리",
                            "congestion_rate": 55.0,
                            "visitor_count": 720,
                        }
                    ],
                    "visited_places": [],
                    "visited_place_ids": [],
                    "radius_m": 1000,
                }
            ]
        },
    )

    selectedPlace: Place = Field(
        validation_alias=AliasChoices(
            "selected_place", "selectedPlace", "currentPlace", "current_place"
        ),
        description="사용자가 현재 선택한 장소",
    )
    nearbyPlaces: Candidates = Field(
        validation_alias=AliasChoices("nearby_places", "nearbyPlaces", "candidates"),
        description="Spring이 조회한 인근 장소 목록 또는 Page 객체",
    )
    visitedPlaces: list[Place] = Field(
        default_factory=list,
        validation_alias=AliasChoices("visited_places", "visitedPlaces", "recent_places"),
        description="사용자가 방문한 장소 목록",
    )
    visitedPlaceIds: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("visited_place_ids", "visitedPlaceIds"),
    )
    currentTime: str | None = Field(
        default=None,
        validation_alias=AliasChoices("current_time", "currentTime"),
        description="추천 기준 시각(생략 시 현재 시각)",
    )
    weather: str | None = None
    userPreferences: str = Field(
        default="",
        validation_alias=AliasChoices("user_preferences", "userPreferences"),
    )
    radiusM: float = Field(
        default=1000,
        gt=0,
        le=1000,
        validation_alias=AliasChoices("radius_m", "radiusM"),
        description="후보 반경(m)",
    )

    @model_validator(mode="before")
    @classmethod
    def adapt_legacy_request(cls, value: Any) -> Any:
        """기존 snake_case 계약도 새 Spring DTO 계약으로 변환한다."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        # 이전 호출 클라이언트가 보내던 추적용 값은 추천 계약에서 사용하지 않는다.
        normalized.pop("request_id", None)
        normalized.pop("requestId", None)
        normalized.pop("session_id", None)
        normalized.pop("sessionId", None)
        legacy_context = normalized.pop("context", None)
        if isinstance(legacy_context, dict):
            normalized.setdefault("currentTime", legacy_context.get("current_time"))
            normalized.setdefault("weather", legacy_context.get("weather"))
            normalized.setdefault("userPreferences", legacy_context.get("user_preferences", ""))
            normalized.setdefault("radiusM", legacy_context.get("radius_m", 1000))
        return normalized

    def to_processor_request(self) -> dict[str, Any]:
        request_id = f"next-{self.selectedPlace.id}"
        session_id = request_id
        visited_rows = [place.model_dump(mode="json") for place in self.visitedPlaces]
        visited_rows.sort(key=lambda place: str(place.get("visitedAt") or ""))
        visited_ids = list(
            dict.fromkeys(
                [*self.visitedPlaceIds, *(place["id"] for place in visited_rows)]
            )
        )
        return {
            "request_id": request_id,
            "session_id": session_id,
            "current_place": self.selectedPlace.model_dump(mode="json"),
            "recent_places": visited_rows[-4:],
            "visited_place_ids": visited_ids,
            "candidates": [place.model_dump(mode="json") for place in self.nearbyPlaces],
            "context": {
                "current_time": self.currentTime
                or datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "weather": self.weather,
                "user_preferences": self.userPreferences,
                "radius_m": self.radiusM,
                "top_k": 5,
            },
        }


class NextPlacesResponse(BaseModel):
    generated_at: str
    current_place_id: int
    visited_place_ids: list[int]
    next_places: list[dict[str, Any]]


class ReceiptItem(BaseModel):
    name: str
    quantity: int
    unitPrice: int
    amount: int


class ReceiptResult(BaseModel):
    merchantName: str | None = None
    businessNumber: str | None = None
    transactionDate: str | None = None
    transactionTime: str | None = None
    address: str | None = None
    approvalNumber: str | None = None
    supplyAmount: int | None = None
    vat: int | None = None
    totalAmount: int
    paymentMethod: str | None = None
    items: list[ReceiptItem] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class ReceiptOcrFieldsResponse(BaseModel):
    """영수증 분석 API가 공통으로 반환하는 핵심 OCR 결과."""

    ocrStatus: str = Field(description="OCR 처리 상태", examples=["SUCCESS"])
    ocrPlaceName: str | None = Field(default=None, description="OCR로 추출한 상호명")
    ocrPlaceAddress: str | None = Field(default=None, description="OCR로 추출한 주소")
    ocrPaidAt: str | None = Field(
        default=None,
        description="OCR로 추출한 결제 일시(ISO-8601)",
        examples=["2026-08-26T07:29:51"],
    )


class ReceiptSpringOcrResponse(ReceiptOcrFieldsResponse):
    """Spring이 receiptId에 대응해 저장할 S3 영수증 OCR 응답."""

    receiptUuid: str = Field(description="Spring이 전달한 영수증 UUID")


class S3ReceiptAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestId: str | None = Field(default=None, description="요청 추적 ID")
    documentId: str | None = Field(default=None, description="영수증 문서 ID")
    userId: int | None = Field(default=None, description="사용자 ID")
    s3Key: str = Field(
        min_length=1,
        max_length=1024,
        description="설정된 S3 버킷 안의 영수증 이미지 객체 키",
        examples=["receipts/2026/08/receipt-001.heic"],
    )


class S3ReceiptReadCheckRequest(BaseModel):
    """OCR 수행 없이 S3 영수증 객체를 읽을 수 있는지 확인하는 요청."""

    model_config = ConfigDict(extra="forbid")

    requestId: str | None = Field(default=None, description="요청 추적 ID")
    objectKey: str = Field(
        min_length=1,
        max_length=1024,
        description="읽기 여부를 확인할 S3 영수증 이미지 객체 키",
        examples=["receipts/2026/08/receipt-001.heic"],
    )


class S3ReceiptReadCheckResponse(BaseModel):
    requestId: str | None = None
    status: Literal["AVAILABLE"] = "AVAILABLE"
    objectKey: str
    contentType: str
    sizeBytes: int = Field(gt=0)


class S3ReceiptObjectMetadata(BaseModel):
    objectKey: str
    sizeBytes: int = Field(ge=0)
    lastModified: str | None = None


class S3ReceiptObjectListResponse(BaseModel):
    status: Literal["AVAILABLE"] = "AVAILABLE"
    prefix: str
    objectCount: int = Field(ge=0, le=50)
    objects: list[S3ReceiptObjectMetadata] = Field(default_factory=list)


class ReceiptOcrRequest(BaseModel):
    """Spring이 영수증 식별자와 S3 객체 키만 전달할 때 사용하는 요청."""

    model_config = ConfigDict(extra="forbid")

    receiptUuid: str = Field(
        min_length=1,
        max_length=128,
        description="Spring에서 발급한 영수증 식별 UUID",
        examples=["2d6ae292-3e3b-4c95-a102-779562ee12bc"],
    )
    objectKey: str = Field(
        min_length=1,
        max_length=1024,
        description="설정된 S3 버킷 안의 영수증 이미지 객체 키",
        examples=["receipts/2026/08/receipt-001.heic"],
    )


class S3FaceMosaicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestId: str | None = Field(default=None, description="요청 추적 ID")
    userId: int | None = Field(default=None, description="사용자 ID")
    s3Key: str = Field(
        min_length=1,
        max_length=1024,
        description="설정된 S3 버킷 안의 원본 이미지 객체 키",
        examples=["uploads/2026/08/photo-001.heic"],
    )


class FaceMosaicResponse(BaseModel):
    requestId: str | None = None
    userId: int | None = None
    status: Literal["COMPLETED"] = "COMPLETED"
    sourceS3Key: str
    outputS3Key: str
    contentType: Literal["image/jpeg"] = "image/jpeg"
    faceCount: int = Field(ge=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    processedAt: str


class ReceiptVisionUsage(BaseModel):
    inputTokens: int = Field(ge=0)
    outputTokens: int = Field(ge=0)
    totalTokens: int = Field(ge=0)
