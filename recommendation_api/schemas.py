from typing import Annotated, Any, Literal

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

    id: str = Field(validation_alias=AliasChoices("id", "placeId"))
    name: str = Field(validation_alias=AliasChoices("name", "placeName"))
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
        if "id" not in normalized and normalized.get("placeId") is not None:
            normalized["id"] = str(normalized["placeId"])
        if "description" not in normalized:
            normalized["description"] = normalized.get("placeDescription") or ""

        if "category" not in normalized:
            normalized["category"] = (
                normalized.get("categorySmall")
                or normalized.get("categoryMedium")
                or normalized.get("categoryLarge")
                or "unknown"
            )

        if "tags" not in normalized:
            normalized["tags"] = list(
                dict.fromkeys(
                    str(item)
                    for item in (
                        normalized.get("categoryLarge"),
                        normalized.get("categoryMedium"),
                        normalized.get("categorySmall"),
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
    visited_place_ids: list[str]
    candidates: Candidates
    context: RecommendationContext


class RecommendationResponse(BaseModel):
    request_id: str
    session_id: str
    generated_at: str
    current_place_id: str
    similar_places: list[dict[str, Any]]
    next_places: list[dict[str, Any]]
    recommendation_log: dict[str, Any]


class SimilarPlacesContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    radius_m: float = Field(gt=0, le=1000)
    top_k: Literal[5] = 5


class SimilarPlacesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str
    selected_place: Place
    visited_place_ids: list[str] = Field(default_factory=list)
    candidates: Candidates
    context: SimilarPlacesContext


class SimilarPlacesResponse(BaseModel):
    request_id: str
    session_id: str
    generated_at: str
    selected_place_id: str
    similar_places: list[dict[str, Any]]


class NextPlacesContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_time: str
    weather: str | None = None
    user_preferences: str = ""
    radius_m: float = Field(gt=0, le=1000)
    top_k: Literal[5] = 5


class NextPlacesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str
    current_place: Place
    recent_places: RecentPlaces
    visited_place_ids: list[str] = Field(default_factory=list)
    candidates: Candidates
    context: NextPlacesContext


class NextPlacesResponse(BaseModel):
    request_id: str
    session_id: str
    generated_at: str
    current_place_id: str
    next_places: list[dict[str, Any]]
    recommendation_log: dict[str, Any]


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
