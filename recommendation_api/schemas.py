from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Place(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    latitude: float
    longitude: float
    congestion: float = 50.0
    monthly_visitors: int = 0
    selected_count: int = 0
    category: str = "unknown"
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class RecommendationContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    current_time: str
    weather: str | None = None
    user_preferences: str = ""
    radius_m: float = Field(gt=0, le=1000)
    similar_top_k: Literal[5]
    next_top_k: Literal[5]


RecentPlaces = Annotated[list[Place], Field(max_length=4)]
Candidates = Annotated[list[Place], Field(min_length=10, max_length=10)]


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


class ReceiptAnalysisResponse(BaseModel):
    requestId: str | None = None
    documentId: str | None = None
    userId: int | None = None
    documentType: Literal["RECEIPT"] = "RECEIPT"
    status: Literal["COMPLETED"] = "COMPLETED"
    result: ReceiptResult
    warnings: list[str] = Field(default_factory=list)
    processedAt: str
    rawOcrCharCount: int = Field(ge=0)


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


class GptReceiptAnalysisResponse(BaseModel):
    requestId: str | None = None
    documentId: str | None = None
    userId: int | None = None
    documentType: Literal["RECEIPT"] = "RECEIPT"
    status: Literal["COMPLETED"] = "COMPLETED"
    model: str
    result: ReceiptResult
    warnings: list[str] = Field(default_factory=list)
    processedAt: str
    processingTimeMs: int = Field(ge=0)
    usage: ReceiptVisionUsage | None = None
