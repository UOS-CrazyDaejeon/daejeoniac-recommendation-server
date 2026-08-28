# -*- coding: utf-8 -*-
"""
대전 숨은 명소 추천 시스템 - 코드 점수 + PPR + LLM 전이 판단

[전체 구조]
1. 사용자 현재 위치 반경 1km 이내 장소를 후보로 조회한다.
2. 코드가 혼잡도, 월간 방문자 수, 앱 선택 횟수, 거리 기반 기본 점수를 계산한다.
3. 최근 선택 장소를 personalization으로 사용해 거리 그래프 PPR을 계산한다.
4. 기본 점수가 높은 후보만 LLM에 전달한다.
5. LLM은 숫자 계산 대신 "최근 방문 흐름에서 다음 장소로 자연스러운가"만 0~1로 판단한다.
6. 코드가 기본 점수 + 정규화 PPR + LLM 전이 점수를 합쳐 최종 Top-K를 결정한다.
7. LLM이 실패하거나 API 키가 없으면 규칙 기반 전이 scorer로 자동 대체한다.
8. 추천 노출과 실제 선택 여부를 기록해 향후 랭킹 학습 데이터로 사용한다.

[LLM 사용]
- pip install openai networkx
- OPENAI_API_KEY 환경변수 설정
- API 키가 있으면 자동 활성화, 끄려면 USE_LLM=0
- 기본 모델: gpt-5-nano (선택: OPENAI_MODEL로 변경)

LLM은 최종 점수를 직접 계산하지 않는다. 코드가 만든 후보와 문맥을 보고 후보별
transition_score와 간단한 추천 이유만 반환한다.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, Sequence

import networkx as nx


LOGGER = logging.getLogger(__name__)

MONTHLY_VISITOR_REFERENCE = 100_000
SELECTED_COUNT_REFERENCE = 1_000
DEFAULT_MAX_CONGESTION = 70.0
DEFAULT_MAX_MONTHLY_VISITORS = 50_000
MAX_CANDIDATE_RADIUS_M = 1_000
DEFAULT_LLM_CANDIDATE_LIMIT = 10
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DUMMY_DATA_PATH = Path(__file__).with_name("dummy.json")
KAKAO_DUMMY_DATA_PATH = Path(__file__).with_name("dummy_kakao.json")
SCENARIO_DATA_PATH = Path(__file__).with_name("scenario_requests.json")
RESULT_FILE_SUFFIX = "_result.json"
SIMILAR_TOP_K = 5
NEXT_TOP_K = 5

KAKAO_CATEGORY_GROUPS = {
    "MT1": "대형마트",
    "CS2": "편의점",
    "PS3": "어린이집, 유치원",
    "SC4": "학교",
    "AC5": "학원",
    "PK6": "주차장",
    "OL7": "주유소, 충전소",
    "SW8": "지하철역",
    "BK9": "은행",
    "CT1": "문화시설",
    "AG2": "중개업소",
    "PO3": "공공기관",
    "AT4": "관광명소",
    "AD5": "숙박",
    "FD6": "음식점",
    "CE7": "카페",
    "HP8": "병원",
    "PM9": "약국",
}

# 외부 요청에 코드가 없을 때만 사용하는 내부 category -> Kakao group 추정값.
INTERNAL_CATEGORY_TO_KAKAO = {
    "restaurant": "FD6",
    "bakery": "FD6",
    "cafe": "CE7",
    "bookstore": "CT1",
    "museum": "CT1",
    "gallery": "CT1",
    "history": "CT1",
    "nature": "AT4",
    "walk": "AT4",
    "view": "AT4",
    "river": "AT4",
    "hot_spring": "AT4",
}

DEFAULT_BASE_WEIGHTS = {
    "congestion": 0.30,
    "monthly_visitors": 0.20,
    "selected_count": 0.10,
    "distance": 0.15,
    "tag_similarity": 0.25,
}

DEFAULT_HYBRID_WEIGHTS = {
    "base": 0.65,
    "ppr": 0.10,
    "transition": 0.25,
}


# =========================================================
# 1. 공통 유틸리티와 그래프 생성
# =========================================================


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """숫자를 지정 범위로 제한한다."""
    return max(minimum, min(float(value), maximum))


def normalize_tags(tags: Sequence[Any]) -> list[str]:
    """Normalize tags for stable comparisons and remove duplicates."""
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = str(tag).strip().lower()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def build_tag_vocabulary(places: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Build a stable sparse-vector index from tags in the supplied places."""
    tags = {
        tag
        for place in places
        for tag in normalize_tags(place.get("tags", []))
    }
    return {tag: index for index, tag in enumerate(sorted(tags))}


def build_tag_vector(tags: Sequence[Any], vocabulary: dict[str, int]) -> list[float]:
    """Convert tags to a multi-hot vector."""
    vector = [0.0] * len(vocabulary)
    for tag in normalize_tags(tags):
        index = vocabulary.get(tag)
        if index is not None:
            vector[index] = 1.0
    return vector


def build_user_tag_vector(
    recent_places: Sequence[dict[str, Any]],
    vocabulary: dict[str, int],
    decay_rate: float = 0.5,
) -> list[float]:
    """Combine up to four recent places into a recency-weighted tag vector."""
    if not 0 < decay_rate <= 1:
        raise ValueError("decay_rate must be in (0, 1]")

    places = list(recent_places)[-4:]
    vector = [0.0] * len(vocabulary)
    count = len(places)
    for index, place in enumerate(places):
        weight = decay_rate ** (count - 1 - index)
        place_vector = build_tag_vector(place.get("tags", []), vocabulary)
        for vector_index, value in enumerate(place_vector):
            vector[vector_index] += weight * value
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for non-negative tag vectors in the 0..1 range."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right)
    )
    return clamp(dot_product / (left_norm * right_norm))


def compute_pairwise_tag_cosine_scores(
    current_place: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """현재 장소와 후보의 multi-hot 태그 임베딩 코사인 유사도를 계산한다."""
    places = [current_place, *candidates]
    vocabulary = build_tag_vocabulary(places)
    current_vector = build_tag_vector(current_place.get("tags", []), vocabulary)
    return {
        str(candidate["id"]): cosine_similarity(
            current_vector,
            build_tag_vector(candidate.get("tags", []), vocabulary),
        )
        for candidate in candidates
    }


def load_dummy_places(path: Path = DUMMY_DATA_PATH) -> list[dict[str, Any]]:
    """JSON 파일에서 데모 장소 데이터를 읽는다."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"dummy.json을 읽을 수 없습니다: {path}") from exc

    if not isinstance(payload, list):
        raise ValueError("dummy.json의 최상위 값은 배열이어야 합니다")
    return payload


def load_scenario_requests(
    path: Path = SCENARIO_DATA_PATH,
) -> list[dict[str, Any]]:
    """Spring 요청 형식의 데모 시나리오를 읽고 기본 계약을 검증한다."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"scenario_requests.json을 읽을 수 없습니다: {path}") from exc

    if isinstance(payload, dict):
        candidate_pool = payload.get("candidate_pool")
        requests = payload.get("requests")
        if not isinstance(candidate_pool, list) or not isinstance(requests, list):
            raise ValueError(
                "scenario_requests.json에는 candidate_pool과 requests 배열이 필요합니다"
            )
        payload = [
            {**request, "candidates": [dict(candidate) for candidate in candidate_pool]}
            for request in requests
        ]
    if not isinstance(payload, list):
        raise ValueError("scenario_requests.json 형식이 올바르지 않습니다")
    for request in payload:
        _validate_spring_request(request)
    return payload


def build_result_path(at: datetime | None = None) -> Path:
    """실행 시각을 포함한 결과 파일 경로를 만든다."""
    timestamp = (at or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    return Path(__file__).with_name(f"{timestamp}{RESULT_FILE_SUFFIX}")


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 좌표 사이의 실제 거리(미터)를 계산한다."""
    radius = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def build_graph(
    places: list[dict[str, Any]],
    max_edge_distance_m: float = 1_500,
    edge_sigma_m: float | None = None,
) -> nx.Graph:
    """
    장소를 노드로 등록하고, 일정 거리 이내 장소끼리 연결한다.

    그래프 엣지는 현재 거리 관계만 표현한다. 혼잡도와 인기도는 장소 자체의
    품질이므로 최종 랭킹의 노드 feature로 사용한다. 실제 연속 이동 로그가
    쌓이면 공동선택/전이 확률을 weight에 추가할 수 있다.
    """
    if max_edge_distance_m <= 0:
        raise ValueError("max_edge_distance_m must be positive")

    sigma = edge_sigma_m or max_edge_distance_m / 2
    if sigma <= 0:
        raise ValueError("edge_sigma_m must be positive")

    graph = nx.Graph()

    for place in places:
        graph.add_node(
            place["id"],
            name=place["name"],
            lat=float(place["lat"]),
            lng=float(place["lng"]),
            congestion=float(place["congestion"]),
            monthly_visitors=int(place["monthly_visitors"]),
            selected_count=int(place.get("selected_count", 0)),
            category=str(place.get("category", "unknown")),
            categoryLarge=place.get("categoryLarge"),
            categoryMedium=place.get("categoryMedium"),
            categorySmall=place.get("categorySmall"),
            description=str(place.get("description", place["name"])),
            tags=list(place.get("tags", [str(place.get("category", "unknown"))])),
            distance_m=place.get("distance_m"),
        )

    for index, place_a in enumerate(places):
        for place_b in places[index + 1 :]:
            distance_m = haversine_distance(
                place_a["lat"],
                place_a["lng"],
                place_b["lat"],
                place_b["lng"],
            )
            if distance_m > max_edge_distance_m:
                continue

            distance_weight = math.exp(-(distance_m**2) / (2 * sigma**2))
            graph.add_edge(
                place_a["id"],
                place_b["id"],
                weight=distance_weight,
                base_weight=distance_weight,
                distance_m=round(distance_m, 1),
            )

    return graph


def filter_places_by_radius(
    places: list[dict[str, Any]],
    current_lat: float,
    current_lng: float,
    radius_m: float,
) -> list[dict[str, Any]]:
    """현재 위치 기준 반경 이내 장소만 남기고 가까운 순서로 정렬한다."""
    radius_m = min(max(float(radius_m), 0.0), MAX_CANDIDATE_RADIUS_M)
    nearby_places: list[dict[str, Any]] = []

    for place in places:
        distance_m = haversine_distance(
            current_lat,
            current_lng,
            place["lat"],
            place["lng"],
        )
        if distance_m <= radius_m:
            place_with_distance = place.copy()
            place_with_distance["distance_m"] = round(distance_m, 1)
            nearby_places.append(place_with_distance)

    nearby_places.sort(key=lambda item: item["distance_m"])
    return nearby_places


# =========================================================
# 2. 기본 장소 점수: 혼잡도 / 인기도 / 현재 위치 거리
# =========================================================


def compute_static_scores(
    graph: nx.Graph,
    current_lat: float,
    current_lng: float,
    distance_sigma_m: float = 1_000,
) -> dict[str, dict[str, float]]:
    """각 장소의 기본 feature를 0~1 범위로 계산한다."""
    if distance_sigma_m <= 0:
        raise ValueError("distance_sigma_m must be positive")

    scores: dict[str, dict[str, float]] = {}

    for node_id, attrs in graph.nodes(data=True):
        quietness = clamp((100 - attrs["congestion"]) / 100)
        congestion_score = quietness**2

        visitor_ratio = math.log1p(max(attrs["monthly_visitors"], 0)) / math.log1p(
            MONTHLY_VISITOR_REFERENCE
        )
        visitor_hidden_score = 1 - min(visitor_ratio, 1.0)

        selected_ratio = math.log1p(max(attrs["selected_count"], 0)) / math.log1p(
            SELECTED_COUNT_REFERENCE
        )
        selection_hidden_score = 1 - min(selected_ratio, 1.0)

        distance_m = haversine_distance(
            current_lat,
            current_lng,
            attrs["lat"],
            attrs["lng"],
        )
        distance_score = math.exp(
            -(distance_m**2) / (2 * distance_sigma_m**2)
        )

        scores[node_id] = {
            "congestion_score": congestion_score,
            "visitor_hidden_score": visitor_hidden_score,
            "selection_hidden_score": selection_hidden_score,
            "distance_score": distance_score,
            "distance_m": distance_m,
        }

    return scores


def compute_base_score(
    feature_scores: dict[str, float],
    weights: dict[str, float] | None = None,
    tag_similarity_score: float = 0.0,
) -> float:
    """혼잡도·숨은 정도·거리 feature를 하나의 기본 점수로 합친다."""
    selected_weights = weights or DEFAULT_BASE_WEIGHTS
    score = (
        selected_weights["congestion"] * feature_scores["congestion_score"]
        + selected_weights["monthly_visitors"]
        * feature_scores["visitor_hidden_score"]
        + selected_weights["selected_count"]
        * feature_scores["selection_hidden_score"]
        + selected_weights["distance"] * feature_scores["distance_score"]
        + selected_weights.get("tag_similarity", 0.0) * tag_similarity_score
    )
    return clamp(score)


# =========================================================
# 3. Personalized PageRank
# =========================================================


def compute_ppr_scores(
    graph: nx.Graph,
    selected_history: list[str],
    decay_rate: float = 0.5,
    alpha: float = 0.85,
) -> dict[str, float]:
    """최근 선택 장소일수록 높은 personalization을 주어 PPR을 계산한다."""
    if not 0 < decay_rate <= 1:
        raise ValueError("decay_rate must be in (0, 1]")

    valid_history = [place_id for place_id in selected_history if place_id in graph]
    if not valid_history:
        return {node: 0.0 for node in graph.nodes()}

    personalization: dict[str, float] = {}
    count = len(valid_history)
    for index, place_id in enumerate(valid_history):
        recency_weight = decay_rate ** (count - 1 - index)
        personalization[place_id] = (
            personalization.get(place_id, 0.0) + recency_weight
        )

    total = sum(personalization.values())
    personalization = {
        node: personalization.get(node, 0.0) / total for node in graph.nodes()
    }

    return nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
    )


def normalize_candidate_scores(
    raw_scores: dict[str, float],
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    """후보 집합 안에서 min-max 정규화한다. 모두 같으면 0으로 반환한다."""
    ids = list(candidate_ids)
    if not ids:
        return {}

    values = [float(raw_scores.get(candidate_id, 0.0)) for candidate_id in ids]
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        return {candidate_id: 0.0 for candidate_id in ids}

    denominator = maximum - minimum
    return {
        candidate_id: clamp((float(raw_scores.get(candidate_id, 0.0)) - minimum) / denominator)
        for candidate_id in ids
    }


# =========================================================
# 4. 다음 장소 전이 scorer: LLM + 규칙 fallback
# =========================================================


@dataclass(frozen=True)
class TransitionContext:
    recent_places: list[dict[str, Any]]
    current_time: str
    user_preferences: str = "조용하고 덜 알려진 장소"
    weather: str | None = None
    current_place: dict[str, Any] | None = None


@dataclass(frozen=True)
class TransitionScore:
    score: float
    reason: str
    source: str
    recommendation_reason: str = ""


@dataclass(frozen=True)
class SimilarityScore:
    score: float
    reason: str
    source: str
    tag_cosine_score: float = 0.0


class TransitionScorer(Protocol):
    def score_candidates(
        self,
        context: TransitionContext,
        candidates: list[dict[str, Any]],
    ) -> dict[str, TransitionScore]:
        """후보별 다음 장소 적합도를 반환한다."""


class SimilarityScorer(Protocol):
    def score_candidates(
        self,
        current_place: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, SimilarityScore]:
        """현재 장소와 후보의 분위기 유사도를 반환한다."""


CATEGORY_TRANSITIONS: dict[str, dict[str, float]] = {
    "bookstore": {
        "cafe": 0.90,
        "nature": 0.78,
        "walk": 0.78,
        "gallery": 0.68,
        "museum": 0.62,
        "bookstore": 0.20,
    },
    "museum": {
        "cafe": 0.86,
        "walk": 0.78,
        "nature": 0.74,
        "bookstore": 0.66,
        "gallery": 0.50,
        "museum": 0.20,
    },
    "gallery": {
        "cafe": 0.86,
        "walk": 0.76,
        "nature": 0.70,
        "bookstore": 0.66,
        "gallery": 0.20,
    },
    "cafe": {
        "walk": 0.86,
        "nature": 0.82,
        "gallery": 0.68,
        "museum": 0.64,
        "bookstore": 0.62,
        "cafe": 0.25,
    },
    "bakery": {
        "walk": 0.86,
        "nature": 0.78,
        "cafe": 0.54,
        "shopping": 0.62,
        "bakery": 0.20,
    },
    "restaurant": {
        "cafe": 0.92,
        "bakery": 0.85,
        "walk": 0.72,
        "nature": 0.64,
        "restaurant": 0.15,
    },
    "shopping": {
        "cafe": 0.84,
        "walk": 0.80,
        "nature": 0.68,
        "view": 0.64,
        "shopping": 0.20,
    },
    "nature": {
        "cafe": 0.82,
        "restaurant": 0.76,
        "view": 0.68,
        "gallery": 0.56,
        "nature": 0.28,
        "walk": 0.42,
    },
    "walk": {
        "cafe": 0.84,
        "restaurant": 0.78,
        "view": 0.72,
        "gallery": 0.60,
        "walk": 0.24,
        "nature": 0.40,
    },
    "river": {
        "cafe": 0.82,
        "restaurant": 0.76,
        "view": 0.70,
        "river": 0.22,
    },
    "view": {
        "cafe": 0.78,
        "restaurant": 0.76,
        "walk": 0.58,
        "view": 0.22,
    },
    "history": {
        "cafe": 0.82,
        "walk": 0.74,
        "gallery": 0.68,
        "bookstore": 0.62,
        "history": 0.22,
    },
    "hot_spring": {
        "cafe": 0.74,
        "restaurant": 0.82,
        "walk": 0.70,
        "nature": 0.64,
        "hot_spring": 0.20,
    },
}


class HeuristicTransitionScorer:
    """실제 이동 로그가 쌓이기 전 사용하는 결정론적 전이 fallback."""

    def score_candidates(
        self,
        context: TransitionContext,
        candidates: list[dict[str, Any]],
    ) -> dict[str, TransitionScore]:
        if not candidates:
            return {}

        recent_categories = [
            str(place.get("category", "unknown")) for place in context.recent_places
        ]
        current_place = context.current_place or {}
        latest_category = str(
            current_place.get(
                "category",
                recent_categories[-1] if recent_categories else "unknown",
            )
        )
        current_name = str(current_place.get("name", "현재 장소"))
        transition_table = CATEGORY_TRANSITIONS.get(latest_category, {})
        hour = _extract_hour(context.current_time)

        results: dict[str, TransitionScore] = {}
        for candidate in candidates:
            place_id = str(candidate["place_id"])
            category = str(candidate.get("category", "unknown"))
            score = transition_table.get(category, 0.50)

            if category == latest_category:
                score = min(score, 0.25)
            if category in recent_categories[-2:]:
                score -= 0.08

            score += _time_category_adjustment(hour, category)
            score = clamp(score)
            reason = (
                f"최근 {latest_category} 방문 이후 {category} 활동으로의 "
                f"전환 가능성을 규칙 기반으로 평가"
            )
            results[place_id] = TransitionScore(
                score=score,
                reason=reason,
                source="heuristic",
                recommendation_reason=(
                    f"{current_name} 다음에 {candidate['name']}에서 "
                    f"새로운 {category} 분위기를 즐기기 좋아요!"
                ),
            )

        return results


def _extract_hour(value: str) -> int | None:
    try:
        return datetime.fromisoformat(value).hour
    except (TypeError, ValueError):
        return None


def _time_category_adjustment(hour: int | None, category: str) -> float:
    if hour is None:
        return 0.0
    if 11 <= hour <= 13 and category in {"restaurant", "bakery"}:
        return 0.10
    if 14 <= hour <= 17 and category in {"cafe", "walk", "nature"}:
        return 0.07
    if 18 <= hour <= 21 and category in {"restaurant", "view", "walk"}:
        return 0.08
    return 0.0


class OpenAITransitionScorer:
    """
    OpenAI Responses API를 사용해 후보의 다음 장소 적합도를 평가한다.

    client를 주입할 수 있어 테스트에서는 실제 API를 호출하지 않는다.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        fallback: TransitionScorer | None = None,
        max_output_tokens: int = 3_000,
        reasoning_effort: str = "minimal",
    ) -> None:
        self.client = client
        self.model = model
        self.fallback = fallback or HeuristicTransitionScorer()
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai 패키지가 설치되어 있지 않습니다") from exc
        self.client = OpenAI()
        return self.client

    def score_candidates(
        self,
        context: TransitionContext,
        candidates: list[dict[str, Any]],
    ) -> dict[str, TransitionScore]:
        if not candidates:
            return {}

        fallback_scores = self.fallback.score_candidates(context, candidates)
        candidate_ids = {str(candidate["place_id"]) for candidate in candidates}
        payload = {
            "current_time": context.current_time,
            "weather": context.weather,
            "user_preferences": context.user_preferences,
            "current_place": _normalize_place_text(context.current_place),
            "recent_places": [
                _normalize_place_text(place) for place in context.recent_places[-4:]
            ],
            "category_catalog": KAKAO_CATEGORY_GROUPS,
            "candidates": [
                _normalize_place_text(candidate) for candidate in candidates
            ],
        }

        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "place_id": {"type": "string"},
                            "transition_score": {"type": "number"},
                            "reason": {"type": "string"},
                            "recommendation_reason": {"type": "string"},
                        },
                        "required": [
                            "place_id",
                            "transition_score",
                            "reason",
                            "recommendation_reason",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        }

        instructions = (
            "당신은 여행 동선의 다음 장소 적합도를 평가하는 랭킹 보조 모델이다. "
            "각 후보가 최근 방문 흐름에서 실제 다음 행선지로 얼마나 자연스러운지 0부터 1로 평가하라. "
            "입력의 category_catalog는 카카오맵 카테고리 그룹 코드와 의미다. "
            "후보의 카카오맵 카테고리, 내부 category, tags, description을 함께 보고 후보별로 직접 점수화하라. "
            "현재 장소와 단순히 같은 카테고리라는 이유만으로 높은 점수를 주지 말고, "
            "활동 전환, 시간대, 이동 부담, 최근 방문의 반복 여부, 사용자의 조용하고 숨은 장소 선호를 고려하라. "
            "음식점 다음에 카페를 무조건 높게 주는 고정 규칙을 적용하지 말고, "
            "현재 문맥에서 각 후보가 실제로 자연스러운지 개별적으로 판단하라. "
            "후보의 기본 점수는 참고 정보일 뿐 다시 산술 계산하지 마라. "
            "입력에 없는 장소나 사실을 만들지 말고 모든 입력 후보를 정확히 한 번씩 반환하라. "
            "reason은 내부 판단 근거를 한국어 한 문장으로 간결하게 작성하라. "
            "recommendation_reason은 사용자 화면에 표시할 문구로, 현재 장소와 후보 장소의 "
            "연결 이유가 드러나도록 친근한 한국어 한 문장으로 작성하라."
        )

        try:
            response = self._get_client().responses.create(
                model=self.model,
                instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "next_place_transition_scores",
                        "description": "후보별 다음 장소 전이 적합도",
                        "schema": schema,
                        "strict": True,
                    }
                },
                max_output_tokens=self.max_output_tokens,
                reasoning={"effort": self.reasoning_effort},
                store=False,
            )
            output_text = str(getattr(response, "output_text", "") or "").strip()
            if not output_text:
                status = getattr(response, "status", "unknown")
                incomplete_details = getattr(response, "incomplete_details", None)
                reason = getattr(incomplete_details, "reason", "unknown")
                raise ValueError(
                    f"empty LLM output (status={status}, reason={reason})"
                )
            parsed = json.loads(output_text)
            raw_items = parsed.get("scores", [])
            if not isinstance(raw_items, list):
                raise ValueError("scores must be a list")
        except Exception as exc:  # API 장애가 추천 전체 장애가 되지 않게 한다.
            LOGGER.warning("LLM transition scoring failed; using fallback: %s", exc)
            return _rename_score_source(fallback_scores, "heuristic_fallback")

        results: dict[str, TransitionScore] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            place_id = str(item.get("place_id", ""))
            if place_id not in candidate_ids or place_id in results:
                continue
            try:
                score = clamp(float(item.get("transition_score", 0.0)))
            except (TypeError, ValueError):
                continue
            reason = str(item.get("reason", "다음 장소 흐름을 LLM이 평가함")).strip()
            recommendation_reason = str(
                item.get("recommendation_reason", "")
            ).strip()
            results[place_id] = TransitionScore(
                score=score,
                reason=reason or "다음 장소 흐름을 LLM이 평가함",
                source="openai",
                recommendation_reason=(
                    recommendation_reason
                    or fallback_scores[place_id].recommendation_reason
                ),
            )

        for place_id in candidate_ids - results.keys():
            fallback_score = fallback_scores[place_id]
            results[place_id] = TransitionScore(
                score=fallback_score.score,
                reason=fallback_score.reason,
                source="heuristic_fallback",
                recommendation_reason=fallback_score.recommendation_reason,
            )

        return results


class HeuristicSimilarityScorer:
    """LLM을 사용할 수 없을 때 기존 코드 기반 유사도를 제공한다."""

    def score_candidates(
        self,
        current_place: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, SimilarityScore]:
        tag_cosine_scores = compute_pairwise_tag_cosine_scores(
            current_place,
            candidates,
        )
        rows = recommend_similar_places(
            current_place,
            candidates,
            top_k=len(candidates),
        )
        return {
            row["place_id"]: SimilarityScore(
                score=row["similarity_score"],
                reason="카테고리, 태그, 설명, 거리를 코드로 비교",
                source="heuristic",
                tag_cosine_score=tag_cosine_scores.get(row["place_id"], 0.0),
            )
            for row in rows
        }


class OpenAISimilarityScorer:
    """gpt-5-nano로 현재 장소와 후보의 분위기 유사도를 평가한다."""

    def __init__(
        self,
        client: Any | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        fallback: SimilarityScorer | None = None,
        max_output_tokens: int = 3_000,
        reasoning_effort: str = "minimal",
    ) -> None:
        self.client = client
        self.model = model
        self.fallback = fallback or HeuristicSimilarityScorer()
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai 패키지가 설치되어 있지 않습니다") from exc
        self.client = OpenAI()
        return self.client

    def score_candidates(
        self,
        current_place: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, SimilarityScore]:
        if not candidates:
            return {}

        tag_cosine_scores = compute_pairwise_tag_cosine_scores(
            current_place,
            candidates,
        )
        fallback_scores = self.fallback.score_candidates(current_place, candidates)
        candidate_ids = {str(candidate["id"]) for candidate in candidates}
        normalized_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            normalized = _normalize_place_text(candidate) or {}
            normalized["tag_cosine_score"] = round(
                tag_cosine_scores.get(str(candidate["id"]), 0.0),
                6,
            )
            normalized_candidates.append(normalized)
        payload = {
            "category_catalog": KAKAO_CATEGORY_GROUPS,
            "current_place": _normalize_place_text(current_place),
            "candidates": normalized_candidates,
        }
        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "place_id": {"type": "string"},
                            "similarity_score": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["place_id", "similarity_score", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        }
        instructions = (
            "당신은 현재 장소와 비슷한 느낌의 장소를 찾는 여행 추천 랭커다. "
            "category_catalog의 카카오맵 그룹 코드, 장소의 category, kakao_category_code, "
            "tags, description, tag_cosine_score를 함께 보고 후보별 분위기 유사도를 0부터 1까지 직접 평가하라. "
            "단순히 같은 카테고리인지보다 이용 목적, 분위기, 활동, 공간 성격이 얼마나 비슷한지 판단하라. "
            "tag_cosine_score는 코드가 계산한 태그 기준점이므로 참고하되, description과 활동 맥락을 반영해 보정하라. "
            "거리는 가까울수록 약간 유리하게 보되, 유사도의 핵심은 장소의 느낌과 활동이다. "
            "모든 후보를 정확히 한 번씩 반환하고, reason은 한국어 한 문장으로 작성하라."
        )

        try:
            response = self._get_client().responses.create(
                model=self.model,
                instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "similar_place_scores",
                        "description": "현재 장소와 후보의 분위기 유사도",
                        "schema": schema,
                        "strict": True,
                    }
                },
                max_output_tokens=self.max_output_tokens,
                reasoning={"effort": self.reasoning_effort},
                store=False,
            )
            output_text = str(getattr(response, "output_text", "") or "").strip()
            if not output_text:
                status = getattr(response, "status", "unknown")
                incomplete_details = getattr(response, "incomplete_details", None)
                reason = getattr(incomplete_details, "reason", "unknown")
                raise ValueError(
                    f"empty LLM output (status={status}, reason={reason})"
                )
            parsed = json.loads(output_text)
            raw_items = parsed.get("scores", [])
            if not isinstance(raw_items, list):
                raise ValueError("scores must be a list")
        except Exception as exc:  # 유사도 API 장애가 전체 추천 장애가 되지 않게 한다.
            LOGGER.warning("LLM similarity scoring failed; using fallback: %s", exc)
            return {
                place_id: SimilarityScore(
                    score=value.score,
                    reason=value.reason,
                    source="heuristic_fallback",
                    tag_cosine_score=value.tag_cosine_score,
                )
                for place_id, value in fallback_scores.items()
            }

        results: dict[str, SimilarityScore] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            place_id = str(item.get("place_id", ""))
            if place_id not in candidate_ids or place_id in results:
                continue
            try:
                score = clamp(float(item.get("similarity_score", 0.0)))
            except (TypeError, ValueError):
                continue
            reason = str(item.get("reason", "현재 장소와의 분위기 유사도를 LLM이 평가함")).strip()
            results[place_id] = SimilarityScore(
                score=score,
                reason=reason or "현재 장소와의 분위기 유사도를 LLM이 평가함",
                source="openai",
                tag_cosine_score=tag_cosine_scores.get(place_id, 0.0),
            )

        for place_id in candidate_ids - results.keys():
            fallback_score = fallback_scores[place_id]
            results[place_id] = SimilarityScore(
                score=fallback_score.score,
                reason=fallback_score.reason,
                source="heuristic_fallback",
                tag_cosine_score=fallback_score.tag_cosine_score,
            )
        return results


def _rename_score_source(
    scores: dict[str, TransitionScore],
    source: str,
) -> dict[str, TransitionScore]:
    return {
        place_id: TransitionScore(
            score=value.score,
            reason=value.reason,
            source=source,
            recommendation_reason=value.recommendation_reason,
        )
        for place_id, value in scores.items()
    }


def _one_line_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalize_place_text(place: dict[str, Any] | None) -> dict[str, Any] | None:
    if place is None:
        return None
    normalized = dict(place)
    if "description" in normalized:
        normalized["description"] = _one_line_text(normalized["description"])
    category_code = normalized.get("kakao_category_code") or normalized.get(
        "category_group_code"
    )
    if category_code not in KAKAO_CATEGORY_GROUPS:
        category_value = str(normalized.get("category", ""))
        category_code = (
            category_value
            if category_value in KAKAO_CATEGORY_GROUPS
            else INTERNAL_CATEGORY_TO_KAKAO.get(category_value.lower())
        )
    if category_code:
        normalized["kakao_category_code"] = category_code
    return normalized


def create_default_transition_scorer(
    use_llm: bool | None = None,
    model: str | None = None,
) -> TransitionScorer:
    """
    환경변수에 따라 scorer를 생성한다.

    `USE_LLM`이 명시되지 않으면 API 키 존재 여부로 LLM 사용을 결정한다.
    `USE_LLM=0`은 API 키가 있어도 LLM을 비활성화한다.
    """
    if use_llm is None:
        raw_use_llm = os.getenv("USE_LLM")
        if raw_use_llm is None:
            use_llm = bool(os.getenv("OPENAI_API_KEY"))
        else:
            use_llm = raw_use_llm.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

    fallback = HeuristicTransitionScorer()
    if not use_llm:
        return fallback
    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.warning("USE_LLM is enabled but OPENAI_API_KEY is missing")
        return fallback

    return OpenAITransitionScorer(
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        fallback=fallback,
    )


def create_default_similarity_scorer(
    use_llm: bool | None = None,
    model: str | None = None,
) -> SimilarityScorer:
    """환경변수에 따라 유사 장소 scorer를 생성한다."""
    if use_llm is None:
        raw_use_llm = os.getenv("USE_LLM")
        if raw_use_llm is None:
            use_llm = bool(os.getenv("OPENAI_API_KEY"))
        else:
            use_llm = raw_use_llm.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

    fallback = HeuristicSimilarityScorer()
    if not use_llm:
        return fallback
    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.warning("USE_LLM is enabled but OPENAI_API_KEY is missing")
        return fallback

    return OpenAISimilarityScorer(
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        fallback=fallback,
    )


# =========================================================
# 5. 후보 사전 선별과 하이브리드 최종 랭킹
# =========================================================


def combine_recommendation_scores(
    base_score: float,
    ppr_score: float,
    transition_score: float,
    has_history: bool,
    weights: dict[str, float] | None = None,
) -> float:
    """선택 이력이 있으면 기본·PPR·전이 점수를 합치고, 없으면 기본 점수만 쓴다."""
    if not has_history:
        return clamp(base_score)

    selected_weights = weights or DEFAULT_HYBRID_WEIGHTS
    score = (
        selected_weights["base"] * base_score
        + selected_weights["ppr"] * ppr_score
        + selected_weights["transition"] * transition_score
    )
    return clamp(score)


def _policy_passed(
    attrs: dict[str, Any],
    max_congestion: float,
    max_monthly_visitors: int,
) -> bool:
    return (
        attrs["congestion"] <= max_congestion
        and attrs["monthly_visitors"] <= max_monthly_visitors
    )


def preselect_candidates(
    graph: nx.Graph,
    static_scores: dict[str, dict[str, float]],
    ppr_scores: dict[str, float],
    selected_history: Sequence[str],
    limit: int = DEFAULT_LLM_CANDIDATE_LIMIT,
    base_weights: dict[str, float] | None = None,
    max_congestion: float = DEFAULT_MAX_CONGESTION,
    max_monthly_visitors: int = DEFAULT_MAX_MONTHLY_VISITORS,
    tag_similarity_scores: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    """비싼 LLM 호출 전에 코드 점수로 후보를 최대 limit개까지 줄인다."""
    selected_set = set(selected_history)
    candidate_ids = [node_id for node_id in graph.nodes() if node_id not in selected_set]
    normalized_ppr = normalize_candidate_scores(ppr_scores, candidate_ids)
    tag_similarity_scores = tag_similarity_scores or {}
    base_scores = {
        node_id: compute_base_score(
            static_scores[node_id],
            base_weights,
            tag_similarity_scores.get(node_id, 0.0),
        )
        for node_id in candidate_ids
    }

    rows: list[dict[str, Any]] = []
    has_history = bool(selected_history)
    for node_id in candidate_ids:
        attrs = graph.nodes[node_id]
        pre_rank_score = (
            0.90 * base_scores[node_id] + 0.10 * normalized_ppr[node_id]
            if has_history
            else base_scores[node_id]
        )
        rows.append(
            {
                "place_id": node_id,
                "name": attrs["name"],
                "category": attrs.get("category", "unknown"),
                "categoryLarge": attrs.get("categoryLarge"),
                "categoryMedium": attrs.get("categoryMedium"),
                "categorySmall": attrs.get("categorySmall"),
                "description": attrs.get("description", attrs["name"]),
                "tags": list(attrs.get("tags", [])),
                "distance_m": round(static_scores[node_id]["distance_m"], 1),
                "congestion": attrs["congestion"],
                "monthly_visitors": attrs["monthly_visitors"],
                "selected_count": attrs["selected_count"],
                "base_score": round(base_scores[node_id], 6),
                "tag_similarity_score": round(
                    tag_similarity_scores.get(node_id, 0.0), 6
                ),
                "ppr_score": round(normalized_ppr[node_id], 6),
                "policy_passed": _policy_passed(
                    attrs,
                    max_congestion,
                    max_monthly_visitors,
                ),
                "pre_rank_score": pre_rank_score,
            }
        )

    rows.sort(
        key=lambda row: (row["policy_passed"], row["pre_rank_score"]),
        reverse=True,
    )
    return rows[: max(limit, 0)], base_scores, normalized_ppr


def _recent_places_from_graph(
    graph: nx.Graph,
    selected_history: Sequence[str],
) -> list[dict[str, Any]]:
    recent_places: list[dict[str, Any]] = []
    for place_id in selected_history[-4:]:
        if place_id not in graph:
            continue
        attrs = graph.nodes[place_id]
        recent_places.append(
            {
                "id": place_id,
                "name": attrs["name"],
                "category": attrs.get("category", "unknown"),
                "description": attrs.get("description", attrs["name"]),
                "tags": list(attrs.get("tags", [])),
            }
        )
    return recent_places


def compute_tag_similarity_scores(
    graph: nx.Graph,
    selected_history: Sequence[str],
    recent_places: Sequence[dict[str, Any]] | None = None,
    current_place: dict[str, Any] | None = None,
    decay_rate: float = 0.5,
) -> dict[str, float]:
    """최근 선택 장소의 태그 취향과 후보 태그의 코사인 유사도를 계산한다."""
    scores = {node_id: 0.0 for node_id in graph.nodes()}
    if not selected_history:
        return scores

    preference_places: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for place in list(recent_places or [])[-4:]:
        place_id = str(place.get("id", ""))
        if place_id and place_id in seen_ids:
            continue
        preference_places.append(place)
        if place_id:
            seen_ids.add(place_id)

    if current_place is not None:
        current_id = str(current_place.get("id", ""))
        if not current_id or current_id not in seen_ids:
            preference_places.append(current_place)

    if not preference_places:
        preference_places = _recent_places_from_graph(graph, selected_history)
    if not preference_places:
        return scores

    graph_places = [
        {"tags": attrs.get("tags", [])} for _, attrs in graph.nodes(data=True)
    ]
    vocabulary = build_tag_vocabulary(graph_places + preference_places)
    user_vector = build_user_tag_vector(
        preference_places,
        vocabulary,
        decay_rate=decay_rate,
    )
    return {
        node_id: cosine_similarity(
            user_vector,
            build_tag_vector(attrs.get("tags", []), vocabulary),
        )
        for node_id, attrs in graph.nodes(data=True)
    }


def recommend_next_places_hybrid(
    graph: nx.Graph,
    static_scores: dict[str, dict[str, float]],
    ppr_scores: dict[str, float],
    selected_history: list[str],
    transition_scorer: TransitionScorer | None = None,
    recent_places: list[dict[str, Any]] | None = None,
    current_time: str | None = None,
    user_preferences: str = "조용하고 덜 알려진 장소",
    weather: str | None = None,
    current_place: dict[str, Any] | None = None,
    base_weights: dict[str, float] | None = None,
    hybrid_weights: dict[str, float] | None = None,
    top_k: int = 5,
    llm_candidate_limit: int = DEFAULT_LLM_CANDIDATE_LIMIT,
    max_congestion: float = DEFAULT_MAX_CONGESTION,
    max_monthly_visitors: int = DEFAULT_MAX_MONTHLY_VISITORS,
) -> list[dict[str, Any]]:
    """코드 기본 점수, PPR, LLM 전이 점수를 결합해 최종 장소를 추천한다."""
    tag_similarity_scores = compute_tag_similarity_scores(
        graph=graph,
        selected_history=selected_history,
        recent_places=recent_places,
        current_place=current_place,
    )
    candidates, base_scores, normalized_ppr = preselect_candidates(
        graph=graph,
        static_scores=static_scores,
        ppr_scores=ppr_scores,
        selected_history=selected_history,
        tag_similarity_scores=tag_similarity_scores,
        limit=max(llm_candidate_limit, top_k),
        base_weights=base_weights,
        max_congestion=max_congestion,
        max_monthly_visitors=max_monthly_visitors,
    )
    if not candidates:
        return []

    has_history = bool(selected_history)
    transition_scores: dict[str, TransitionScore] = {
        candidate["place_id"]: TransitionScore(
            score=0.0,
            reason="선택 이력이 없어 기본 점수만 사용",
            source="cold_start",
            recommendation_reason=(
                f"{candidate['name']}은 조용하고 덜 알려진 장소 조건에 잘 맞아요!"
            ),
        )
        for candidate in candidates
    }

    if has_history and transition_scorer is None:
        transition_scorer = create_default_transition_scorer()

    if has_history and transition_scorer is not None:
        context = TransitionContext(
            recent_places=(
                recent_places
                if recent_places is not None
                else _recent_places_from_graph(graph, selected_history)
            )[-4:],
            current_time=current_time or datetime.now().astimezone().isoformat(),
            user_preferences=user_preferences,
            weather=weather,
            current_place=current_place,
        )
        transition_scores = transition_scorer.score_candidates(context, candidates)

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        node_id = candidate["place_id"]
        attrs = graph.nodes[node_id]
        transition = transition_scores.get(
            node_id,
            TransitionScore(
                score=0.0,
                reason="전이 점수를 얻지 못해 0점 처리",
                source="missing",
                recommendation_reason=f"{candidate['name']}을 다음 장소로 살펴보세요!",
            ),
        )
        final_score = combine_recommendation_scores(
            base_score=base_scores[node_id],
            ppr_score=normalized_ppr[node_id],
            transition_score=transition.score,
            has_history=has_history,
            weights=hybrid_weights,
        )

        feature_scores = static_scores[node_id]
        results.append(
            {
                "place_id": node_id,
                "name": attrs["name"],
                "categoryLarge": attrs.get("categoryLarge"),
                "categoryMedium": attrs.get("categoryMedium"),
                "categorySmall": attrs.get("categorySmall"),
                "description": attrs.get("description", attrs["name"]),
                "tags": list(attrs.get("tags", [])),
                "final_score": round(final_score, 6),
                "filter_relaxed": not candidate["policy_passed"],
                "transition_reason": transition.reason,
                "recommendation_reason": transition.recommendation_reason,
                "transition_source": transition.source,
                "detail": {
                    "base_score": round(base_scores[node_id], 6),
                    "tag_similarity_score": candidate["tag_similarity_score"],
                    "congestion_score": round(
                        feature_scores["congestion_score"], 6
                    ),
                    "visitor_hidden_score": round(
                        feature_scores["visitor_hidden_score"], 6
                    ),
                    "selection_hidden_score": round(
                        feature_scores["selection_hidden_score"], 6
                    ),
                    "distance_score": round(feature_scores["distance_score"], 6),
                    "distance_m": round(feature_scores["distance_m"], 1),
                    "ppr_score": round(normalized_ppr[node_id], 6),
                    "transition_score": round(transition.score, 6),
                },
            }
        )

    results.sort(
        key=lambda item: (not item["filter_relaxed"], item["final_score"]),
        reverse=True,
    )
    return results[:top_k]


# 기존 호출부와의 호환을 위해 기존 함수 이름도 유지한다.
def recommend_next_places(
    graph: nx.Graph,
    static_scores: dict[str, dict[str, float]],
    ppr_scores: dict[str, float],
    selected_history: list[str],
    weights: dict[str, float] | None = None,
    top_k: int = 5,
    max_congestion: float = DEFAULT_MAX_CONGESTION,
    max_monthly_visitors: int = DEFAULT_MAX_MONTHLY_VISITORS,
) -> list[dict[str, Any]]:
    """기존 정적 점수 + PPR 보너스 방식. 신규 코드는 hybrid 함수를 권장한다."""
    legacy_weights = weights or {
        "congestion": 0.40,
        "monthly_visitors": 0.25,
        "selected_count": 0.15,
        "distance": 0.20,
        "ppr": 0.10,
    }
    results: list[dict[str, Any]] = []

    for node_id in graph.nodes():
        if node_id in selected_history:
            continue
        feature_scores = static_scores[node_id]
        ppr = ppr_scores.get(node_id, 0.0)
        static_part = (
            legacy_weights["congestion"] * feature_scores["congestion_score"]
            + legacy_weights["monthly_visitors"]
            * feature_scores["visitor_hidden_score"]
            + legacy_weights["selected_count"]
            * feature_scores["selection_hidden_score"]
            + legacy_weights["distance"] * feature_scores["distance_score"]
        )
        final_score = static_part * (1 + legacy_weights["ppr"] * ppr)
        attrs = graph.nodes[node_id]
        results.append(
            {
                "place_id": node_id,
                "name": attrs["name"],
                "final_score": round(final_score, 4),
                "filter_relaxed": not _policy_passed(
                    attrs,
                    max_congestion,
                    max_monthly_visitors,
                ),
                "detail": {
                    "congestion_score": round(
                        feature_scores["congestion_score"], 3
                    ),
                    "visitor_hidden_score": round(
                        feature_scores["visitor_hidden_score"], 3
                    ),
                    "selection_hidden_score": round(
                        feature_scores["selection_hidden_score"], 3
                    ),
                    "distance_score": round(feature_scores["distance_score"], 3),
                    "ppr_score": round(ppr, 4),
                },
            }
        )

    results.sort(
        key=lambda item: (not item["filter_relaxed"], item["final_score"]),
        reverse=True,
    )
    return results[:top_k]


# =========================================================
# 6. Spring 요청 어댑터: 비슷한 장소 + 다음 장소
# =========================================================


def _validate_spring_request(request: dict[str, Any]) -> None:
    if not isinstance(request, dict):
        raise ValueError("추천 요청은 JSON 객체여야 합니다")

    required = {
        "request_id",
        "session_id",
        "current_place",
        "recent_places",
        "visited_place_ids",
        "candidates",
        "context",
    }
    missing = required - request.keys()
    if missing:
        raise ValueError(f"추천 요청 필드가 누락되었습니다: {sorted(missing)}")

    candidates = request["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates는 한 개 이상의 장소 배열이어야 합니다")
    candidate_ids = [str(candidate.get("id", "")) for candidate in candidates]
    if "" in candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate id는 비어 있지 않고 중복되지 않아야 합니다")

    recent_places = request["recent_places"]
    if not isinstance(recent_places, list) or len(recent_places) > 4:
        raise ValueError("recent_places는 최대 4개여야 합니다")

    context = request["context"]
    if context.get("similar_top_k") != SIMILAR_TOP_K:
        raise ValueError("similar_top_k는 5로 고정되어야 합니다")
    if context.get("next_top_k") != NEXT_TOP_K:
        raise ValueError("next_top_k는 5로 고정되어야 합니다")
    radius_m = float(context.get("radius_m", MAX_CANDIDATE_RADIUS_M))
    if not 0 < radius_m <= MAX_CANDIDATE_RADIUS_M:
        raise ValueError("radius_m은 0보다 크고 1000 이하여야 합니다")


def _coordinate(place: dict[str, Any], long_name: str, short_name: str) -> float:
    if long_name in place:
        return float(place[long_name])
    return float(place[short_name])


def _to_graph_place(
    place: dict[str, Any],
    *,
    default_congestion: float = 50.0,
) -> dict[str, Any]:
    return {
        "id": str(place["id"]),
        "name": str(place["name"]),
        "lat": _coordinate(place, "latitude", "lat"),
        "lng": _coordinate(place, "longitude", "lng"),
        "congestion": float(place.get("congestion", default_congestion)),
        "monthly_visitors": int(place.get("monthly_visitors", 0)),
        "selected_count": int(place.get("selected_count", 0)),
        "category": str(place.get("category", "unknown")),
        "categoryLarge": place.get("categoryLarge"),
        "categoryMedium": place.get("categoryMedium"),
        "categorySmall": place.get("categorySmall"),
        "description": _one_line_text(
            place.get("description", place.get("name", ""))
        ),
        "tags": list(place.get("tags", [])),
        "distance_m": place.get("distance_m"),
    }


def _text_tokens(value: Any) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[0-9A-Za-z가-힣_]+", _one_line_text(value))
        if len(token) > 1
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def recommend_similar_places(
    current_place: dict[str, Any],
    candidates: list[dict[str, Any]],
    excluded_place_ids: Sequence[str] = (),
    top_k: int = SIMILAR_TOP_K,
) -> list[dict[str, Any]]:
    """현재 장소의 카테고리·태그·설명만 사용해 비슷한 후보를 고른다."""
    current_id = str(current_place["id"])
    current_category = str(current_place.get("category", "unknown"))
    current_tags = {
        str(tag).strip().lower() for tag in current_place.get("tags", []) if str(tag).strip()
    }
    current_description = _text_tokens(current_place.get("description", ""))
    excluded = {str(place_id) for place_id in excluded_place_ids} | {current_id}

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        place_id = str(candidate["id"])
        if place_id in excluded:
            continue

        if "distance_m" in candidate:
            distance_m = float(candidate["distance_m"])
        else:
            distance_m = haversine_distance(
                _coordinate(current_place, "latitude", "lat"),
                _coordinate(current_place, "longitude", "lng"),
                _coordinate(candidate, "latitude", "lat"),
                _coordinate(candidate, "longitude", "lng"),
            )
        candidate_tags = {
            str(tag).strip().lower()
            for tag in candidate.get("tags", [])
            if str(tag).strip()
        }
        category_score = (
            1.0 if str(candidate.get("category", "unknown")) == current_category else 0.0
        )
        tag_score = _jaccard(current_tags, candidate_tags)
        description_score = _jaccard(
            current_description,
            _text_tokens(candidate.get("description", "")),
        )
        distance_score = math.exp(
            -(distance_m**2) / (2 * MAX_CANDIDATE_RADIUS_M**2)
        )
        similarity_score = (
            0.45 * category_score
            + 0.35 * tag_score
            + 0.10 * description_score
            + 0.10 * distance_score
        )
        results.append(
            {
                "place_id": place_id,
                "name": str(candidate["name"]),
                "categoryLarge": candidate.get("categoryLarge"),
                "categoryMedium": candidate.get("categoryMedium"),
                "categorySmall": candidate.get("categorySmall"),
                "description": _one_line_text(
                    candidate.get("description", candidate["name"])
                ),
                "tags": list(candidate.get("tags", [])),
                "distance_m": round(distance_m, 1),
                "similarity_score": round(similarity_score, 6),
            }
        )

    results.sort(
        key=lambda item: (-item["similarity_score"], item["distance_m"], item["place_id"])
    )
    return [
        {"rank": rank, **item}
        for rank, item in enumerate(results[:top_k], start=1)
    ]


def recommend_similar_places_with_scorer(
    current_place: dict[str, Any],
    candidates: list[dict[str, Any]],
    scorer: SimilarityScorer,
    excluded_place_ids: Sequence[str] = (),
    top_k: int = SIMILAR_TOP_K,
) -> list[dict[str, Any]]:
    """scorer가 계산한 유사도와 이유로 비슷한 장소를 정렬한다."""
    current_id = str(current_place["id"])
    excluded = {str(place_id) for place_id in excluded_place_ids} | {current_id}
    eligible_candidates = [
        candidate
        for candidate in candidates
        if str(candidate["id"]) not in excluded
    ]
    scores = scorer.score_candidates(current_place, eligible_candidates)

    results: list[dict[str, Any]] = []
    for candidate in eligible_candidates:
        place_id = str(candidate["id"])
        score = scores.get(
            place_id,
            SimilarityScore(
                score=0.0,
                reason="유사도 점수를 얻지 못했습니다.",
                source="missing",
            ),
        )
        tag_cosine_score = clamp(score.tag_cosine_score)
        context_similarity_score = clamp(score.score)
        final_similarity_score = clamp(
            0.60 * tag_cosine_score + 0.40 * context_similarity_score
        )
        if "distance_m" in candidate:
            distance_m = float(candidate["distance_m"])
        else:
            distance_m = haversine_distance(
                _coordinate(current_place, "latitude", "lat"),
                _coordinate(current_place, "longitude", "lng"),
                _coordinate(candidate, "latitude", "lat"),
                _coordinate(candidate, "longitude", "lng"),
            )
        results.append(
            {
                "place_id": place_id,
                "name": str(candidate["name"]),
                "categoryLarge": candidate.get("categoryLarge"),
                "categoryMedium": candidate.get("categoryMedium"),
                "categorySmall": candidate.get("categorySmall"),
                "description": _one_line_text(
                    candidate.get("description", candidate["name"])
                ),
                "tags": list(candidate.get("tags", [])),
                "distance_m": round(distance_m, 1),
                "similarity_score": round(final_similarity_score, 6),
                "tag_cosine_score": round(tag_cosine_score, 6),
                "context_similarity_score": round(
                    context_similarity_score, 6
                ),
                "similarity_reason": score.reason,
                "similarity_source": score.source,
            }
        )

    results.sort(
        key=lambda item: (-item["similarity_score"], item["distance_m"], item["place_id"])
    )
    return [
        {"rank": rank, **item}
        for rank, item in enumerate(results[:top_k], start=1)
    ]


def _validate_split_recommendation_request(
    request: dict[str, Any],
    *,
    place_field: str,
    require_recent_places: bool,
) -> None:
    if not isinstance(request, dict):
        raise ValueError("추천 요청은 JSON 객체여야 합니다")

    required = {
        "request_id",
        "session_id",
        place_field,
        "visited_place_ids",
        "candidates",
        "context",
    }
    if require_recent_places:
        required.add("recent_places")
    missing = required - request.keys()
    if missing:
        raise ValueError(f"추천 요청 필드가 누락되었습니다: {sorted(missing)}")

    candidates = request["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates는 한 개 이상의 장소 배열이어야 합니다")
    candidate_ids = [str(candidate.get("id", "")) for candidate in candidates]
    if "" in candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate id는 비어 있지 않고 중복되지 않아야 합니다")

    if require_recent_places:
        recent_places = request["recent_places"]
        if not isinstance(recent_places, list) or len(recent_places) > 4:
            raise ValueError("recent_places는 최대 4개여야 합니다")

    context = request["context"]
    if not isinstance(context, dict) or context.get("top_k") != 5:
        raise ValueError("top_k는 5로 고정되어야 합니다")
    if require_recent_places and not str(context.get("current_time", "")).strip():
        raise ValueError("다음 장소 추천에는 current_time이 필요합니다")
    radius_m = float(context.get("radius_m", MAX_CANDIDATE_RADIUS_M))
    if not 0 < radius_m <= MAX_CANDIDATE_RADIUS_M:
        raise ValueError("radius_m은 0보다 크고 1000 이하여야 합니다")


def _eligible_candidates_within_radius(
    current_place: dict[str, Any],
    candidates: list[dict[str, Any]],
    visited_place_ids: Sequence[str],
    radius_m: float,
) -> list[dict[str, Any]]:
    current_lat = _coordinate(current_place, "latitude", "lat")
    current_lng = _coordinate(current_place, "longitude", "lng")
    excluded_ids = {str(place_id) for place_id in visited_place_ids} | {
        str(current_place["id"])
    }
    eligible_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if str(candidate["id"]) in excluded_ids:
            continue
        distance_m = haversine_distance(
            current_lat,
            current_lng,
            _coordinate(candidate, "latitude", "lat"),
            _coordinate(candidate, "longitude", "lng"),
        )
        if distance_m > radius_m:
            continue
        normalized = dict(candidate)
        normalized["distance_m"] = round(distance_m, 1)
        normalized["description"] = _one_line_text(
            candidate.get("description", candidate["name"])
        )
        eligible_candidates.append(normalized)

    if len(eligible_candidates) < 5:
        raise ValueError("반경 내의 미방문 후보가 최소 5개 필요합니다")
    return eligible_candidates


def process_spring_similar_places_request(
    request: dict[str, Any],
    similarity_scorer: SimilarityScorer | None = None,
) -> dict[str, Any]:
    """선택한 장소와 비슷한 장소만 계산한다."""
    _validate_split_recommendation_request(
        request,
        place_field="selected_place",
        require_recent_places=False,
    )
    selected_place = request["selected_place"]
    eligible_candidates = _eligible_candidates_within_radius(
        current_place=selected_place,
        candidates=request["candidates"],
        visited_place_ids=request["visited_place_ids"],
        radius_m=float(request["context"]["radius_m"]),
    )
    scorer = similarity_scorer or create_default_similarity_scorer()
    similar_places = recommend_similar_places_with_scorer(
        current_place=selected_place,
        candidates=eligible_candidates,
        scorer=scorer,
        top_k=SIMILAR_TOP_K,
    )
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "selected_place_id": str(selected_place["id"]),
        "similar_places": similar_places,
    }


def process_spring_next_places_request(
    request: dict[str, Any],
    transition_scorer: TransitionScorer | None = None,
) -> dict[str, Any]:
    """현재 위치와 최근 선택 흐름을 사용해 다음 장소만 계산한다."""
    _validate_split_recommendation_request(
        request,
        place_field="current_place",
        require_recent_places=True,
    )
    current_place = request["current_place"]
    current_lat = _coordinate(current_place, "latitude", "lat")
    current_lng = _coordinate(current_place, "longitude", "lng")
    eligible_candidates = _eligible_candidates_within_radius(
        current_place=current_place,
        candidates=request["candidates"],
        visited_place_ids=request["visited_place_ids"],
        radius_m=float(request["context"]["radius_m"]),
    )

    graph_places = [_to_graph_place(current_place)]
    known_ids = {str(current_place["id"])}
    for recent_place in request["recent_places"][-4:]:
        recent_id = str(recent_place["id"])
        if recent_id not in known_ids:
            graph_places.append(_to_graph_place(recent_place))
            known_ids.add(recent_id)
    for candidate in eligible_candidates:
        candidate_id = str(candidate["id"])
        if candidate_id not in known_ids:
            graph_places.append(_to_graph_place(candidate))
            known_ids.add(candidate_id)

    graph = build_graph(
        graph_places,
        max_edge_distance_m=1_500,
        edge_sigma_m=750,
    )
    selected_history = [
        str(place["id"]) for place in request["recent_places"][-4:]
    ] + [str(current_place["id"])]
    static_scores = compute_static_scores(graph, current_lat, current_lng)
    ppr_scores = compute_ppr_scores(graph, selected_history, decay_rate=0.5)
    next_places = recommend_next_places_hybrid(
        graph=graph,
        static_scores=static_scores,
        ppr_scores=ppr_scores,
        selected_history=selected_history,
        transition_scorer=transition_scorer,
        recent_places=[
            _normalize_place_text(place) for place in request["recent_places"][-4:]
        ],
        current_time=str(request["context"]["current_time"]),
        user_preferences=str(request["context"].get("user_preferences", "")),
        weather=request["context"].get("weather"),
        current_place=_normalize_place_text(current_place),
        top_k=NEXT_TOP_K,
        llm_candidate_limit=len(eligible_candidates),
    )

    candidate_ids = {str(candidate["id"]) for candidate in eligible_candidates}
    next_places = [
        {"rank": rank, **place}
        for rank, place in enumerate(
            (place for place in next_places if place["place_id"] in candidate_ids),
            start=1,
        )
    ][:NEXT_TOP_K]
    if len(next_places) != NEXT_TOP_K:
        raise ValueError("다음 장소 추천 5개를 만들 수 없습니다")

    recommendation_log = log_recommendation_event(
        next_places,
        selected_place_id=None,
        session_id=str(request["session_id"]),
    )
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "current_place_id": str(current_place["id"]),
        "visited_place_ids": [str(place_id) for place_id in request["visited_place_ids"]],
        "next_places": next_places,
        "recommendation_log": recommendation_log,
    }


def process_spring_recommendation_request(
    request: dict[str, Any],
    transition_scorer: TransitionScorer | None = None,
    similarity_scorer: SimilarityScorer | None = None,
) -> dict[str, Any]:
    """기존 통합 계약을 위해 유사 장소와 다음 장소 결과를 함께 반환한다."""
    _validate_spring_request(request)
    similar_response = process_spring_similar_places_request(
        {
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "selected_place": request["current_place"],
            "visited_place_ids": request["visited_place_ids"],
            "candidates": request["candidates"],
            "context": {
                "radius_m": request["context"]["radius_m"],
                "top_k": request["context"]["similar_top_k"],
            },
        },
        similarity_scorer=similarity_scorer,
    )
    next_response = process_spring_next_places_request(
        {
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "current_place": request["current_place"],
            "recent_places": request["recent_places"],
            "visited_place_ids": request["visited_place_ids"],
            "candidates": request["candidates"],
            "context": {
                "current_time": request["context"]["current_time"],
                "weather": request["context"].get("weather"),
                "user_preferences": request["context"].get(
                    "user_preferences", ""
                ),
                "radius_m": request["context"]["radius_m"],
                "top_k": request["context"]["next_top_k"],
            },
        },
        transition_scorer=transition_scorer,
    )
    return {
        "request_id": str(request["request_id"]),
        "session_id": str(request["session_id"]),
        "generated_at": datetime.now().astimezone().isoformat(),
        "current_place_id": str(request["current_place"]["id"]),
        "similar_places": similar_response["similar_places"],
        "next_places": next_response["next_places"],
        "recommendation_log": next_response["recommendation_log"],
    }


# =========================================================
# 7. 추천 로그
# =========================================================


def log_recommendation_event(
    recommendations: list[dict[str, Any]],
    selected_place_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    추천 노출 feature를 저장한다.

    selected_place_id가 None이면 아직 선택 결과가 확정되지 않은 것이므로 label도
    None으로 둔다. 실제 선택 또는 세션 만료 시점에 1/0으로 갱신해야 한다.
    """
    candidates = []
    for rank, recommendation in enumerate(recommendations, start=1):
        if selected_place_id is None:
            label: int | None = None
        else:
            label = 1 if recommendation["place_id"] == selected_place_id else 0

        candidates.append(
            {
                "place_id": recommendation["place_id"],
                "rank": rank,
                "features": recommendation.get("detail", {}),
                "transition_reason": recommendation.get("transition_reason"),
                "recommendation_reason": recommendation.get(
                    "recommendation_reason"
                ),
                "transition_source": recommendation.get("transition_source"),
                "label": label,
            }
        )

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "session_id": session_id,
        "selected_place_id": selected_place_id,
        "candidates": candidates,
    }


def append_result_run(
    run: dict[str, Any],
    path: Path | None = None,
) -> None:
    """기존 실행 기록을 유지하면서 새 실행 결과를 원자적으로 추가한다."""
    path = path or build_result_path()
    payload: dict[str, Any] = {"runs": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"기존 결과 파일을 읽을 수 없습니다: {path}") from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("runs"), list):
            raise ValueError(f"기존 결과 파일 형식이 올바르지 않습니다: {path}")
        payload = loaded

    payload["runs"].append(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


# =========================================================
# 8. 데모 실행
# =========================================================


def _place_summary(place: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": place["id"],
        "name": place["name"],
        "category": place.get("category", "unknown"),
        "description": place.get("description", place["name"]),
        "tags": list(place.get("tags", [])),
    }


def run_demo_scenario(
    title: str,
    all_places: list[dict[str, Any]],
    current_lat: float,
    current_lng: float,
    selected_history: list[str],
    transition_scorer: TransitionScorer,
    radius_m: float = MAX_CANDIDATE_RADIUS_M,
    top_k: int = 5,
    current_time: str | None = None,
    user_preferences: str = "조용하고 덜 알려진 장소",
    weather: str | None = None,
) -> dict[str, Any]:
    nearby_places = filter_places_by_radius(
        all_places,
        current_lat,
        current_lng,
        radius_m,
    )
    nearby_ids = {place["id"] for place in nearby_places}
    ppr_history = [place_id for place_id in selected_history if place_id in nearby_ids]
    all_place_map = {place["id"]: place for place in all_places}
    recent_places = [
        _place_summary(all_place_map[place_id])
        for place_id in selected_history[-4:]
        if place_id in all_place_map
    ]

    print(f"\n=== {title} ===")
    print(f"- 현재 위치: lat={current_lat}, lng={current_lng}")
    print(f"- 후보 반경: {radius_m}m")
    print(f"- 근방 후보 수: {len(nearby_places)}개")
    print(f"- 선택 이력: {selected_history or '없음'}")

    if not nearby_places:
        print("추천 후보가 없습니다.")
        return {
            "title": title,
            "current_location": {
                "latitude": current_lat,
                "longitude": current_lng,
            },
            "recommendations": [],
            "recommendation_log": log_recommendation_event([]),
        }

    graph = build_graph(
        nearby_places,
        max_edge_distance_m=1_500,
        edge_sigma_m=750,
    )
    static_scores = compute_static_scores(
        graph,
        current_lat=current_lat,
        current_lng=current_lng,
    )
    ppr_scores = compute_ppr_scores(graph, ppr_history, decay_rate=0.5)
    recommendations = recommend_next_places_hybrid(
        graph=graph,
        static_scores=static_scores,
        ppr_scores=ppr_scores,
        selected_history=selected_history,
        transition_scorer=transition_scorer,
        recent_places=recent_places,
        current_time=current_time,
        user_preferences=user_preferences,
        weather=weather,
        top_k=top_k,
    )

    ranked_recommendations: list[dict[str, Any]] = []
    for rank, recommendation in enumerate(recommendations, start=1):
        attrs = graph.nodes[recommendation["place_id"]]
        detail = recommendation["detail"]
        ranked_recommendations.append({"rank": rank, **recommendation})
        print(
            f"{rank}. {recommendation['name']} ({recommendation['category']}) "
            f"- 최종: {recommendation['final_score']:.4f} "
            f"/ 기본: {detail['base_score']:.4f} "
            f"/ 태그유사도: {detail['tag_similarity_score']:.4f} "
            f"/ 전이: {detail['transition_score']:.4f} "
            f"/ PPR: {detail['ppr_score']:.4f} "
            f"/ 거리: {detail['distance_m']:.0f}m "
            f"/ 혼잡도: {attrs['congestion']}"
        )
        print(
            f"   [{recommendation['transition_source']}] "
            f"{recommendation['transition_reason']}"
        )

    log_entry = log_recommendation_event(recommendations, selected_place_id=None)
    print(f"- 로그 후보 수: {len(log_entry['candidates'])}개 (label은 선택 전 None)")
    return {
        "title": title,
        "current_location": {
            "latitude": current_lat,
            "longitude": current_lng,
        },
        "selected_history": selected_history,
        "recommendations": ranked_recommendations,
        "recommendation_log": log_entry,
    }


def run_spring_demo_requests(
    requests: list[dict[str, Any]] | None = None,
    transition_scorer: TransitionScorer | None = None,
    similarity_scorer: SimilarityScorer | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    """Spring 요청 예시를 모두 처리하고 한 번의 실행 기록으로 누적 저장한다."""
    result_path = result_path or build_result_path()
    request_items = requests if requests is not None else load_scenario_requests()
    scorer = transition_scorer or create_default_transition_scorer()
    similar_scorer = similarity_scorer or create_default_similarity_scorer()
    responses: list[dict[str, Any]] = []

    print(f"전이 scorer: {scorer.__class__.__name__}")
    for index, request in enumerate(request_items, start=1):
        response = process_spring_recommendation_request(
            request,
            transition_scorer=scorer,
            similarity_scorer=similar_scorer,
        )
        responses.append(response)
        history_count = len(request["recent_places"])
        print(
            f"\n=== 시나리오 {index}: {request['request_id']} "
            f"(이전 선택 {history_count}개) ==="
        )
        print("- 비슷한 장소")
        for place in response["similar_places"]:
            print(
                f"  {place['rank']}. {place['name']} "
                f"(유사도 {place['similarity_score']:.4f}, "
                f"{place['similarity_source']})"
            )
        print("- 다음 장소")
        for place in response["next_places"]:
            print(
                f"  {place['rank']}. {place['name']} "
                f"(최종 {place['final_score']:.4f})"
            )
            print(f"     {place['recommendation_reason']}")

    run = {
        "executed_at": datetime.now().astimezone().isoformat(),
        "scorer": scorer.__class__.__name__,
        "model": getattr(scorer, "model", None),
        "scenario_count": len(responses),
        "scenarios": responses,
    }
    append_result_run(run, result_path)
    print(f"\n{len(responses)}개 시나리오 결과를 {result_path}에 누적했습니다.")
    return run


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_spring_demo_requests()
