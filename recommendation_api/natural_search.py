"""자연어를 검색 의도 태그로 바꾼 뒤 임베딩으로 검색하는 기능."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from recommend_llm import EmbeddingProvider, clamp, cosine_similarity


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
_STOP_WORDS = {
    "가고",
    "가고싶어",
    "갈",
    "곳",
    "나는",
    "내가",
    "에서",
    "으로",
    "있는",
    "좋은",
    "추천",
    "추천해줘",
    "장소",
    "하고",
    "할",
    "싶어",
    "싶은",
    "찾아줘",
    "해줘",
    "좀",
}

# 자연어 표현을 데이터베이스 tag 값으로 연결한다. 값은 반드시 실제 tag에 넣을 수
# 있는 짧은 명사형으로 유지해, 매칭 근거를 응답에서 이해하기 쉽게 한다.
_INTENT_TAGS: dict[str, tuple[str, ...]] = {
    "따뜻": ("따뜻함", "포근함", "아늑함", "카페", "휴식", "힐링"),
    "포근": ("포근함", "아늑함", "따뜻함", "카페", "휴식"),
    "아늑": ("아늑함", "포근함", "카페", "휴식"),
    "조용": ("조용한휴식", "도서관", "미술전시", "숲산책", "힐링"),
    "데이트": ("데이트", "야경", "포토스팟", "카페", "미술전시"),
    "야경": ("야경", "전망대", "포토스팟", "데이트"),
    "사진": ("포토스팟", "야경", "메타세쿼이아", "벽화마을", "전망"),
    "아이": ("가족나들이", "어린이체험", "동물원", "체험", "야외체험"),
    "가족": ("가족나들이", "어린이체험", "체험", "동물원", "공원"),
    "산책": ("산책", "숲산책", "하천산책", "도심산책", "공원"),
    "걷기": ("걷기여행", "숲산책", "맨발걷기", "트레킹", "둘레길"),
    "등산": ("등산", "트레킹", "전망", "숲산책"),
    "힐링": ("힐링", "휴식", "숲치유", "산림욕", "조용한휴식"),
    "자연": ("자연휴식", "도심자연", "숲산책", "호수풍경", "생태공원"),
    "캠핑": ("캠핑", "야외숙박", "가족캠핑", "숲캠핑"),
    "과학": ("과학기술", "과학체험", "과학관광", "천문관측", "교육체험"),
    "전시": ("전시", "미술전시", "전시관람", "예술감상", "박물관"),
    "역사": ("역사탐방", "문화유산", "근대건축", "전통건축", "박물관"),
    "빵": ("베이커리", "빵지순례", "디저트", "대전명물"),
    "디저트": ("디저트", "베이커리", "카페", "빵지순례"),
    "카페": ("카페", "커피", "휴식", "대화"),
    "맛집": ("로컬맛집", "동네맛집", "한식", "식사"),
    "쇼핑": ("쇼핑", "백화점", "전통시장", "실내쇼핑", "장보기"),
    "시장": ("전통시장", "오일장", "지역먹거리", "장보기"),
    "운동": ("스포츠", "운동", "자전거", "러닝", "수상레저"),
    "비": ("실내체험", "실내관람", "박물관", "미술전시", "카페"),
}


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.lower())


def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(query.lower()):
        token = _normalize(raw_token)
        # "빵", "비"처럼 한 글자여도 의도 사전에 정의한 검색어는 보존한다.
        if (len(token) < 2 and token not in _INTENT_TAGS) or token in _STOP_WORDS:
            continue
        tokens.append(token)
        # 자주 쓰는 한국어 관형형/조사만 제거한다. 형태소 분석기가 없어도
        # "따뜻한"과 "따뜻함" 같은 태그를 연결하기 위한 최소 보정이다.
        for suffix in ("하는", "한", "은", "인", "을", "를", "에", "와", "과"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                tokens.append(token[: -len(suffix)])
                break
    return list(dict.fromkeys(tokens))


def _intent_tags(query_tokens: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for token in query_tokens:
        for intent, tags in _INTENT_TAGS.items():
            if intent in token or token in intent:
                expanded.extend(tags)
    return list(dict.fromkeys(expanded))


def _search_query_tags(query_tokens: Sequence[str]) -> list[str]:
    """원문에서 얻은 핵심어와 동의어 확장 태그를 하나의 검색 프로필로 만든다."""
    return list(dict.fromkeys([*query_tokens, *_intent_tags(query_tokens)]))


def _tag_profile(tags: Sequence[str], *, label: str) -> str:
    """임베딩에 넣을 태그 전용 텍스트를 만든다.

    장소명·설명을 섞지 않아, 검색 점수가 사용자가 관리하는 tag 값에만 근거하도록
    한다.
    """
    return f"{label}: {', '.join(tags)}"


def _tag_match_score(tag: str, query_tokens: Sequence[str], expanded_tags: Sequence[str]) -> float:
    normalized_tag = _normalize(tag)
    if not normalized_tag:
        return 0.0

    score = 0.0
    for token in query_tokens:
        if token == normalized_tag:
            score = max(score, 4.0)
        elif len(token) >= 2 and (token in normalized_tag or normalized_tag in token):
            score = max(score, 3.0)
    for intent_tag in expanded_tags:
        normalized_intent_tag = _normalize(intent_tag)
        if normalized_intent_tag == normalized_tag:
            score = max(score, 2.0)
        elif len(normalized_intent_tag) >= 2 and (
            normalized_intent_tag in normalized_tag
            or normalized_tag in normalized_intent_tag
        ):
            score = max(score, 1.5)
    return score


def search_places_by_tags(
    query: str,
    places: Sequence[dict[str, Any]],
    *,
    top_k: int = 5,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """태그 프로필 임베딩 코사인 유사도로 자연어 장소를 검색한다.

    임베딩 제공자가 없거나 호출에 실패하면, 기존의 태그 직접 매칭으로 안전하게
    대체한다. 이 경우에도 응답 형식은 동일하다.
    """
    query_tokens = _query_tokens(query)
    query_tags = _search_query_tags(query_tokens)
    candidate_rows: list[dict[str, Any]] = []

    for place in places:
        tags = [str(tag).strip() for tag in place.get("tags", []) if str(tag).strip()]
        if not tags:
            continue
        matched_tags: list[str] = []
        tag_score = 0.0
        for tag in tags:
            score = _tag_match_score(tag, query_tokens, query_tags)
            if score > 0:
                matched_tags.append(tag)
                tag_score += score
        candidate_rows.append(
            {
                "place_id": int(place["id"]),
                "name": str(place["name"]),
                "categoryLarge": place.get("categoryLarge"),
                "categoryMedium": place.get("categoryMedium"),
                "categorySmall": place.get("categorySmall"),
                "description": str(place.get("description") or place["name"]),
                "tags": tags,
                "matched_tags": matched_tags,
                "tag_score": round(tag_score, 3),
            }
        )

    semantic_similarity_by_id: dict[int, float] = {}
    semantic_similarity_source = "tag_match_fallback"
    if embedding_provider is not None and query_tags and candidate_rows:
        texts = [
            _tag_profile(query_tags, label="검색 의도 태그"),
            *[
                _tag_profile(candidate["tags"], label="장소 태그")
                for candidate in candidate_rows
            ],
        ]
        try:
            vectors = embedding_provider.embed_texts(texts)
            if len(vectors) != len(texts):
                raise RuntimeError("임베딩 벡터 개수가 입력과 일치하지 않습니다")
            query_vector = vectors[0]
            semantic_similarity_by_id = {
                candidate["place_id"]: clamp(cosine_similarity(query_vector, vector))
                for candidate, vector in zip(candidate_rows, vectors[1:], strict=True)
            }
            semantic_similarity_source = "text_embedding"
        except Exception:
            # 호출 실패는 자연어 검색 전체 오류가 아니라 직접 태그 매칭으로 처리한다.
            semantic_similarity_by_id = {}

    if semantic_similarity_by_id:
        results = [
            {
                **candidate,
                "embedding_similarity_score": round(
                    semantic_similarity_by_id[candidate["place_id"]], 6
                ),
                "similarity_score": round(
                    semantic_similarity_by_id[candidate["place_id"]], 6
                ),
                "semantic_similarity_source": semantic_similarity_source,
            }
            for candidate in candidate_rows
        ]
        results.sort(
            key=lambda item: (-item["similarity_score"], item["place_id"])
        )
    else:
        results = [
            {
                **candidate,
                "embedding_similarity_score": None,
                "similarity_score": candidate["tag_score"],
                "semantic_similarity_source": semantic_similarity_source,
            }
            for candidate in candidate_rows
            if candidate["tag_score"] > 0
        ]
        results.sort(key=lambda item: (-item["similarity_score"], item["place_id"]))

    return {
        "query": query,
        "total_count": len(results),
        "search_places": [
            {"rank": rank, **result}
            for rank, result in enumerate(results[:top_k], start=1)
        ],
    }
