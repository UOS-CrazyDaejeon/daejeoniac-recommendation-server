import unittest

from fastapi.testclient import TestClient

from recommend_llm import (
    SimilarityScore,
    TransitionScore,
    process_spring_next_places_request,
    process_spring_recommendation_request,
    process_spring_similar_places_request,
)
from recommendation_api.main import (
    app,
    get_next_places_processor,
    get_similar_places_processor,
)


def place(
    place_id: str,
    name: str,
    latitude: float,
    longitude: float,
    *,
    category: str = "cafe",
) -> dict:
    return {
        "id": place_id,
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "congestion": 35,
        "monthly_visitors": 12000,
        "selected_count": 80,
        "category": category,
        "categoryLarge": "관광",
        "categoryMedium": category,
        "categorySmall": f"{category}-소분류",
        "description": f"조용한 {name}",
        "tags": ["조용한", "로컬", category],
    }


def candidates() -> list[dict]:
    return [
        place(
            f"candidate-{index}",
            f"후보 장소 {index}",
            36.3500 + index * 0.0001,
            127.3800 + index * 0.0001,
            category="cafe" if index <= 5 else "gallery",
        )
        for index in range(1, 11)
    ]


SELECTED_PLACE = place("selected", "선택한 카페", 36.3500, 127.3800)
RECENT_PLACE = place("recent", "최근 서점", 36.3498, 127.3798, category="bookstore")


class DeterministicSimilarityScorer:
    def score_candidates(self, current_place, candidate_rows):
        del current_place
        return {
            candidate["id"]: SimilarityScore(
                score=1.0 - index * 0.03,
                reason="선택 장소와 분위기와 태그가 비슷합니다.",
                source="test_similarity",
                tag_cosine_score=1.0 - index * 0.02,
            )
            for index, candidate in enumerate(candidate_rows)
        }


class DeterministicTransitionScorer:
    def score_candidates(self, context, candidate_rows):
        del context
        return {
            candidate["place_id"]: TransitionScore(
                score=0.8,
                reason="최근 이동 흐름과 자연스럽게 이어집니다.",
                source="test_transition",
                recommendation_reason="다음 장소로 이동하기 좋아요.",
            )
            for candidate in candidate_rows
        }


def similar_request() -> dict:
    return {
        "request_id": "similar-001",
        "session_id": "session-001",
        "selected_place": SELECTED_PLACE,
        "visited_place_ids": [],
        "candidates": candidates(),
        "context": {"radius_m": 1000, "top_k": 5},
    }


def next_request() -> dict:
    return {
        "request_id": "next-001",
        "session_id": "session-001",
        "current_place": SELECTED_PLACE,
        "recent_places": [RECENT_PLACE],
        "visited_place_ids": [RECENT_PLACE["id"]],
        "candidates": candidates(),
        "context": {
            "current_time": "2026-08-15T14:00:00+09:00",
            "weather": "맑음",
            "user_preferences": "조용한 장소",
            "radius_m": 1000,
            "top_k": 5,
        },
    }


class SplitRecommendationProcessorTest(unittest.TestCase):
    def test_similar_processor_only_returns_similar_places(self):
        response = process_spring_similar_places_request(
            similar_request(),
            similarity_scorer=DeterministicSimilarityScorer(),
        )

        self.assertEqual(response["selected_place_id"], "selected")
        self.assertEqual(len(response["similar_places"]), 5)
        self.assertNotIn("next_places", response)
        self.assertEqual(
            response["similar_places"][0]["similarity_source"],
            "test_similarity",
        )
        self.assertEqual(response["similar_places"][0]["categoryLarge"], "관광")
        self.assertEqual(response["similar_places"][0]["categoryMedium"], "cafe")
        self.assertEqual(
            response["similar_places"][0]["categorySmall"], "cafe-소분류"
        )
        self.assertNotIn("category", response["similar_places"][0])

    def test_next_processor_only_returns_next_places(self):
        response = process_spring_next_places_request(
            next_request(),
            transition_scorer=DeterministicTransitionScorer(),
        )

        self.assertEqual(response["current_place_id"], "selected")
        self.assertEqual(response["visited_place_ids"], ["recent"])
        self.assertEqual(len(response["next_places"]), 5)
        self.assertNotIn("similar_places", response)
        self.assertEqual(
            response["next_places"][0]["transition_source"],
            "test_transition",
        )
        self.assertEqual(response["next_places"][0]["categoryLarge"], "관광")
        self.assertEqual(response["next_places"][0]["categoryMedium"], "cafe")
        self.assertEqual(
            response["next_places"][0]["categorySmall"], "cafe-소분류"
        )
        self.assertNotIn("category", response["next_places"][0])

    def test_processors_return_available_candidates_when_fewer_than_five(self):
        similar = similar_request()
        similar["candidates"] = similar["candidates"][:1]
        similar_response = process_spring_similar_places_request(
            similar,
            similarity_scorer=DeterministicSimilarityScorer(),
        )

        next_places = next_request()
        next_places["candidates"] = next_places["candidates"][:1]
        next_response = process_spring_next_places_request(
            next_places,
            transition_scorer=DeterministicTransitionScorer(),
        )

        self.assertEqual(len(similar_response["similar_places"]), 1)
        self.assertEqual(len(next_response["next_places"]), 1)

    def test_legacy_combined_processor_still_returns_both_results(self):
        request = next_request()
        request["request_id"] = "combined-001"
        request["context"]["similar_top_k"] = 5
        request["context"]["next_top_k"] = 5
        request["context"].pop("top_k")

        response = process_spring_recommendation_request(
            request,
            transition_scorer=DeterministicTransitionScorer(),
            similarity_scorer=DeterministicSimilarityScorer(),
        )

        self.assertEqual(len(response["similar_places"]), 5)
        self.assertEqual(len(response["next_places"]), 5)


class SplitRecommendationApiTest(unittest.TestCase):
    def tearDown(self):
        app.dependency_overrides.clear()

    def test_similar_places_endpoint_has_separate_response(self):
        def processor(request):
            return {
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "generated_at": "2026-08-15T14:00:00+09:00",
                "selected_place_id": request["selected_place"]["id"],
                "similar_places": [{"rank": 1, "place_id": "candidate-1"}],
            }

        app.dependency_overrides[get_similar_places_processor] = lambda: processor
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/recommendations/similar-places",
                json=similar_request(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("similar_places", response.json())
        self.assertNotIn("next_places", response.json())

    def test_similar_places_accepts_selected_place_and_nearby_places_dto(self):
        captured = {}

        def processor(request):
            captured.update(request)
            return {
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "generated_at": "2026-08-28T12:00:00+09:00",
                "selected_place_id": request["selected_place"]["id"],
                "similar_places": [{"rank": 1, "place_id": "2"}],
            }

        request = {
            "selectedPlace": {
                "placeId": 1,
                "placeName": "성심당 본점",
                "placeDescription": "대전 대표 베이커리",
                "placeAddress": "대전광역시 중구 대종로480번길 15",
                "latitude": 36.3275,
                "longitude": 127.4272,
                "gu": "중구",
                "dong": "은행동",
                "categoryLarge": "음식",
                "categoryMedium": "카페/디저트",
                "categorySmall": "베이커리",
                "congestionRate": 72.5,
                "visitorCount": 1320,
                "visitedAt": None,
                "description": "대표 빵집",
                "tag": "빵,디저트,대전",
            },
            "nearbyPlaces": [
                {
                    "placeId": 2,
                    "placeName": "중앙로 지하상가",
                    "placeDescription": "대전 원도심 쇼핑 공간",
                    "placeAddress": "대전광역시 중구 중앙로",
                    "latitude": 36.3281,
                    "longitude": 127.4265,
                    "gu": "중구",
                    "dong": "은행동",
                    "categoryLarge": "쇼핑",
                    "categoryMedium": "상가",
                    "categorySmall": "지하상가",
                    "congestionRate": 61.0,
                    "visitorCount": 980,
                    "visitedAt": None,
                    "description": "원도심 쇼핑",
                    "tag": "쇼핑,원도심,대전",
                }
            ],
        }

        app.dependency_overrides[get_similar_places_processor] = lambda: processor
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/recommendations/similar-places",
                json=request,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["selected_place"]["id"], "1")
        self.assertEqual(captured["selected_place"]["congestion"], 72.5)
        self.assertEqual(captured["selected_place"]["monthly_visitors"], 1320)
        self.assertEqual(captured["selected_place"]["tags"], ["빵", "디저트", "대전"])
        self.assertEqual(captured["candidates"][0]["id"], "2")
        self.assertEqual(captured["candidates"][0]["tags"], ["쇼핑", "원도심", "대전"])

    def test_next_places_endpoint_has_separate_response(self):
        def processor(request):
            return {
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "generated_at": "2026-08-15T14:00:00+09:00",
                "current_place_id": request["current_place"]["id"],
                "visited_place_ids": request["visited_place_ids"],
                "next_places": [{"rank": 1, "place_id": "candidate-1"}],
                "recommendation_log": {"items": []},
            }

        app.dependency_overrides[get_next_places_processor] = lambda: processor
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/recommendations/next-places",
                json=next_request(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("next_places", response.json())
        self.assertNotIn("similar_places", response.json())
        self.assertNotIn("recommendation_log", response.json())

    def test_next_places_accepts_selected_nearby_and_visited_places_dto(self):
        captured = {}

        def processor(request):
            captured.update(request)
            return {
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "generated_at": "2026-08-28T12:00:00+09:00",
                "current_place_id": request["current_place"]["id"],
                "visited_place_ids": request["visited_place_ids"],
                "next_places": [{"rank": 1, "place_id": "2"}],
                "recommendation_log": {"items": []},
            }

        request = {
            "selectedPlace": {
                "placeId": 1,
                "placeName": "성심당 본점",
                "placeDescription": "대전 대표 베이커리",
                "latitude": 36.3275,
                "longitude": 127.4272,
                "categorySmall": "베이커리",
                "congestionRate": 72.5,
                "visitorCount": 1320,
                "tag": "빵,디저트,대전",
            },
            "nearbyPlaces": [
                {
                    "placeId": 2,
                    "placeName": "중앙로 지하상가",
                    "placeDescription": "대전 원도심 쇼핑 공간",
                    "latitude": 36.3281,
                    "longitude": 127.4265,
                    "categorySmall": "지하상가",
                    "congestionRate": 61.0,
                    "visitorCount": 980,
                    "tag": "쇼핑,원도심,대전",
                }
            ],
            "visitedPlaces": [
                {
                    "placeId": 10,
                    "placeName": "한밭수목원",
                    "placeDescription": "도심 속 대형 수목원",
                    "latitude": 36.3662,
                    "longitude": 127.3882,
                    "categorySmall": "수목원",
                    "visitedAt": "2026-08-28T14:30:00",
                    "tag": "자연,산책,대전",
                }
            ],
        }

        app.dependency_overrides[get_next_places_processor] = lambda: processor
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/recommendations/next-places",
                json=request,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["current_place"]["id"], "1")
        self.assertEqual(captured["candidates"][0]["id"], "2")
        self.assertEqual(captured["visited_place_ids"], ["10"])
        self.assertEqual(captured["recent_places"][0]["id"], "10")
        self.assertTrue(captured["context"]["current_time"])

    def test_accepts_spring_page_candidates_and_uses_content_only(self):
        captured = {}

        def processor(request):
            captured.update(request)
            return {
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "generated_at": "2026-08-15T14:00:00+09:00",
                "selected_place_id": request["selected_place"]["id"],
                "similar_places": [{"rank": 1, "place_id": "8"}],
            }

        request = similar_request()
        request["candidates"] = {
            "content": [
                {
                    "placeId": 8,
                    "placeName": "대전오월드 주랜드",
                    "placeDescription": None,
                    "placeAddress": "대전 중구 사정동 100",
                    "latitude": 36.2890919504445,
                    "longitude": 127.400486744411,
                    "gu": "중구",
                    "dong": "사정동",
                    "categoryLarge": "여행",
                    "categoryMedium": "관광,명소",
                    "categorySmall": "동물원",
                },
                {
                    "placeId": 24,
                    "placeName": "뿌리공원",
                    "placeDescription": "전국 유일의 ‘효’ 테마공원",
                    "placeAddress": "대전 중구 뿌리공원로 79",
                    "latitude": 36.28538,
                    "longitude": 127.3883,
                    "gu": "중구",
                    "dong": "뿌리공원로",
                    "categoryLarge": "관광지",
                    "categoryMedium": None,
                    "categorySmall": None,
                },
            ],
            "number": 0,
            "size": 10,
            "totalElements": 2,
        }

        app.dependency_overrides[get_similar_places_processor] = lambda: processor
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/recommendations/similar-places",
                json=request,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(captured["candidates"]), 2)
        self.assertEqual(captured["candidates"][0]["id"], "8")
        self.assertEqual(captured["candidates"][0]["name"], "대전오월드 주랜드")
        self.assertEqual(captured["candidates"][0]["category"], "동물원")
        self.assertEqual(
            captured["candidates"][0]["tags"],
            ["여행", "관광,명소", "동물원", "중구", "사정동"],
        )


if __name__ == "__main__":
    unittest.main()
