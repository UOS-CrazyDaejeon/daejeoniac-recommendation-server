import unittest

from fastapi.testclient import TestClient

from recommendation_api.main import app
from recommendation_api.main import get_natural_search_embedding_provider
from recommendation_api.natural_search import _query_tokens, _search_query_tags, search_places_by_tags


PLACES = [
    {
        "id": 2,
        "name": "대전오월드",
        "description": "동물원과 놀이공원을 함께 즐기는 테마파크",
        "tags": ["동물원", "놀이공원", "사파리", "가족나들이", "야외체험"],
        "categoryLarge": "여행",
        "categoryMedium": "관광,명소",
        "categorySmall": "테마파크",
    },
    {
        "id": 5,
        "name": "으능정이스카이로드",
        "description": "원도심 미디어아트 거리",
        "tags": ["원도심", "미디어아트", "야경", "포토스팟", "데이트"],
        "categoryLarge": "여행",
        "categoryMedium": "관광,명소",
        "categorySmall": "관광지",
    },
    {
        "id": 83,
        "name": "성심당",
        "description": "대전 대표 베이커리",
        "tags": ["베이커리", "빵지순례", "대전명물", "디저트", "원도심"],
        "categoryLarge": "관광지",
        "categoryMedium": "간식",
        "categorySmall": "제과,베이커리",
    },
]


class DeterministicEmbeddingProvider:
    """OpenAI 호출 없이 태그 임베딩 정렬을 검증하는 테스트 제공자."""

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            if text.startswith("검색 의도"):
                vectors.append([1.0, 0.0])
            elif "포근함" in text:
                vectors.append([0.99, 0.01])
            elif "야경" in text:
                vectors.append([0.4, 0.6])
            else:
                vectors.append([0.2, 0.8])
        return vectors


class NaturalSearchTest(unittest.TestCase):
    def test_searches_with_intent_tags_and_returns_match_evidence(self):
        response = search_places_by_tags("아이와 동물을 보고 싶은 곳", PLACES)

        self.assertEqual(response["search_places"][0]["place_id"], 2)
        self.assertIn("동물원", response["search_places"][0]["matched_tags"])
        self.assertIn("가족나들이", response["search_places"][0]["matched_tags"])
        self.assertNotIn("matched_query_tags", response)

    def test_searches_bakery_by_natural_language(self):
        response = search_places_by_tags("따뜻한 분위기에서 빵 먹기 좋은 곳", PLACES)

        self.assertEqual(response["search_places"][0]["place_id"], 83)
        self.assertIn("베이커리", response["search_places"][0]["matched_tags"])

    def test_does_not_treat_similar_as_rainy_day_and_strips_particles(self):
        places = [
            {
                **PLACES[0],
                "id": 4,
                "name": "전시관",
                "tags": ["전시", "미술"],
            },
            PLACES[2],
        ]

        response = search_places_by_tags("주변 비슷한 베이커리집으로 찾아줘", places)

        self.assertEqual(response["total_count"], 1)
        self.assertEqual(response["search_places"][0]["place_id"], 83)
        self.assertIn("베이커리", response["search_places"][0]["matched_tags"])
        self.assertNotIn("전시", response["search_places"][0]["matched_tags"])
        self.assertNotIn("미술전시", _search_query_tags(_query_tokens("비슷한")))

    def test_keeps_explicit_single_syllable_intents(self):
        query_tags = _search_query_tags(_query_tokens("비 오는 날 빵집으로"))

        self.assertIn("미술전시", query_tags)
        self.assertIn("베이커리", query_tags)

    def test_reranks_all_tagged_places_with_embedding_cosine_similarity(self):
        places = [
            {
                **PLACES[0],
                "id": 4,
                "name": "포근한 카페",
                "tags": ["포근함", "카페", "휴식"],
            },
            PLACES[1],
        ]
        response = search_places_by_tags(
            "따뜻한 분위기에서 쉬고 싶은 곳",
            places,
            embedding_provider=DeterministicEmbeddingProvider(),
        )

        self.assertEqual(response["search_places"][0]["place_id"], 4)
        self.assertEqual(
            response["search_places"][0]["semantic_similarity_source"],
            "text_embedding",
        )
        self.assertGreater(
            response["search_places"][0]["embedding_similarity_score"],
            response["search_places"][1]["embedding_similarity_score"],
        )

    def test_endpoint_accepts_spring_page_content_and_snake_case(self):
        app.dependency_overrides[get_natural_search_embedding_provider] = (
            lambda: DeterministicEmbeddingProvider()
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/recommendations/natural-search",
                json={
                    "query": "야경 데이트",
                    "places": {
                        "content": [
                            {
                                "place_id": 5,
                                "place_name": "으능정이스카이로드",
                                "latitude": 36.3289,
                                "longitude": 127.4278,
                                "category_large": "여행",
                                "category_medium": "관광,명소",
                                "category_small": "관광지",
                                "tag": "원도심,미디어아트,야경,포토스팟,데이트",
                            }
                        ]
                    },
                    "top_k": 10,
                },
            )

        try:
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["total_count"], 1)
            self.assertEqual(body["search_places"][0]["place_id"], 5)
            self.assertIn("야경", body["search_places"][0]["matched_tags"])
            self.assertEqual(
                body["search_places"][0]["semantic_similarity_source"],
                "text_embedding",
            )
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
