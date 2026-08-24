import unittest

from recommendation_api.main import app


class KoreanOpenApiTest(unittest.TestCase):
    def test_swagger_metadata_is_written_in_korean(self):
        app.openapi_schema = None
        spec = app.openapi()

        self.assertEqual(spec["info"]["title"], "대전 관광 추천 내부 API")
        self.assertIn("내부 API", spec["info"]["description"])
        self.assertEqual(
            [tag["name"] for tag in spec["tags"]],
            ["운영", "장소 추천", "영수증 분석", "이미지 비식별화"],
        )

        expected_summaries = {
            ("/health", "get"): "서버 상태 확인",
            ("/api/v1/recommendations", "post"): "통합 장소 추천(기존 API)",
            (
                "/api/v1/recommendations/similar-places",
                "post",
            ): "선택 장소와 비슷한 장소 추천",
            (
                "/api/v1/recommendations/next-places",
                "post",
            ): "다음 이동 장소 추천",
            ("/api/v1/receipts/analyze", "post"): "영수증 OCR 분석",
            ("/ocr", "post"): "S3 영수증 OCR 분석(Spring 연동용)",
            (
                "/api/v1/receipts/analyze-gpt-mini",
                "post",
            ): "GPT-5 Mini 영수증 분석",
            (
                "/api/v1/receipts/analyze-from-s3",
                "post",
            ): "S3 영수증 OCR 분석",
            (
                "/api/v1/receipts/s3-read-check",
                "post",
            ): "S3 영수증 객체 읽기 확인",
            (
                "/api/v1/receipts/s3-objects",
                "get",
            ): "S3 영수증 객체 목록 조회",
            (
                "/api/v1/receipts/analyze-gpt-mini-from-s3",
                "post",
            ): "S3 영수증 GPT-5 Mini 분석",
            (
                "/api/v1/images/face-mosaic",
                "post",
            ): "S3 이미지 얼굴 모자이크",
            (
                "/api/v1/images/face-mosaic-local",
                "post",
            ): "로컬 이미지 얼굴 모자이크",
        }
        for (path, method), expected_summary in expected_summaries.items():
            operation = spec["paths"][path][method]
            self.assertEqual(operation["summary"], expected_summary)
            self.assertTrue(operation["description"])

        similar_operation = spec["paths"][
            "/api/v1/recommendations/similar-places"
        ]["post"]
        self.assertEqual(similar_operation["tags"], ["장소 추천"])
        self.assertEqual(
            similar_operation["responses"]["422"]["description"],
            "요청 형식 검증 오류",
        )


if __name__ == "__main__":
    unittest.main()
