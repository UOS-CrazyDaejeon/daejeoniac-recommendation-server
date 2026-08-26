import unittest

from fastapi.testclient import TestClient

from recommendation_api.main import (
    app,
    get_gpt_receipt_processor,
    get_receipt_processor,
)
from recommendation_api.receipt_gpt import ReceiptVisionConfigurationError
from recommendation_api.receipt_ocr import ReceiptDocumentError


def completed_analysis(image_bytes: bytes, language: str):
    assert image_bytes == b"image-bytes"
    assert language == "kor+eng"
    return {
        "result": {
            "merchantName": "카페 파도",
            "businessNumber": "123-45-67890",
            "transactionDate": "2026-08-01",
            "transactionTime": "14:32",
            "address": "대전광역시 유성구 대학로 291",
            "approvalNumber": None,
            "supplyAmount": 7273,
            "vat": 727,
            "totalAmount": 8000,
            "paymentMethod": "신용카드",
            "items": [],
            "confidence": 1.0,
            "warnings": [],
        },
        "rawOcrText": "카페 파도\n합계 8,000원",
    }


def completed_gpt_analysis(image_bytes: bytes, content_type: str):
    assert image_bytes == b"image-bytes"
    assert content_type == "image/jpeg"
    return {
        "result": completed_analysis(image_bytes, "kor+eng")["result"],
        "model": "gpt-5-mini-2025-08-07",
        "processingTimeMs": 1260,
        "usage": {
            "inputTokens": 1100,
            "outputTokens": 220,
            "totalTokens": 1320,
        },
    }


class ReceiptApiTest(unittest.TestCase):
    def tearDown(self):
        app.dependency_overrides.clear()

    def test_analyzes_multipart_receipt_image(self):
        app.dependency_overrides[get_receipt_processor] = lambda: completed_analysis

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze",
                data={
                    "requestId": "req_001",
                    "documentId": "doc_1001",
                    "userId": "4",
                },
                files={"image": ("receipt.jpg", b"image-bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload,
            {
                "ocrStatus": "SUCCESS",
                "ocrPlaceName": "카페 파도",
                "ocrPlaceAddress": "대전광역시 유성구 대학로 291",
                "ocrPaidAt": "2026-08-01T14:32:00",
            },
        )

    def test_returns_safe_content_error(self):
        def fail_analysis(image_bytes: bytes, language: str):
            del image_bytes, language
            raise ReceiptDocumentError("영수증 총액을 확인하지 못했습니다.")

        app.dependency_overrides[get_receipt_processor] = lambda: fail_analysis

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze",
                data={"requestId": "req_002"},
                files={"image": ("receipt.jpg", b"image-bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "RECEIPT_ANALYSIS_FAILED")
        self.assertEqual(response.json()["error"]["request_id"], "req_002")

    def test_rejects_non_image_upload(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze",
                files={"image": ("receipt.txt", b"not-an-image", "text/plain")},
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response.json()["error"]["code"],
            "UNSUPPORTED_RECEIPT_MEDIA_TYPE",
        )

    def test_analyzes_receipt_with_gpt_mini(self):
        app.dependency_overrides[get_gpt_receipt_processor] = (
            lambda: completed_gpt_analysis
        )

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze-gpt-mini",
                data={
                    "requestId": "req_gpt_001",
                    "documentId": "doc_1001",
                    "userId": "4",
                },
                files={"image": ("receipt.jpg", b"image-bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload,
            {
                "ocrStatus": "SUCCESS",
                "ocrPlaceName": "카페 파도",
                "ocrPlaceAddress": "대전광역시 유성구 대학로 291",
                "ocrPaidAt": "2026-08-01T14:32:00",
            },
        )

    def test_returns_503_when_gpt_api_key_is_not_configured(self):
        def fail_analysis(image_bytes: bytes, content_type: str):
            del image_bytes, content_type
            raise ReceiptVisionConfigurationError("OPENAI_API_KEY가 필요합니다.")

        app.dependency_overrides[get_gpt_receipt_processor] = lambda: fail_analysis

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze-gpt-mini",
                data={"requestId": "req_gpt_002"},
                files={"image": ("receipt.jpg", b"image-bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"],
            "RECEIPT_VISION_NOT_CONFIGURED",
        )
        self.assertEqual(response.json()["error"]["request_id"], "req_gpt_002")


if __name__ == "__main__":
    unittest.main()
