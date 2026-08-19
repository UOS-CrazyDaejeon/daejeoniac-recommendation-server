from io import BytesIO
import os
from unittest import mock
import unittest

from fastapi.testclient import TestClient
from botocore.exceptions import ClientError

from recommendation_api.main import (
    app,
    get_gpt_receipt_processor,
    get_receipt_processor,
    get_s3_receipt_loader,
)
from recommendation_api.receipt_s3 import (
    S3ReceiptKeyError,
    S3ReceiptNotFoundError,
    S3ReceiptObject,
    S3ReceiptSettings,
    S3ReceiptTooLargeError,
    load_s3_receipt_object,
)


class ClosingBody:
    def __init__(self, value: bytes):
        self.stream = BytesIO(value)
        self.closed = False

    def read(self, amount: int) -> bytes:
        return self.stream.read(amount)

    def close(self) -> None:
        self.closed = True
        self.stream.close()


class RecordingS3Client:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FailingS3Client:
    def __init__(self, error: Exception):
        self.error = error

    def get_object(self, **kwargs):
        del kwargs
        raise self.error


def settings(prefix: str | None = "receipts/") -> S3ReceiptSettings:
    return S3ReceiptSettings(
        region="ap-northeast-2",
        bucket="receipt-test-bucket",
        allowed_prefix=prefix,
        expected_bucket_owner="521701612202",
    )


def receipt_result() -> dict:
    return {
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
        "confidence": 0.95,
        "warnings": [],
    }


class S3ReceiptLoaderTest(unittest.TestCase):
    def test_reads_dotted_environment_setting_names(self):
        dotted_environment = {
            "aws.region": "ap-northeast-2",
            "aws.access-key": "test-access-key",
            "aws.secret-key": "test-secret-key",
            "aws.s3.bucket": "receipt-test-bucket",
        }

        with mock.patch.dict(os.environ, dotted_environment, clear=True):
            loaded = S3ReceiptSettings.from_environment()

        self.assertEqual(loaded.region, "ap-northeast-2")
        self.assertEqual(loaded.bucket, "receipt-test-bucket")
        self.assertEqual(loaded.access_key_id, "test-access-key")
        self.assertEqual(loaded.secret_access_key, "test-secret-key")

    def test_loads_only_the_configured_bucket_and_key(self):
        body = ClosingBody(b"image-bytes")
        client = RecordingS3Client(
            {
                "Body": body,
                "ContentLength": len(b"image-bytes"),
                "ContentType": "image/heic",
            }
        )

        receipt = load_s3_receipt_object(
            "receipts/2026/receipt.heic",
            max_bytes=1024,
            client=client,
            settings=settings(),
        )

        self.assertEqual(receipt.image_bytes, b"image-bytes")
        self.assertEqual(receipt.content_type, "image/heic")
        self.assertEqual(
            client.calls,
            [
                {
                    "Bucket": "receipt-test-bucket",
                    "Key": "receipts/2026/receipt.heic",
                    "ExpectedBucketOwner": "521701612202",
                }
            ],
        )
        self.assertTrue(body.closed)

    def test_rejects_url_instead_of_object_key(self):
        client = RecordingS3Client({})

        with self.assertRaises(S3ReceiptKeyError):
            load_s3_receipt_object(
                "https://example.com/receipt.jpg",
                max_bytes=1024,
                client=client,
                settings=settings(),
            )

        self.assertEqual(client.calls, [])

    def test_rejects_key_outside_allowed_prefix(self):
        with self.assertRaises(S3ReceiptKeyError):
            load_s3_receipt_object(
                "private/receipt.jpg",
                max_bytes=1024,
                client=RecordingS3Client({}),
                settings=settings(),
            )

    def test_stops_before_reading_oversized_object(self):
        body = ClosingBody(b"large-image")
        client = RecordingS3Client(
            {
                "Body": body,
                "ContentLength": 2048,
                "ContentType": "image/jpeg",
            }
        )

        with self.assertRaises(S3ReceiptTooLargeError):
            load_s3_receipt_object(
                "receipts/large.jpg",
                max_bytes=1024,
                client=client,
                settings=settings(),
            )

        self.assertTrue(body.closed)

    def test_maps_s3_no_such_key_to_not_found(self):
        error = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "not found"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )

        with self.assertRaises(S3ReceiptNotFoundError):
            load_s3_receipt_object(
                "receipts/missing.jpg",
                max_bytes=1024,
                client=FailingS3Client(error),
                settings=settings(),
            )


class S3ReceiptApiTest(unittest.TestCase):
    def tearDown(self):
        app.dependency_overrides.clear()

    def test_analyzes_s3_receipt_with_ocr(self):
        def loader(s3_key: str):
            self.assertEqual(s3_key, "receipts/receipt-001.heic")
            return S3ReceiptObject(s3_key, "image/heic", b"image-bytes")

        def processor(image_bytes: bytes, language: str):
            self.assertEqual(image_bytes, b"image-bytes")
            self.assertEqual(language, "kor+eng")
            return {"result": receipt_result(), "rawOcrText": "합계 8,000원"}

        app.dependency_overrides[get_s3_receipt_loader] = lambda: loader
        app.dependency_overrides[get_receipt_processor] = lambda: processor

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze-from-s3",
                json={
                    "requestId": "req-s3-001",
                    "documentId": "doc-001",
                    "userId": 4,
                    "s3Key": "receipts/receipt-001.heic",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["totalAmount"], 8000)
        self.assertEqual(response.json()["requestId"], "req-s3-001")

    def test_analyzes_s3_receipt_with_gpt_mini(self):
        def loader(s3_key: str):
            return S3ReceiptObject(s3_key, "image/jpeg", b"image-bytes")

        def processor(image_bytes: bytes, content_type: str):
            self.assertEqual(image_bytes, b"image-bytes")
            self.assertEqual(content_type, "image/jpeg")
            return {
                "result": receipt_result(),
                "model": "gpt-5-mini-2025-08-07",
                "processingTimeMs": 900,
                "usage": None,
            }

        app.dependency_overrides[get_s3_receipt_loader] = lambda: loader
        app.dependency_overrides[get_gpt_receipt_processor] = lambda: processor

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze-gpt-mini-from-s3",
                json={"s3Key": "receipts/receipt-001.jpg"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "gpt-5-mini-2025-08-07")

    def test_returns_not_found_when_s3_object_is_missing(self):
        def loader(s3_key: str):
            del s3_key
            raise S3ReceiptNotFoundError(
                "S3에서 영수증 이미지를 찾지 못했습니다."
            )

        app.dependency_overrides[get_s3_receipt_loader] = lambda: loader

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/analyze-from-s3",
                json={"requestId": "req-s3-404", "s3Key": "receipts/missing.jpg"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "S3_RECEIPT_NOT_FOUND")
        self.assertEqual(response.json()["error"]["request_id"], "req-s3-404")


if __name__ == "__main__":
    unittest.main()
