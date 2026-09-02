from datetime import datetime, timezone
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
    get_s3_receipt_object_lister,
    get_s3_receipt_loader,
)
from recommendation_api.receipt_callback import (
    ReceiptOcrCallbackError,
    build_ocr_result_callback_payload,
    post_ocr_result_callback,
)
from recommendation_api.receipt_s3 import (
    S3ReceiptKeyError,
    S3ReceiptNotFoundError,
    S3ReceiptObject,
    S3ReceiptObjectList,
    S3ReceiptObjectSummary,
    S3ReceiptSettings,
    S3ReceiptTooLargeError,
    load_s3_receipt_object,
    list_s3_receipt_objects,
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


class ListingS3Client:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def list_objects_v2(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class RecordingCallbackClient:
    def __init__(self):
        self.calls = []

    def post(self, url, *, json, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self

    def raise_for_status(self):
        return None


class FailingCallbackClient(RecordingCallbackClient):
    def raise_for_status(self):
        raise RuntimeError("Spring callback failed")


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

    def test_lists_only_receipt_prefix_metadata(self):
        client = ListingS3Client(
            {
                "Contents": [
                    {
                        "Key": "receipts/2026/08/receipt-001.heic",
                        "Size": 1234,
                        "LastModified": datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
                    }
                ]
            }
        )

        result = list_s3_receipt_objects(
            max_keys=20,
            client=client,
            settings=settings(),
        )

        self.assertEqual(result.prefix, "receipts/")
        self.assertEqual(result.objects[0].key, "receipts/2026/08/receipt-001.heic")
        self.assertEqual(result.objects[0].size_bytes, 1234)
        self.assertEqual(
            client.calls,
            [
                {
                    "Bucket": "receipt-test-bucket",
                    "Prefix": "receipts/",
                    "MaxKeys": 20,
                    "ExpectedBucketOwner": "521701612202",
                }
            ],
        )


class OcrResultCallbackTest(unittest.TestCase):
    def test_builds_spring_ocr_result_contract(self):
        payload = build_ocr_result_callback_payload(
            "receipt-uuid-001",
            {
                "merchantName": "카페 파도",
                "address": "대전광역시 유성구 대학로 291",
                "transactionDate": "2026-08-23",
                "transactionTime": "14:32",
            },
        )

        self.assertEqual(
            payload,
            {
                "receiptUuid": "receipt-uuid-001",
                "ocrStatus": "SUCCESS",
                "ocrPlaceName": "카페 파도",
                "ocrPlaceAddress": "대전광역시 유성구 대학로 291",
                "ocrPaidAt": "2026-08-23T14:32:00",
            },
        )

    def test_posts_callback_to_configured_spring_endpoint(self):
        client = RecordingCallbackClient()
        payload = {"receiptUuid": "receipt-uuid-001", "ocrStatus": "COMPLETED"}
        environment = {
            "SPRING_OCR_CALLBACK_URL": "http://spring-api:8080/api/v1/receipts/ocr-result",
            "SPRING_OCR_CALLBACK_AUTHORIZATION": "Bearer internal-token",
        }

        with mock.patch.dict(os.environ, environment, clear=True):
            post_ocr_result_callback(payload, client=client)

        self.assertEqual(
            client.calls,
            [
                {
                    "url": "http://spring-api:8080/api/v1/receipts/ocr-result",
                    "json": payload,
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer internal-token",
                    },
                }
            ],
        )

    def test_raises_safe_error_when_spring_callback_fails(self):
        with mock.patch.dict(
            os.environ,
            {"SPRING_OCR_CALLBACK_URL": "http://spring-api:8080/api/v1/receipts/ocr-result"},
            clear=True,
        ):
            with self.assertRaises(ReceiptOcrCallbackError):
                post_ocr_result_callback({}, client=FailingCallbackClient())


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
        self.assertEqual(
            response.json(),
            {
                "ocrStatus": "SUCCESS",
                "ocrPlaceName": "카페 파도",
                "ocrPlaceAddress": "대전광역시 유성구 대학로 291",
                "ocrPaidAt": "2026-08-01T14:32:00",
            },
        )

    def test_reads_s3_receipt_without_running_ocr(self):
        def loader(object_key: str):
            self.assertEqual(object_key, "receipts/2026/08/receipt-001.heic")
            return S3ReceiptObject(object_key, "image/heic", b"image-bytes")

        app.dependency_overrides[get_s3_receipt_loader] = lambda: loader

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/s3-read-check",
                json={
                    "requestId": "req-s3-read-check-001",
                    "objectKey": "receipts/2026/08/receipt-001.heic",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "requestId": "req-s3-read-check-001",
                "status": "AVAILABLE",
                "objectKey": "receipts/2026/08/receipt-001.heic",
                "contentType": "image/heic",
                "sizeBytes": len(b"image-bytes"),
            },
        )

    def test_lists_s3_receipt_metadata(self):
        result = S3ReceiptObjectList(
            prefix="receipts/",
            objects=[
                S3ReceiptObjectSummary(
                    key="receipts/2026/08/receipt-001.heic",
                    size_bytes=1234,
                    last_modified=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
                )
            ],
        )
        app.dependency_overrides[get_s3_receipt_object_lister] = lambda: (
            lambda max_keys: result
        )

        with TestClient(app) as client:
            response = client.get("/api/v1/receipts/s3-objects?maxKeys=20")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "AVAILABLE")
        self.assertEqual(response.json()["prefix"], "receipts/")
        self.assertEqual(response.json()["objectCount"], 1)
        self.assertEqual(
            response.json()["objects"][0]["objectKey"],
            "receipts/2026/08/receipt-001.heic",
        )

    def test_returns_s3_error_from_read_check(self):
        def loader(object_key: str):
            del object_key
            raise S3ReceiptNotFoundError(
                "S3에서 영수증 이미지를 찾지 못했습니다."
            )

        app.dependency_overrides[get_s3_receipt_loader] = lambda: loader

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/receipts/s3-read-check",
                json={
                    "requestId": "req-s3-read-check-404",
                    "objectKey": "receipts/missing.heic",
                },
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "S3_RECEIPT_NOT_FOUND")
        self.assertEqual(
            response.json()["error"]["request_id"],
            "req-s3-read-check-404",
        )

    def test_analyzes_s3_receipt_with_spring_ocr_contract(self):
        def loader(object_key: str):
            self.assertEqual(object_key, "receipts/receipt-001.heic")
            return S3ReceiptObject(object_key, "image/heic", b"image-bytes")

        def processor(image_bytes: bytes, language: str):
            self.assertEqual(image_bytes, b"image-bytes")
            self.assertEqual(language, "kor+eng")
            return {"result": receipt_result(), "rawOcrText": "합계 8,000원"}

        app.dependency_overrides[get_s3_receipt_loader] = lambda: loader
        app.dependency_overrides[get_receipt_processor] = lambda: processor

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/ocr",
                json={
                    "receiptUuid": "receipt-uuid-001",
                    "objectKey": "receipts/receipt-001.heic",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "receiptUuid": "receipt-uuid-001",
                "ocrStatus": "SUCCESS",
                "ocrPlaceName": "카페 파도",
                "ocrPlaceAddress": "대전광역시 유성구 대학로 291",
                "ocrPaidAt": "2026-08-01T14:32:00",
            },
        )

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
        self.assertEqual(
            response.json(),
            {
                "ocrStatus": "SUCCESS",
                "ocrPlaceName": "카페 파도",
                "ocrPlaceAddress": "대전광역시 유성구 대학로 291",
                "ocrPaidAt": "2026-08-01T14:32:00",
            },
        )

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
