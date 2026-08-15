from io import BytesIO
import json
from types import SimpleNamespace
import unittest

from PIL import Image

from recommendation_api.receipt_gpt import (
    ReceiptVisionUpstreamError,
    analyze_receipt_image_with_gpt,
    normalize_receipt_image_for_vision,
)
from recommendation_api.receipt_ocr import ReceiptDocumentError


def image_bytes(image_format: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (80, 120), color="white").save(output, format=image_format)
    return output.getvalue()


def valid_receipt_result() -> dict:
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
        "confidence": 0.93,
        "warnings": [],
    }


class RecordingResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class RecordingClient:
    def __init__(self, response):
        self.responses = RecordingResponses(response)


class ReceiptGptTest(unittest.TestCase):
    def test_calls_gpt_5_mini_with_jpeg_and_structured_output(self):
        response = SimpleNamespace(
            output_text=json.dumps(valid_receipt_result(), ensure_ascii=False),
            model="gpt-5-mini-2025-08-07",
            usage=SimpleNamespace(
                input_tokens=1100,
                output_tokens=220,
                total_tokens=1320,
            ),
        )
        client = RecordingClient(response)

        analysis = analyze_receipt_image_with_gpt(
            image_bytes(),
            "image/png",
            client=client,
            model="gpt-5-mini",
            detail="high",
        )

        self.assertEqual(analysis["result"]["totalAmount"], 8000)
        self.assertEqual(analysis["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(analysis["usage"]["totalTokens"], 1320)
        self.assertGreaterEqual(analysis["processingTimeMs"], 0)

        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-5-mini")
        self.assertFalse(call["store"])
        self.assertTrue(call["text"]["format"]["strict"])
        image_input = call["input"][0]["content"][1]
        self.assertEqual(image_input["detail"], "high")
        self.assertTrue(image_input["image_url"].startswith("data:image/jpeg;base64,"))

    def test_normalizes_heif_to_jpeg(self):
        normalized = normalize_receipt_image_for_vision(image_bytes("HEIF"))

        with Image.open(BytesIO(normalized)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (80, 120))

    def test_rejects_result_without_total_amount(self):
        result = valid_receipt_result()
        result["totalAmount"] = None
        client = RecordingClient(
            SimpleNamespace(
                output_text=json.dumps(result, ensure_ascii=False),
                model="gpt-5-mini",
                usage=None,
            )
        )

        with self.assertRaises(ReceiptDocumentError):
            analyze_receipt_image_with_gpt(
                image_bytes(),
                client=client,
                model="gpt-5-mini",
                detail="high",
            )

    def test_rejects_malformed_model_output(self):
        client = RecordingClient(
            SimpleNamespace(output_text="not-json", model="gpt-5-mini", usage=None)
        )

        with self.assertRaises(ReceiptVisionUpstreamError):
            analyze_receipt_image_with_gpt(
                image_bytes(),
                client=client,
                model="gpt-5-mini",
                detail="high",
            )


if __name__ == "__main__":
    unittest.main()
