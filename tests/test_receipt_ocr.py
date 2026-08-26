from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from recommendation_api.receipt_ocr import (
    ReceiptDocumentError,
    _crop_and_rectify_receipt,
    _load_image_variants,
    extract_receipt_text_from_image_bytes,
    parse_receipt_text,
)


class ReceiptParserTest(unittest.TestCase):
    def test_loads_heif_image_bytes_for_ocr(self):
        source = BytesIO()
        Image.new("RGB", (32, 24), color="white").save(source, format="HEIF")

        variants = _load_image_variants(source.getvalue())

        self.assertEqual(
            [name for name, _ in variants],
            ["original", "contrast", "binary"],
        )
        self.assertEqual(variants[0][1].mode, "RGB")
        self.assertEqual(variants[0][1].size, (32, 24))

    def test_crops_and_rectifies_detected_receipt(self):
        import cv2

        canvas = np.zeros((420, 420, 3), dtype=np.uint8)
        corners = np.array([[75, 45], [345, 85], [300, 370], [95, 330]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, corners, (255, 255, 255))
        cv2.polylines(canvas, [corners], True, (220, 220, 220), 4)

        receipt = _crop_and_rectify_receipt(Image.fromarray(canvas))

        self.assertGreaterEqual(receipt.width, 1200)
        self.assertGreater(receipt.height, 900)
        self.assertLess(receipt.height, 1600)

    def test_extracts_structured_receipt_fields(self):
        result = parse_receipt_text(
            """
            카페 파도
            사업자등록번호 123-45-67890
            주소 대전광역시 유성구 대학로 291 (궁동) 3층
            2026-08-01 14:32
            아메리카노 2 8,000원
            공급가액 7,273원
            부가세 727원
            합계 8,000원
            신용카드
            """
        )

        self.assertEqual(result["merchantName"], "카페 파도")
        self.assertEqual(result["businessNumber"], "123-45-67890")
        self.assertEqual(result["transactionDate"], "2026-08-01")
        self.assertEqual(result["transactionTime"], "14:32")
        self.assertEqual(result["supplyAmount"], 7273)
        self.assertEqual(result["vat"], 727)
        self.assertEqual(result["totalAmount"], 8000)
        self.assertEqual(result["paymentMethod"], "신용카드")
        self.assertEqual(result["warnings"], [])

    def test_prefers_store_name_near_business_number_over_card_details(self):
        result = parse_receipt_text(
            """
            [영수증]
            청화원(대학로) / 637-34-00700 / 길영남
            서울 종로구 대명길 36 (명륜2가) 1,2층
            2025-06-11 18:35:36
            합계 금액 68,000원
            할부 개월: 일시불
            """
        )

        self.assertEqual(result["merchantName"], "청화원(대학로)")
        self.assertEqual(result["address"], "서울 종로구 대명길 36 (명륜2가) 1,2층")
        self.assertEqual(result["transactionDate"], "2025-06-11")
        self.assertEqual(result["transactionTime"], "18:35:36")

    def test_requires_total_amount(self):
        with self.assertRaises(ReceiptDocumentError):
            parse_receipt_text("카페 파도\n2026-08-01\n아메리카노")

    def test_uses_fast_path_when_basic_ocr_is_sufficient(self):
        receipt_text = """
        카페 파도
        사업자등록번호 123-45-67890
        주소 대전광역시 유성구 대학로 291
        2026-08-01 14:32
        공급가액 7,273원
        부가세 727원
        합계 8,000원
        """
        calls: list[int] = []

        def run_tesseract(_image, *, language, psm):
            calls.append(psm)
            return receipt_text

        with (
            patch(
                "recommendation_api.receipt_ocr._load_image_variants",
                return_value=[("original", object()), ("contrast", object()), ("binary", object())],
            ),
            patch(
                "recommendation_api.receipt_ocr._run_tesseract",
                side_effect=run_tesseract,
            ),
        ):
            text = extract_receipt_text_from_image_bytes(b"image")

        self.assertEqual(text, receipt_text)
        self.assertEqual(calls, [6, 6, 6])

    def test_falls_back_to_all_layouts_when_fast_result_is_insufficient(self):
        receipt_text = "카페 파도\n합계 8,000원"
        calls: list[int] = []

        def run_tesseract(_image, *, language, psm):
            calls.append(psm)
            return receipt_text if psm == 11 else ""

        with (
            patch(
                "recommendation_api.receipt_ocr._load_image_variants",
                return_value=[("original", object()), ("contrast", object()), ("binary", object())],
            ),
            patch(
                "recommendation_api.receipt_ocr._run_tesseract",
                side_effect=run_tesseract,
            ),
        ):
            text = extract_receipt_text_from_image_bytes(b"image")

        self.assertEqual(text, receipt_text)
        self.assertEqual(len(calls), 12)


if __name__ == "__main__":
    unittest.main()
