from io import BytesIO
import unittest

from PIL import Image

from recommendation_api.receipt_ocr import (
    ReceiptDocumentError,
    _load_image_variants,
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

    def test_requires_total_amount(self):
        with self.assertRaises(ReceiptDocumentError):
            parse_receipt_text("카페 파도\n2026-08-01\n아메리카노")


if __name__ == "__main__":
    unittest.main()
