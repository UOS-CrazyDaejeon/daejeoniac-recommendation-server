"""Image-only receipt OCR and structured field extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable

from pillow_heif import register_heif_opener


# Register HEIC/HEIF as Pillow image formats before Image.open() is called.
# Thumbnails are unnecessary for OCR and can add avoidable decoding work.
register_heif_opener(thumbnails=False)


class ReceiptDocumentError(ValueError):
    """Raised when the uploaded image cannot produce a usable receipt result."""


PSM_MODES = (6, 4, 11, 3)
# 일반적인 영수증은 기본 레이아웃(PSM 6)만으로 충분히 읽힌다. 나머지 모드는
# 결과가 불충분할 때만 실행해 저사양 배포 환경의 처리 시간을 줄인다.
FAST_PSM_MODES = (6,)
# 빠른 경로는 거의 모든 핵심 필드가 잡힌 경우에만 사용한다. 애매한 결과는
# 보조 OCR까지 실행해 정확도를 우선한다.
FAST_PATH_MIN_CONFIDENCE = 0.86
# 사진 전체를 먼저 지나치게 축소하면 멀리 찍힌 영수증 글자가 사라질 수 있다.
# 문서 보정 전에는 조금 더 큰 크기를 보존하고, 보정 후 OCR 크기로 제한한다.
MAX_SOURCE_LONG_EDGE = 4000
MAX_IMAGE_LONG_EDGE = 3000
RECEIPT_DETECTION_MAX_EDGE = 1600
# 멀리 촬영된 영수증도 후보로 잡되, 너무 작은 노이즈 윤곽은 제외한다.
RECEIPT_MIN_AREA_RATIO = 0.03
RECEIPT_MIN_EDGE = 96
RECEIPT_MIN_OCR_WIDTH = 1200

_AMOUNT_TOKEN_RE = re.compile(
    r"(?<!\d)[0-9Oo](?:[0-9Oo,\s]*[0-9Oo])?(?:\s*원)?"
)
_DATE_RE = re.compile(r"(?<!\d)(20\d{2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{1,2})")
_TIME_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[:시]\s*(\d{2})(?:\s*[:분]\s*(\d{2}))?"
)
_BUSINESS_NUMBER_RE = re.compile(r"(?<!\d)(\d{3})\D?(\d{2})\D?(\d{5})(?!\d)")
_APPROVAL_NUMBER_RE = re.compile(
    r"(?:영수증|승인번호|승인)[^\d]{0,12}"
    r"(\d{6,}(?:[-/]\d+)+|\d{8,})"
)
_STANDALONE_APPROVAL_NUMBER_RE = re.compile(r"(?<!\d)(20\d{6}-\d{2}-\d{4})(?!\d)")
_ROAD_ADDRESS_RE = re.compile(
    r"(?:특별시|광역시|서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주|시|도|군|구)"
    r".*?(?:대로|로|길)\s*\d+"
)
_ADDRESS_SIGNAL_RE = re.compile(r"주소|주\s*소|소재지")


@dataclass(frozen=True)
class _OcrAttempt:
    variant: str
    psm: int
    text: str
    score: int


def _resize_max_edge(image: Any, max_edge: int) -> Any:
    from PIL import Image

    if max(image.size) <= max_edge:
        return image
    ratio = max_edge / max(image.size)
    size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _order_receipt_corners(points: Any) -> Any:
    """Return four document corners in top-left, top-right, bottom-right, bottom-left order."""
    import numpy as np

    corners = np.asarray(points, dtype="float32").reshape(4, 2)
    sums = corners.sum(axis=1)
    differences = corners[:, 0] - corners[:, 1]
    return np.array(
        [
            corners[sums.argmin()],
            corners[differences.argmax()],
            corners[sums.argmax()],
            corners[differences.argmin()],
        ],
        dtype="float32",
    )


def _find_receipt_corners(image: Any) -> Any | None:
    """Detect the largest plausible four-corner document boundary in an RGB image."""
    import cv2
    import numpy as np

    source = np.asarray(image)
    height, width = source.shape[:2]
    scale = min(1.0, RECEIPT_DETECTION_MAX_EDGE / max(height, width))
    if scale < 1.0:
        detection = cv2.resize(
            source,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection = source

    gray = cv2.cvtColor(detection, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = detection.shape[0] * detection.shape[1] * RECEIPT_MIN_AREA_RATIO

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        corners = _order_receipt_corners(approximation.reshape(4, 2))
        if scale < 1.0:
            corners /= scale
        return corners
    return None


def _crop_and_rectify_receipt(image: Any) -> Any:
    """Crop and perspective-correct a receipt, falling back to the original image safely."""
    try:
        import cv2
        import numpy as np
        from PIL import Image

        corners = _find_receipt_corners(image)
        if corners is None:
            return image

        top_left, top_right, bottom_right, bottom_left = corners
        target_width = round(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
        target_height = round(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
        if min(target_width, target_height) < RECEIPT_MIN_EDGE:
            return image

        transform = cv2.getPerspectiveTransform(
            corners,
            np.array(
                [
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1],
                ],
                dtype="float32",
            ),
        )
        rectified = cv2.warpPerspective(
            np.asarray(image),
            transform,
            (target_width, target_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        receipt = Image.fromarray(rectified)
        if receipt.width < RECEIPT_MIN_OCR_WIDTH:
            ratio = RECEIPT_MIN_OCR_WIDTH / receipt.width
            receipt = receipt.resize(
                (RECEIPT_MIN_OCR_WIDTH, max(1, round(receipt.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        return receipt
    except Exception:
        # 문서 테두리 검출은 보조 전처리이므로, 실패해도 기존 OCR 경로를 유지한다.
        return image


def _load_image_variants(image_bytes: bytes) -> list[tuple[str, Any]]:
    """Load a camera image and create a small set of OCR-friendly variants."""
    if not image_bytes:
        raise ReceiptDocumentError("영수증 이미지가 비어 있습니다.")

    try:
        from PIL import Image, ImageEnhance, ImageOps

        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        image = _resize_max_edge(image, MAX_SOURCE_LONG_EDGE)
        image = _crop_and_rectify_receipt(image)
        image = _resize_max_edge(image, MAX_IMAGE_LONG_EDGE)

        gray = ImageOps.grayscale(image)
        contrast = ImageEnhance.Contrast(gray).enhance(1.8)
        try:
            import cv2
            import numpy as np

            adaptive = cv2.adaptiveThreshold(
                np.asarray(contrast),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9,
            )
            binary = Image.fromarray(adaptive)
        except Exception:
            binary = contrast.point(lambda pixel: 255 if pixel >= 170 else 0)
        return [
            ("original", image),
            ("contrast", contrast),
            ("binary", binary),
        ]
    except ReceiptDocumentError:
        raise
    except Exception as error:
        raise ReceiptDocumentError("영수증 이미지 형식을 읽지 못했습니다.") from error


def _run_tesseract(image: Any, *, language: str, psm: int) -> str:
    try:
        import pytesseract

        return pytesseract.image_to_string(
            image,
            lang=language,
            config=f"--psm {psm}",
        ) or ""
    except Exception as error:
        raise ReceiptDocumentError("Tesseract OCR 실행에 실패했습니다.") from error


def _ocr_quality_score(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return 0

    score = min(len(normalized), 1200)
    score += 250 if re.search(r"합계|총액|결제금액|받을금액", normalized) else 0
    score += 150 if re.search(r"\d{3}\D?\d{2}\D?\d{5}", normalized) else 0
    score += 100 if _DATE_RE.search(normalized) else 0
    score += 100 if re.search(r"공급가액|부가세|VAT", normalized, re.IGNORECASE) else 0
    return score


def _is_sufficient_fast_ocr_result(text: str) -> bool:
    """기본 OCR 결과가 구조화에 충분한 경우 느린 보조 시도를 생략한다."""
    if not text.strip():
        return False
    try:
        result = parse_receipt_text(text)
    except ReceiptDocumentError:
        return False
    return float(result["confidence"]) >= FAST_PATH_MIN_CONFIDENCE


def extract_receipt_text_from_image_bytes(
    image_bytes: bytes,
    language: str = "kor+eng",
) -> str:
    """OCR a receipt with a fast path and fallback layouts when needed."""
    attempts: list[_OcrAttempt] = []
    variants = _load_image_variants(image_bytes)

    for variant_name, image in variants:
        for psm in FAST_PSM_MODES:
            text = _run_tesseract(image, language=language, psm=psm)
            attempts.append(
                _OcrAttempt(
                    variant=variant_name,
                    psm=psm,
                    text=text,
                    score=_ocr_quality_score(text),
                )
            )

    best_fast_attempt = max(attempts, key=lambda attempt: attempt.score, default=None)
    if best_fast_attempt and _is_sufficient_fast_ocr_result(best_fast_attempt.text):
        return best_fast_attempt.text

    # 기본 결과가 불충분한 영수증만 기존의 모든 보조 레이아웃으로 재시도한다.
    for variant_name, image in variants:
        for psm in PSM_MODES:
            if psm in FAST_PSM_MODES:
                continue
            text = _run_tesseract(image, language=language, psm=psm)
            attempts.append(
                _OcrAttempt(
                    variant=variant_name,
                    psm=psm,
                    text=text,
                    score=_ocr_quality_score(text),
                )
            )

    best = max(attempts, key=lambda attempt: attempt.score, default=None)
    if best is None or not best.text.strip():
        raise ReceiptDocumentError("영수증에서 텍스트를 추출하지 못했습니다.")

    # A layout with the best overall score can still miss a small field such
    # as the approval number. Preserve high-signal lines found by other passes.
    best_lines = _clean_lines(best.text)
    high_signal_lines = []
    address_lines = []
    for attempt in attempts:
        attempt_lines = _clean_lines(attempt.text)
        for index, line in enumerate(attempt_lines):
            if (
                _APPROVAL_NUMBER_RE.search(line)
                or _STANDALONE_APPROVAL_NUMBER_RE.search(line)
            ) and line not in best_lines and line not in high_signal_lines:
                high_signal_lines.append(line)

            if _ADDRESS_SIGNAL_RE.search(line) or _ROAD_ADDRESS_RE.search(line):
                for offset, candidate in enumerate(attempt_lines[index:index + 3]):
                    if offset and (_is_receipt_section_boundary(candidate) or _DATE_RE.search(candidate)):
                        break
                    if candidate not in best_lines and candidate not in address_lines:
                        address_lines.append(candidate)

    if high_signal_lines or address_lines:
        return "\n".join([best.text.rstrip(), *high_signal_lines, *address_lines])
    return best.text


def _clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]


def _compact_label(text: str) -> str:
    """Remove OCR-inserted spaces from Korean labels only."""
    return re.sub(r"\s+", "", text).lower()


_ADDRESS_LABELS = ("도로명주소", "사업장주소", "가맹점주소", "주소", "소재지")
_RECEIPT_SECTION_LABELS = (
    "매장명",
    "상호명",
    "대표자",
    "tel",
    "매출일",
    "영수증",
    "상품명",
    "합계",
    "공급가액",
    "부가세",
    "받을금액",
    "받은금액",
    "신용카드",
    "체크카드",
    "승인",
    "단말기",
    "주소",
    "도로명주소",
    "사업장주소",
    "가맹점주소",
    "소재지",
)


def _value_after_label(line: str, labels: Iterable[str]) -> str | None:
    for label in sorted(labels, key=len, reverse=True):
        pattern = re.compile(r"\s*".join(re.escape(character) for character in label))
        match = pattern.search(line)
        if match is None:
            continue
        value = line[match.end():].lstrip(" ]:：-")
        return value.strip() or None
    return None


def _is_receipt_section_boundary(line: str) -> bool:
    compact = _compact_label(line).lstrip("[")
    return any(compact.startswith(label.lower()) for label in _RECEIPT_SECTION_LABELS)


def _normalize_address(parts: Iterable[str]) -> str | None:
    address = " ".join(part.strip(" []:：-") for part in parts if part.strip())
    address = re.sub(r"\s+", " ", address).strip()
    address = re.sub(r"^[소주]\s*\]\s*", "", address)
    address = re.sub(r"\(\s+", "(", address)
    address = re.sub(r"\s+\)", ")", address)
    return address or None


def _address_quality_score(address: str) -> int:
    score = min(len(address), 120)
    score += 40 if _ROAD_ADDRESS_RE.search(address) else 0
    score += 20 if re.search(r"\d", address) else 0
    score += 15 if re.search(r"\)|\d+호|\d+층", address) else 0
    score += 10 if re.search(r"서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충|전|경|제주", address) else 0
    score -= 20 * len(re.findall(r"[A-Za-z]{2,}", address))
    return score


def _extract_address(lines: list[str]) -> str | None:
    """Extract a labelled address and any wrapped continuation lines."""
    candidates: list[tuple[str, bool]] = []
    for index, line in enumerate(lines):
        first_line = _value_after_label(line, _ADDRESS_LABELS)
        if first_line is None:
            continue

        parts = [first_line]
        for continuation in lines[index + 1:index + 3]:
            if continuation.lstrip().startswith("[") or _is_receipt_section_boundary(continuation):
                break
            if _DATE_RE.search(continuation) or _BUSINESS_NUMBER_RE.search(continuation):
                break
            parts.append(continuation)
        address = _normalize_address(parts)
        if address:
            candidates.append((address, True))

    # Some POS templates omit the address label but still print a road address.
    for line in lines:
        if (
            _ROAD_ADDRESS_RE.search(line)
            and not _is_receipt_section_boundary(line)
        ):
            address = _normalize_address([line])
            if address:
                candidates.append((address, False))

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: _address_quality_score(candidate[0]) + (100 if candidate[1] else 0),
    )[0]


def _parse_amount(raw: str) -> int | None:
    digits = re.sub(r"[^0-9Oo]", "", raw).replace("O", "0").replace("o", "0")
    if not digits:
        return None
    return int(digits)


def _amounts_on_line(line: str) -> list[int]:
    values: list[int] = []
    for token in _AMOUNT_TOKEN_RE.findall(line):
        value = _parse_amount(token)
        if value is not None:
            values.append(value)
    return values


def _comma_amounts(lines: Iterable[str]) -> list[int]:
    """Collect receipt-like amounts when OCR destroys the nearby label."""
    values: list[int] = []
    for line in lines:
        if _DATE_RE.search(line) or _BUSINESS_NUMBER_RE.search(line):
            continue
        for token in _AMOUNT_TOKEN_RE.findall(line):
            if "," not in token:
                continue
            value = _parse_amount(token)
            if value is not None and 0 < value <= 10_000_000:
                values.append(value)
    return values


def _infer_supply_and_vat(
    candidates: list[int],
    total: int,
    supply: int | None,
    vat: int | None,
) -> tuple[int | None, int | None]:
    """Recover tax fields when OCR leaves only the numeric lines."""
    if supply is not None and vat is not None:
        return supply, vat

    unique_candidates = sorted(set(candidates), reverse=True)
    for first in unique_candidates:
        for second in unique_candidates:
            if first <= second or first + second != total:
                continue
            return supply or first, vat or second
    return supply, vat


def _labeled_amount(
    lines: Iterable[str],
    labels: tuple[str, ...],
    excluded_labels: tuple[str, ...] = (),
) -> int | None:
    for line in lines:
        compact = _compact_label(line)
        if not any(label.lower() in compact for label in labels):
            continue
        if any(label.lower() in compact for label in excluded_labels):
            continue
        values = _amounts_on_line(line)
        if values:
            return values[-1]
    return None


def _extract_merchant(lines: list[str]) -> str | None:
    def normalize(candidate: str) -> str | None:
        value = _BUSINESS_NUMBER_RE.sub("", candidate)
        value = value.split("/", 1)[0].strip(" /[]:：-")
        value = re.sub(r"\s+", " ", value)
        if len(value) < 2 or not re.search(r"[가-힣A-Za-z]", value):
            return None
        return value

    labels = ("상호명", "가맹점명", "상호", "매장명", "업체명")
    for line in lines:
        for label in labels:
            if label in line:
                value = normalize(line.split(label, 1)[1].lstrip(" ]:：-"))
                if value:
                    return value

    # POS 영수증은 보통 "상호 / 사업자번호 / 대표자" 형식의 첫 줄을 사용한다.
    # 하단 카드 승인 정보보다 이 줄을 우선해야 할부 개월 같은 값을 상호로 오인하지 않는다.
    for line in lines:
        business_match = _BUSINESS_NUMBER_RE.search(line)
        if business_match is None:
            continue
        value = normalize(line[:business_match.start()])
        if value and not any(token in value for token in ("사업자", "등록번호", "대표자")):
            return value

    ignored = (
        "사업자",
        "합계",
        "총액",
        "공급가액",
        "부가세",
        "결제",
        "신용카드",
        "체크카드",
        "할부",
        "개월",
        "일시불",
        "카드번호",
        "승인번호",
        "승인일시",
        "가맹점번호",
        "판매금액",
        "영수증",
        "상품명",
        "단가",
        "수량",
    )
    for line in lines:
        if (
            len(line) <= 60
            and not any(token in line for token in ignored)
            and not _DATE_RE.search(line)
            and not _amounts_on_line(line)
        ):
            value = normalize(line)
            if value:
                return value
    return None


def _extract_items(lines: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    number = r"[0-9Oo](?:[0-9Oo,\s]*[0-9Oo])?"
    four_column_pattern = re.compile(
        rf"^(?P<name>.+?)\s+(?P<unit_price>{number})\s+"
        rf"(?P<quantity>[0-9Oo]+)\s+(?P<amount>{number})\s*원?$"
    )
    pattern = re.compile(
        r"^(?P<name>.+?)\s+(?P<quantity>[0-9Oo]+)\s+(?P<amount>[0-9Oo][0-9Oo,\s]*)\s*원?$"
    )
    excluded = ("공급가액", "부가세", "합계", "총액", "결제", "사업자")
    for line in lines:
        if any(label in line for label in excluded) or _DATE_RE.search(line):
            continue

        four_column_match = four_column_pattern.match(line)
        if four_column_match:
            unit_price = _parse_amount(four_column_match.group("unit_price"))
            quantity = _parse_amount(four_column_match.group("quantity"))
            amount = _parse_amount(four_column_match.group("amount"))
            name = four_column_match.group("name").strip(" :-")
            if unit_price is not None and quantity is not None and amount is not None and name:
                items.append({
                    "name": name,
                    "quantity": quantity,
                    "unitPrice": unit_price,
                    "amount": amount,
                })
                continue

        match = pattern.match(line)
        if not match:
            continue
        amount = _parse_amount(match.group("amount"))
        quantity = _parse_amount(match.group("quantity"))
        name = match.group("name").strip(" :-")
        if amount is None or quantity is None or not name:
            continue
        items.append({
            "name": name,
            "quantity": quantity,
            "unitPrice": round(amount / quantity) if quantity else amount,
            "amount": amount,
        })
    return items


def _extract_items_from_word_rows(
    rows: list[tuple[int, list[tuple[int, str]]]],
    total: int,
) -> list[dict[str, Any]]:
    """Rebuild table rows from Tesseract words grouped by their y coordinate."""
    if not rows:
        return []

    row_texts = [
        " ".join(
            text for _, text in sorted(words, key=lambda item: item[0])
            if text.strip() not in {"|", "[", "]"}
        )
        for _, words in rows
    ]
    header_index = next(
        (index for index, text in enumerate(row_texts) if "상품명" in _compact_label(text)),
        None,
    )
    if header_index is None:
        return []

    end_index = len(row_texts)
    for index in range(header_index + 1, len(row_texts)):
        compact = _compact_label(row_texts[index])
        if total in _amounts_on_line(row_texts[index]) or any(
            label in compact for label in ("합계", "받을금액", "받은금액")
        ):
            end_index = index
            break

    items: list[dict[str, Any]] = []
    for line in row_texts[header_index + 1:end_index]:
        parsed = _extract_items([line])
        for item in parsed:
            item["name"] = re.sub(r"\s+", "", item["name"])
            item["name"] = re.sub(r"^[0-9Oo]{3,}(?=[가-힣])", "", item["name"])
        parsed = [
            item for item in parsed
            if item["unitPrice"] * item["quantity"] == item["amount"]
        ]
        items.extend(parsed)
    return items


def _group_tesseract_words(data: dict[str, list[Any]]) -> list[tuple[int, list[tuple[int, str]]]]:
    groups: list[dict[str, Any]] = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            top = int(data["top"][index])
            left = int(data["left"][index])
        except (KeyError, TypeError, ValueError):
            continue

        group = min(groups, key=lambda candidate: abs(candidate["top"] - top), default=None)
        if group is None or abs(group["top"] - top) > 50:
            group = {"top": top, "words": []}
            groups.append(group)
        group["words"].append((left, text))

    return [
        (group["top"], sorted(group["words"], key=lambda item: item[0]))
        for group in sorted(groups, key=lambda group: group["top"])
    ]


def _extract_items_from_image(
    image_bytes: bytes,
    language: str,
    total: int,
) -> list[dict[str, Any]]:
    """Use OCR coordinates as a fallback when string OCR loses table columns."""
    try:
        import pytesseract
        from pytesseract import Output

        candidates: list[list[dict[str, Any]]] = []
        for _, image in _load_image_variants(image_bytes)[:2]:
            data = pytesseract.image_to_data(
                image,
                lang=language,
                config="--psm 6",
                output_type=Output.DICT,
            )
            rows = _group_tesseract_words(data)
            items = _extract_items_from_word_rows(rows, total)
            if items:
                candidates.append(items)
        return max(candidates, key=len, default=[])
    except Exception:
        # Missing coordinate OCR support must not make an otherwise valid receipt fail.
        return []


def parse_receipt_text(text: str) -> dict[str, Any]:
    """Convert OCR text into receipt fields and validation warnings."""
    lines = _clean_lines(text)
    if not lines:
        raise ReceiptDocumentError("영수증 OCR 결과가 비어 있습니다.")

    joined = "\n".join(lines)
    date_match = _DATE_RE.search(joined)
    time_match = _TIME_RE.search(joined)
    business_match = _BUSINESS_NUMBER_RE.search(joined)
    approval_match = _APPROVAL_NUMBER_RE.search(joined)
    if approval_match is None:
        approval_match = _STANDALONE_APPROVAL_NUMBER_RE.search(joined)

    comma_amounts = _comma_amounts(lines)
    total = _labeled_amount(
        lines,
        ("총결제금액", "총액", "합계금액", "합계", "결제금액", "받을금액", "승인금액"),
    )
    if total is None:
        total = _labeled_amount(lines, ("금액",))
    if total is None and comma_amounts:
        total = max(comma_amounts)
    if total is None or total <= 0:
        raise ReceiptDocumentError("영수증 총액을 확인하지 못했습니다.")

    supply = _labeled_amount(lines, ("과세물품가액", "공급가액", "공급가"))
    vat = _labeled_amount(
        lines,
        ("부가세", "VAT"),
        excluded_labels=("과세물품가액",),
    )
    supply, vat = _infer_supply_and_vat(comma_amounts, total, supply, vat)
    address = _extract_address(lines)
    warnings: list[str] = []

    if supply is not None and vat is not None and supply + vat != total:
        warnings.append("공급가액과 부가세 합계가 총액과 일치하지 않습니다.")

    payment_method = None
    compact_joined = _compact_label(joined)
    for method in ("신용카드", "체크카드", "현금", "간편결제", "카카오페이", "네이버페이"):
        if _compact_label(method) in compact_joined:
            payment_method = method
            break

    merchant = _extract_merchant(lines)
    if merchant is None:
        warnings.append("상호명을 확인하지 못했습니다.")
    if address is None:
        warnings.append("주소를 확인하지 못했습니다.")
    if date_match is None:
        warnings.append("거래일자를 확인하지 못했습니다.")
    if business_match is None:
        warnings.append("사업자등록번호를 확인하지 못했습니다.")

    confidence_fields = [merchant, address, date_match, business_match, supply, vat, total]
    confidence = round(sum(value is not None for value in confidence_fields) / len(confidence_fields), 2)

    return {
        "merchantName": merchant,
        "businessNumber": (
            f"{business_match.group(1)}-{business_match.group(2)}-{business_match.group(3)}"
            if business_match else None
        ),
        "transactionDate": (
            f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
            if date_match else None
        ),
        "transactionTime": (
            (
                f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
                + (f":{int(time_match.group(3)):02d}" if time_match.group(3) else "")
            )
            if time_match else None
        ),
        "address": address,
        "approvalNumber": approval_match.group(1) if approval_match else None,
        "supplyAmount": supply,
        "vat": vat,
        "totalAmount": total,
        "paymentMethod": payment_method,
        "items": [],
        "confidence": confidence,
        "warnings": warnings,
    }


def analyze_receipt_image_bytes(
    image_bytes: bytes,
    language: str = "kor+eng",
) -> dict[str, Any]:
    text = extract_receipt_text_from_image_bytes(image_bytes, language=language)
    result = parse_receipt_text(text)
    return {"result": result, "rawOcrText": text}
