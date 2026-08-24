# Daejeon Internal API

Spring이 MySQL에서 조회한 장소 후보와 최근 선택 이력을 받아 기존
`recommend_llm.py` 추천 파이프라인을 실행하는 내부 FastAPI 서비스다.
영수증 이미지 업로드를 받아 Tesseract OCR 결과도 반환한다. Python
서비스는 MySQL에 직접 연결하지 않는다.

## API

### 상태 확인

```http
GET /health
```

```json
{"status": "ok"}
```

### 추천 API 구분

| 목적 | 엔드포인트 | 실제 계산 결과 |
| --- | --- | --- |
| 선택 장소와 비슷한 장소 | `POST /api/v1/recommendations/similar-places` | `similar_places` 5개 |
| 최근 이동 흐름에 맞는 다음 장소 | `POST /api/v1/recommendations/next-places` | `next_places` 5개와 추천 로그 |
| 기존 통합 API | `POST /api/v1/recommendations` | 위 두 결과를 모두 계산하며 deprecated |

분리 API를 사용하면 필요하지 않은 추천 모델을 함께 실행하지 않는다.

### 선택한 장소와 비슷한 장소 추천

```http
POST /api/v1/recommendations/similar-places
Content-Type: application/json
```

`selected_place`가 유사도 비교 기준이다. 장소의 카테고리, 태그, 설명과
후보별 거리를 사용하며, OpenAI가 활성화되어 있으면 문맥 유사도도 함께
평가한다. 이 요청에는 최근 이동 이력이나 현재 시각이 필요 없다. 아래
`candidates` 값은 문서 가독성을 위한 축약 표기이며 실제 요청에는 장소 객체
10개를 넣어야 한다.

```json
{
  "request_id": "similar-001",
  "session_id": "session-001",
  "selected_place": {
    "id": "place-100",
    "name": "선택한 카페",
    "latitude": 36.35,
    "longitude": 127.38,
    "category": "cafe",
    "description": "조용한 로컬 카페",
    "tags": ["조용한", "로컬", "커피"]
  },
  "visited_place_ids": [],
  "candidates": ["Spring에서 조회한 장소 객체 정확히 10개"],
  "context": {
    "radius_m": 1000,
    "top_k": 5
  }
}
```

응답에는 `selected_place_id`와 `similar_places`만 포함되고 `next_places`는
계산하거나 반환하지 않는다.

### 다음 장소 추천

```http
POST /api/v1/recommendations/next-places
Content-Type: application/json
```

`current_place`, 최근 선택 장소인 `recent_places`, 시간·날씨 문맥을 사용해
다음 이동 장소를 계산한다. 응답에는 `next_places`와 `recommendation_log`만
포함되고 `similar_places`는 계산하지 않는다. 아래 `candidates`도 실제
요청에서는 장소 객체 10개로 바꿔야 한다.

```json
{
  "request_id": "next-001",
  "session_id": "session-001",
  "current_place": {
    "id": "place-100",
    "name": "현재 장소",
    "latitude": 36.35,
    "longitude": 127.38
  },
  "recent_places": [],
  "visited_place_ids": [],
  "candidates": ["Spring에서 조회한 장소 객체 정확히 10개"],
  "context": {
    "current_time": "2026-08-15T14:00:00+09:00",
    "weather": "맑음",
    "user_preferences": "조용한 장소",
    "radius_m": 1000,
    "top_k": 5
  }
}
```

### 기존 통합 추천 요청(deprecated)

```http
POST /api/v1/recommendations
Content-Type: application/json
```

요청 본문은 다음 필드를 사용한다.

| 필드 | 조건 |
| --- | --- |
| `request_id` | Spring에서 생성한 요청 추적 ID |
| `session_id` | 사용자 이동 세션 ID |
| `current_place` | 현재 장소와 위도·경도 |
| `recent_places` | 최근 선택 장소 0~4개 |
| `visited_place_ids` | 이미 방문한 장소 ID 목록 |
| `candidates` | MySQL에서 조회한 후보 정확히 10개 |
| `context.radius_m` | 0보다 크고 1000 이하 |
| `context.similar_top_k` | 5 |
| `context.next_top_k` | 5 |

장소 객체는 `id`, `name`, `latitude`, `longitude`를 필수로 사용한다.
`congestion`, `monthly_visitors`, `selected_count`, `category`, `description`,
`tags`를 함께 보내면 추천 점수에 반영된다.

성공 응답에는 `similar_places` 5개, `next_places` 5개와 추천 노출 로그가
포함된다. 기존 Spring 클라이언트의 호환성을 위해 남겨 두었으며, 신규
호출부는 목적에 맞는 분리 API를 사용한다. 오류 응답은 다음 형식으로
통일된다.

```json
{
  "error": {
    "code": "INVALID_RECOMMENDATION_REQUEST",
    "message": "1km 이내의 미방문 후보가 최소 5개 필요합니다",
    "request_id": "rec-20260724-001"
  }
}
```

### 영수증 이미지 분석

```http
POST /api/v1/receipts/analyze
Content-Type: multipart/form-data
```

| multipart 필드 | 조건 |
| --- | --- |
| `image` | 필수 영수증 이미지(JPEG, PNG, WebP, HEIC, HEIF), 기본 최대 10MB |
| `requestId` | 선택 요청 추적 ID |
| `documentId` | 선택 영수증 문서 ID |
| `userId` | 선택 사용자 ID(정수) |

로컬 이미지로 요청하는 예시는 다음과 같다.

```bash
curl --fail-with-body \
  -F 'image=@/absolute/path/to/receipt.jpg;type=image/jpeg' \
  -F 'requestId=req_001' \
  -F 'documentId=doc_1001' \
  -F 'userId=4' \
  http://127.0.0.1:8000/api/v1/receipts/analyze
```

성공 응답은 Lambda 결과와 호환되는 camelCase 필드를 사용한다. OCR 원문은
영수증 개인정보 보호를 위해 응답하지 않고 글자 수만 반환한다.

```json
{
  "requestId": "req_001",
  "documentId": "doc_1001",
  "userId": 4,
  "documentType": "RECEIPT",
  "status": "COMPLETED",
  "result": {
    "merchantName": "카페 파도",
    "businessNumber": "123-45-67890",
    "transactionDate": "2026-08-01",
    "transactionTime": "14:32",
    "address": "대전광역시 유성구 대학로 291",
    "approvalNumber": null,
    "supplyAmount": 7273,
    "vat": 727,
    "totalAmount": 8000,
    "paymentMethod": "신용카드",
    "items": [],
    "confidence": 1.0,
    "warnings": []
  },
  "warnings": [],
  "processedAt": "2026-08-15T12:00:00Z",
  "rawOcrCharCount": 120
}
```

이미지를 읽을 수 없거나 총액을 추출하지 못하면 `422`, 10MB를 초과하면
`413`, 이미지가 아닌 Content-Type이면 `415`를 반환한다.

### GPT-5 mini 영수증 분석

기존 Tesseract 결과가 불명확할 때 비교하거나 fallback으로 호출할 수 있는
별도 Vision 엔드포인트다.

```http
POST /api/v1/receipts/analyze-gpt-mini
Content-Type: multipart/form-data
```

multipart 필드는 기존 `/api/v1/receipts/analyze`와 같다. HEIC/HEIF를 포함한
휴대폰 이미지를 서버에서 방향 보정한 JPEG로 바꾼 다음 `gpt-5-mini`에
전송한다. 호출 전 `recommendation_api/.env`에 `OPENAI_API_KEY`를 설정해야
하며, 설정되지 않으면 `503 RECEIPT_VISION_NOT_CONFIGURED`를 반환한다.

```bash
curl --fail-with-body \
  -F 'image=@/absolute/path/to/receipt.heic;type=image/heic' \
  -F 'requestId=req_gpt_001' \
  -F 'documentId=doc_1001' \
  -F 'userId=4' \
  http://127.0.0.1:8000/api/v1/receipts/analyze-gpt-mini
```

응답의 `processingTimeMs`와 `usage`로 지연 시간과 토큰 사용량을 비교할 수
있다. `confidence`는 모델이 스스로 보고한 판독 확신도이므로 실제 정확도
평가값으로 간주하지 말고, 정답을 기록한 여러 영수증으로 필드별 정확도를
별도로 측정해야 한다.

```json
{
  "requestId": "req_gpt_001",
  "documentId": "doc_1001",
  "userId": 4,
  "documentType": "RECEIPT",
  "status": "COMPLETED",
  "model": "gpt-5-mini-2025-08-07",
  "result": {
    "merchantName": "카페 파도",
    "businessNumber": "123-45-67890",
    "transactionDate": "2026-08-01",
    "transactionTime": "14:32",
    "address": "대전광역시 유성구 대학로 291",
    "approvalNumber": null,
    "supplyAmount": 7273,
    "vat": 727,
    "totalAmount": 8000,
    "paymentMethod": "신용카드",
    "items": [],
    "confidence": 0.93,
    "warnings": []
  },
  "warnings": [],
  "processedAt": "2026-08-15T12:00:00Z",
  "processingTimeMs": 1260,
  "usage": {
    "inputTokens": 1100,
    "outputTokens": 220,
    "totalTokens": 1320
  }
}
```

GPT 호출 실패는 `502 RECEIPT_VISION_UPSTREAM_ERROR`, 이미지나 총액을
판독하지 못한 경우는 `422 RECEIPT_ANALYSIS_FAILED`로 구분한다. Spring은
먼저 기존 OCR API를 호출한 뒤 `422`이거나 응답의 `warnings`가 존재하는
경우에만 이 엔드포인트를 호출하면 불필요한 외부 API 비용을 줄일 수 있다.

### S3 영수증 이미지 분석

Spring은 이미지 파일이나 S3 URL 전체를 보내지 않고 버킷 내부 객체 키인
`s3Key`만 보낸다. Python 서버는 `.env`에 고정된 버킷에서만 이미지를 읽는다.

| 분석 방식 | 엔드포인트 |
| --- | --- |
| Spring OCR 연동 | `POST /ocr` |
| Tesseract OCR | `POST /api/v1/receipts/analyze-from-s3` |
| GPT-5 Mini | `POST /api/v1/receipts/analyze-gpt-mini-from-s3` |

두 API의 JSON 요청 형식은 같다.

Spring이 사용하는 기본 OCR 연동은 `POST /ocr`이다. 이미지 파일이나 S3 URL이
아닌 영수증 UUID와 버킷 내부 객체 키만 전달한다.

```json
{
  "receiptUuid": "2d6ae292-3e3b-4c95-a102-779562ee12bc",
  "objectKey": "receipts/2026/08/receipt-001.heic"
}
```

OCR이 끝나면 Python 서버는 `SPRING_OCR_CALLBACK_URL`에 설정한 Spring의
`POST /api/v1/receipts/ocr-result`로 아래 결과를 먼저 전송한다. 이 콜백이
2xx가 아닌 경우 `/ocr`도 `502 OCR_RESULT_CALLBACK_FAILED`로 실패 처리한다.

```json
{
  "receiptUuid": "2d6ae292-3e3b-4c95-a102-779562ee12bc",
  "ocrStatus": "COMPLETED",
  "ocrPlaceName": "카페 파도",
  "ocrPlaceAddress": "대전광역시 유성구 대학로 291",
  "ocrPaidAt": "2026-08-23T14:32:00"
}
```

`ocrPaidAt`은 `LocalDateTime` 형식이며, 결제 날짜 또는 시간을 OCR로 판독하지
못하면 `null`이다. `ocrStatus`는 기본값이 `COMPLETED`이며 Spring의 상태값에
맞춰 환경변수로 변경할 수 있다.

`/ocr` 호출 응답에도 전체 OCR 결과를 반환하지만, Spring DB 반영 기준은 위
콜백이다.

배포 서버가 `http://3.39.230.42`라면 Spring의 호출 주소는
`http://3.39.230.42/ocr`이다. 운영 환경에서는 이 서버를 Spring 서버만 접근할
수 있는 보안 그룹 또는 내부 네트워크로 제한한다.

```json
{
  "requestId": "req-s3-001",
  "documentId": "doc-1001",
  "userId": 4,
  "s3Key": "receipts/2026/08/receipt-001.heic"
}
```

S3 설정은 `recommendation_api/.env`에 넣는다. 아래 점(`.`) 형식과 AWS 표준
환경 변수(`AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_S3_BUCKET`)를 모두 지원한다.

```dotenv
SPRING_OCR_CALLBACK_URL=http://spring-server:8080/api/v1/receipts/ocr-result
SPRING_OCR_CALLBACK_TIMEOUT_SECONDS=5
SPRING_OCR_SUCCESS_STATUS=COMPLETED
# Spring 내부 인증이 필요한 경우에만 설정한다.
SPRING_OCR_CALLBACK_AUTHORIZATION=

aws.region=ap-northeast-2
aws.access-key=
aws.secret-key=
aws.s3.bucket=uos-crazy-daejeon-images-521701612202-ap-northeast-2-an
aws.s3.receipt-prefix=receipts/
aws.s3.expected-bucket-owner=
```

`aws.s3.receipt-prefix`를 설정하면 다른 경로의 객체 키는 `400`으로 거절한다.
`aws.s3.expected-bucket-owner`는 실제 버킷 소유 AWS 계정 ID를 확인한 경우에만
설정한다.
객체가 없으면 `404`, 10MB 초과는 `413`, 이미지가 아닌 객체는 `415`, S3
호출 장애나 권한 오류는 `502`, 설정이나 자격 증명 누락은 `503`을 반환한다.

EC2 운영 환경에서는 장기 액세스 키를 `.env`에 저장하지 않고 인스턴스 IAM
Role에 다음과 같이 읽기 권한만 부여하는 방식을 권장한다. 이 경우
`aws.access-key`, `aws.secret-key`는 비워 두면 Boto3가 Role의 임시 자격
증명을 자동으로 사용한다.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::uos-crazy-daejeon-images-521701612202-ap-northeast-2-an/receipts/*"
    }
  ]
}
```

- Boto3 자격 증명: https://docs.aws.amazon.com/boto3/latest/guide/credentials.html
- S3 `GetObject`: https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/get_object.html

### 얼굴 모자이크

신원을 식별하는 얼굴 인식은 사용하지 않고, OpenCV YuNet으로 얼굴의
위치만 검출해 픽셀 모자이크를 적용한다. YuNet ONNX 모델은 약 227KB이고
CPU로 실행되므로 2 vCPU, 2GiB인 EC2 `t3.small`에서도 사용할 수 있다.

운영 API는 S3에서 원본을 읽고 결과를 새 S3 객체로 저장한다.

```http
POST /api/v1/images/face-mosaic
Content-Type: application/json
```

```json
{
  "requestId": "req-face-001",
  "userId": 4,
  "s3Key": "uploads/2026/08/photo-001.heic"
}
```

응답의 `outputS3Key`를 Spring DB에 저장하면 된다.

```json
{
  "requestId": "req-face-001",
  "userId": 4,
  "status": "COMPLETED",
  "sourceS3Key": "uploads/2026/08/photo-001.heic",
  "outputS3Key": "mosaics/8c3f...-mosaic.jpg",
  "contentType": "image/jpeg",
  "faceCount": 2,
  "width": 1920,
  "height": 1080,
  "processedAt": "2026-08-16T12:00:00Z"
}
```

`.env`의 추가 설정은 다음과 같다.

```dotenv
# 비우면 버킷의 모든 이미지 키를 입력으로 허용한다.
aws.s3.image-prefix=uploads/
aws.s3.mosaic-prefix=mosaics/

# t3.small 권장값
MAX_FACE_IMAGE_BYTES=10485760
FACE_MAX_IMAGE_PIXELS=24000000
FACE_OUTPUT_MAX_EDGE=2560
FACE_DETECTION_MAX_EDGE=1280
FACE_DETECTION_SCORE_THRESHOLD=0.75
FACE_BOX_PADDING=0.18
FACE_MOSAIC_BLOCK_SIZE=14
FACE_OPENCV_THREADS=1
FACE_MAX_CONCURRENT_JOBS=1
```

얼굴을 하나도 검출하지 못하면 비식별화되지 않은 원본을 공개하지
않도록 `422 FACE_NOT_DETECTED`를 반환하고 S3에 결과를 저장하지 않는다.
얼굴 검출은 100% 보장되지 않으므로 공개 전 검수 또는 사용자 확인 절차를
추가하는 것이 좋다.

EC2 IAM Role은 원본 경로의 `s3:GetObject`와 결과 경로의
`s3:PutObject` 권한이 필요하다.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::uos-crazy-daejeon-images-521701612202-ap-northeast-2-an/uploads/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::uos-crazy-daejeon-images-521701612202-ap-northeast-2-an/mosaics/*"
    }
  ]
}
```

로컬 확인용 API는 요청 경로 끝에 `-local`을 붙였다. S3 설정 없이
파일을 바로 올리고 결과 JPEG를 저장할 수 있다.

```bash
curl --fail-with-body \
  -X POST \
  -F "image=@/absolute/path/photo.heic" \
  http://127.0.0.1:8000/api/v1/images/face-mosaic-local \
  --output face-mosaic.jpg
```

Spring `WebClient` 요청 예시는 다음과 같다.

```java
public record FaceMosaicRequest(String requestId, Long userId, String s3Key) {}

public record FaceMosaicResponse(
        String requestId,
        Long userId,
        String status,
        String sourceS3Key,
        String outputS3Key,
        String contentType,
        int faceCount,
        int width,
        int height,
        String processedAt) {}

public Mono<FaceMosaicResponse> mosaicFaces(String s3Key, Long userId) {
    var request = new FaceMosaicRequest(
            UUID.randomUUID().toString(), userId, s3Key);

    return recommendationWebClient.post()
            .uri("/api/v1/images/face-mosaic")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(request)
            .retrieve()
            .bodyToMono(FaceMosaicResponse.class);
}
```

## 로컬 실행

### Python으로 실행

```bash
.venv/bin/python -m pip install -r recommendation_api/requirements.txt
.venv/bin/uvicorn recommendation_api.main:app --host 127.0.0.1 --port 8000
```

다른 터미널에서 확인한다.

```bash
curl --fail http://127.0.0.1:8000/health
```

OpenAI를 사용할 때만 `OPENAI_API_KEY`를 설정한다. 키가 없으면 기존
추천 코드의 fallback scorer가 사용되지만 GPT 영수증 엔드포인트는 `503`을
반환한다. `RECEIPT_VISION_MODEL`의 기본값은 `gpt-5-mini`, 이미지 세부 수준인
`RECEIPT_VISION_DETAIL`의 기본값은 OCR에 적합한 `high`다.

영수증 OCR은 Tesseract 한국어·영어 데이터가 필요하므로 Docker 실행을
권장한다. Docker 이미지에는 필요한 Tesseract 패키지와 HEIC/HEIF 디코더가
포함된다. 휴대폰 사진은 EXIF 방향을 보정하고 RGB로 변환한 뒤 OCR한다.

### Docker Compose로 실행

```bash
cp recommendation_api/.env.example recommendation_api/.env
chmod 600 recommendation_api/.env
docker compose -f recommendation_api/deploy/docker-compose.yml up -d --build
curl --fail http://127.0.0.1:8000/health
```

`.env`의 `OPENAI_API_KEY` 값은 이미지에 포함되지 않는다. API 문서는
로컬 실행 중 `http://127.0.0.1:8000/docs`에서 확인할 수 있다.

기존 첫 번째 Spring 시나리오를 서버에 보내는 로컬 검증 명령은 다음과
같다.

```bash
.venv/bin/python -c "import httpx, recommend_llm; request=recommend_llm.load_scenario_requests()[0]; response=httpx.post('http://127.0.0.1:8000/api/v1/recommendations', json=request, timeout=90); print(response.status_code); print(response.json())"
```

실제 OpenAI 키가 설정되어 있으면 GPT 점수와 추천 이유가 생성되고 API
사용량이 발생한다.

## Spring 연결

요청 흐름은 다음과 같다.

```text
웹사이트 -> Spring -> MySQL 조회 -> FastAPI -> recommend_llm.py
                              <- 추천 결과 <-
```

Spring 설정은 Python EC2의 사설 IP를 사용한다.

```yaml
recommendation:
  base-url: ${RECOMMENDATION_API_BASE_URL:http://127.0.0.1:8000}
```

운영 환경 변수 예시는 다음과 같다.

```dotenv
RECOMMENDATION_API_BASE_URL=http://10.0.2.15
```

`WebClient`는 GPT 응답 시간을 고려해 연결 제한 3초, 전체 응답 제한
90초를 사용한다.

```java
@Bean
WebClient recommendationWebClient(
        WebClient.Builder builder,
        @Value("${recommendation.base-url}") String baseUrl) {
    HttpClient httpClient = HttpClient.create()
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 3000)
            .responseTimeout(Duration.ofSeconds(90));

    return builder
            .baseUrl(baseUrl)
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            .build();
}
```

```java
public Mono<SimilarPlacesResponse> recommendSimilarPlaces(
        SimilarPlacesRequest request) {
    return recommendationWebClient.post()
            .uri("/api/v1/recommendations/similar-places")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(SimilarPlacesResponse.class);
}

public Mono<NextPlacesResponse> recommendNextPlaces(NextPlacesRequest request) {
    return recommendationWebClient.post()
            .uri("/api/v1/recommendations/next-places")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(NextPlacesResponse.class);
}
```

Spring이 S3 업로드를 마친 뒤 Python에 객체 키를 전달하는 코드는 다음처럼
구성할 수 있다.

```java
public record S3ReceiptAnalysisRequest(
        String requestId,
        String documentId,
        Long userId,
        String s3Key) {
}

public Mono<ReceiptAnalysisResponse> analyzeReceiptFromS3(
        S3ReceiptAnalysisRequest request) {
    return recommendationWebClient.post()
            .uri("/api/v1/receipts/analyze-from-s3")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(request)
            .retrieve()
            .bodyToMono(ReceiptAnalysisResponse.class);
}
```

GPT-5 Mini로 분석하려면 요청 본문은 그대로 두고 URI만
`/api/v1/receipts/analyze-gpt-mini-from-s3`로 바꾼다.

Spring에서 받은 `MultipartFile`을 그대로 전달할 때는 다음 형태로 호출할
수 있다.

```java
public Mono<ReceiptAnalysisResponse> analyzeReceipt(
        MultipartFile image,
        String requestId,
        String documentId,
        Long userId) {
    MultipartBodyBuilder body = new MultipartBodyBuilder();
    body.part("image", image.getResource())
            .filename(image.getOriginalFilename())
            .contentType(MediaType.parseMediaType(image.getContentType()));
    body.part("requestId", requestId);
    body.part("documentId", documentId);
    body.part("userId", userId.toString());

    return recommendationWebClient.post()
            .uri("/api/v1/receipts/analyze")
            .contentType(MediaType.MULTIPART_FORM_DATA)
            .body(BodyInserters.fromMultipartData(body.build()))
            .retrieve()
            .bodyToMono(ReceiptAnalysisResponse.class);
}
```

OCR는 CPU 작업이므로 동시 요청이 많아지면 API 컨테이너 복제 또는 별도
OCR worker 분리를 검토한다. 현재 엔드포인트는 요청이 끝날 때까지 기다리는
동기 방식이다.

Spring은 호출마다 `request_id`를 만들고 양쪽 서버 로그에 남겨야 한다.
FastAPI의 `400`, `422`, `500` 응답 본문도 Spring에서 그대로 기록하면
장애 원인을 요청 단위로 찾을 수 있다.

## EC2 배포

### 1. 네트워크

Python EC2는 Spring EC2와 같은 VPC에 두고 사설 IPv4로 호출한다. Python
EC2 보안 그룹의 인바운드는 다음처럼 제한한다.

| 포트 | 소스 | 용도 |
| --- | --- | --- |
| TCP 80 | Spring EC2 보안 그룹 ID | Nginx Python API (`/api/*`, `/ocr`) |
| TCP 22 | 관리자 고정 IP | 초기 SSH 작업, SSM 사용 시 제거 가능 |

TCP 8000은 외부에 열지 않는다. Compose가 `127.0.0.1:8000`에만
바인딩하므로 Nginx만 컨테이너에 접근한다. 아웃바운드 TCP 443은 OpenAI,
패키지 저장소, 향후 ECR 접근을 위해 필요하다.

AWS는 같은 VPC의 인바운드 규칙에서 다른 보안 그룹을 소스로 참조할 수
있고, 같은 VPC 인스턴스 간에는 사설 IPv4를 사용할 수 있다.

- AWS 보안 그룹 규칙: https://docs.aws.amazon.com/vpc/latest/userguide/security-group-rules.html
- EC2 IP 주소: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-instance-addressing.html

### 2. Docker와 Nginx 설치

Ubuntu 24.04 기준으로 Docker 공식 apt 저장소를 등록한다.

```bash
sudo apt update
sudo apt install -y ca-certificates curl nginx
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

```bash
printf '%s\n' \
  'Types: deb' \
  'URIs: https://download.docker.com/linux/ubuntu' \
  "Suites: $(. /etc/os-release && echo \"${UBUNTU_CODENAME:-$VERSION_CODENAME}\")" \
  'Components: stable' \
  "Architectures: $(dpkg --print-architecture)" \
  'Signed-By: /etc/apt/keyrings/docker.asc' \
  | sudo tee /etc/apt/sources.list.d/docker.sources
```

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker nginx
sudo usermod -aG docker "$USER"
```

그룹 변경은 SSH 재접속 후 적용된다. 설치 절차의 기준은 Docker 공식
Ubuntu 안내다: https://docs.docker.com/engine/install/ubuntu/

### 3. 애플리케이션 시작

코드를 EC2의 `/opt/daejeon-recommendation`에 배치한 뒤 프로젝트 루트에서
실행한다.

```bash
cd /opt/daejeon-recommendation
cp recommendation_api/.env.example recommendation_api/.env
chmod 600 recommendation_api/.env
docker compose -f recommendation_api/deploy/docker-compose.yml up -d --build
docker compose -f recommendation_api/deploy/docker-compose.yml ps
curl --fail http://127.0.0.1:8000/health
```

`recommendation_api/.env`에는 EC2에서만 실제 `OPENAI_API_KEY`를 넣는다.

### 4. Nginx 적용

```bash
sudo cp recommendation_api/deploy/nginx.conf /etc/nginx/sites-available/recommendation-api
sudo ln -sfn /etc/nginx/sites-available/recommendation-api /etc/nginx/sites-enabled/recommendation-api
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
curl --fail http://127.0.0.1/health
```

Spring EC2에서는 Python EC2의 사설 IP로 확인한다.

```bash
curl --fail http://10.0.2.15/health
```

이 저장소만으로 실제 AWS 배포를 수행할 수는 없다. Python EC2 주소,
SSH 키 또는 SSM 권한, VPC와 보안 그룹 ID, AWS 권한이 준비되어야 한다.

## 운영

```bash
docker compose -f recommendation_api/deploy/docker-compose.yml ps
docker compose -f recommendation_api/deploy/docker-compose.yml logs -f --tail=200 recommendation-api
curl --fail http://127.0.0.1/health
```

로컬 빌드 방식의 업데이트는 다음과 같다.

```bash
docker compose -f recommendation_api/deploy/docker-compose.yml build --pull
docker compose -f recommendation_api/deploy/docker-compose.yml up -d
curl --fail http://127.0.0.1/health
```

롤백할 때는 `.env`의 `RECOMMENDATION_IMAGE`를 이전 이미지 태그로 바꾸고
다시 실행한다.

```bash
docker compose -f recommendation_api/deploy/docker-compose.yml up -d
curl --fail http://127.0.0.1/health
```

## 향후 ECR CI/CD

CI/CD는 다음 순서로 추가하면 된다.

1. 전체 Python 테스트를 실행한다.
2. Docker 이미지를 한 번 빌드한다.
3. Git commit SHA를 이미지 태그로 사용해 ECR에 push한다.
4. EC2의 `RECOMMENDATION_IMAGE`를 새 SHA 태그로 변경한다.
5. EC2에서 `docker compose pull`과 `docker compose up -d`를 실행한다.
6. `/health`가 성공할 때까지 제한 시간 동안 확인한다.
7. 실패하면 직전 SHA 태그로 복구한다.

ECR 로그인 형식은 다음과 같다.

```bash
aws ecr get-login-password --region ap-northeast-2 \
  | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.ap-northeast-2.amazonaws.com
```

공식 ECR 인증 예시는 다음 문서를 기준으로 한다:
https://docs.aws.amazon.com/AmazonECR/latest/userguide/example_ecr_GettingStarted_078_section.html
