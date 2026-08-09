# Daejeon Recommendation API

Spring이 MySQL에서 조회한 장소 후보와 최근 선택 이력을 받아 기존
`recommend_llm.py` 추천 파이프라인을 실행하는 내부 FastAPI 서비스다.
Python 서비스는 MySQL에 직접 연결하지 않는다.

## API

### 상태 확인

```http
GET /health
```

```json
{"status": "ok"}
```

### 추천 요청

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
포함된다. 오류 응답은 다음 형식으로 통일된다.

```json
{
  "error": {
    "code": "INVALID_RECOMMENDATION_REQUEST",
    "message": "1km 이내의 미방문 후보가 최소 5개 필요합니다",
    "request_id": "rec-20260724-001"
  }
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
추천 코드의 fallback scorer가 사용된다.

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
public Mono<RecommendationResponse> recommend(RecommendationRequest request) {
    return recommendationWebClient.post()
            .uri("/api/v1/recommendations")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(RecommendationResponse.class);
}
```

Spring은 호출마다 `request_id`를 만들고 양쪽 서버 로그에 남겨야 한다.
FastAPI의 `400`, `422`, `500` 응답 본문도 Spring에서 그대로 기록하면
장애 원인을 요청 단위로 찾을 수 있다.

## EC2 배포

### 1. 네트워크

Python EC2는 Spring EC2와 같은 VPC에 두고 사설 IPv4로 호출한다. Python
EC2 보안 그룹의 인바운드는 다음처럼 제한한다.

| 포트 | 소스 | 용도 |
| --- | --- | --- |
| TCP 80 | Spring EC2 보안 그룹 ID | Nginx 추천 API |
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
