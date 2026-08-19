from io import BytesIO
import os
from unittest import mock
import unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from recommendation_api.face_mosaic import (
    FaceMosaicResult,
    FaceNotDetectedError,
    mosaic_face_image_bytes,
)
from recommendation_api.face_mosaic_s3 import store_s3_mosaic_image
from recommendation_api.main import (
    app,
    get_face_mosaic_processor,
    get_s3_face_image_loader,
    get_s3_mosaic_uploader,
)
from recommendation_api.receipt_s3 import S3ReceiptObject


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
        self.input_size = None

    def setInputSize(self, input_size):
        self.input_size = input_size

    def detect(self, image):
        del image
        return 1, self.faces


def jpeg_bytes() -> bytes:
    image = Image.new("RGB", (120, 80), color=(200, 150, 100))
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class RecordingS3PutClient:
    def __init__(self):
        self.calls = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        return {"ETag": "test-etag"}


class FaceMosaicProcessorTest(unittest.TestCase):
    def test_mosaics_detected_face_and_returns_jpeg(self):
        faces = np.array(
            [[20, 10, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]],
            dtype=np.float32,
        )
        detector = FakeDetector(faces)

        result = mosaic_face_image_bytes(jpeg_bytes(), "image/jpeg", detector=detector)

        self.assertEqual(result.face_count, 1)
        self.assertEqual(result.content_type, "image/jpeg")
        self.assertEqual((result.width, result.height), (120, 80))
        self.assertTrue(result.image_bytes.startswith(b"\xff\xd8"))
        self.assertEqual(detector.input_size, (120, 80))

    def test_fails_closed_when_no_face_is_detected(self):
        with self.assertRaises(FaceNotDetectedError):
            mosaic_face_image_bytes(
                jpeg_bytes(),
                "image/jpeg",
                detector=FakeDetector(None),
            )


class FaceMosaicApiTest(unittest.TestCase):
    def tearDown(self):
        app.dependency_overrides.clear()

    @staticmethod
    def result() -> FaceMosaicResult:
        return FaceMosaicResult(
            image_bytes=b"mosaic-jpeg",
            content_type="image/jpeg",
            face_count=2,
            width=1280,
            height=720,
        )

    def test_processes_s3_image_and_returns_output_key(self):
        app.dependency_overrides[get_s3_face_image_loader] = lambda: (
            lambda key: S3ReceiptObject(key, "image/heic", b"source-image")
        )
        app.dependency_overrides[get_face_mosaic_processor] = lambda: (
            lambda image_bytes, content_type: self.result()
        )
        app.dependency_overrides[get_s3_mosaic_uploader] = lambda: (
            lambda result: "mosaics/result-mosaic.jpg"
        )

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/images/face-mosaic",
                json={
                    "requestId": "req-face-001",
                    "userId": 4,
                    "s3Key": "uploads/photo.heic",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["faceCount"], 2)
        self.assertEqual(
            response.json()["outputS3Key"],
            "mosaics/result-mosaic.jpg",
        )

    def test_local_endpoint_returns_processed_jpeg(self):
        app.dependency_overrides[get_face_mosaic_processor] = lambda: (
            lambda image_bytes, content_type: self.result()
        )

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/images/face-mosaic-local",
                files={"image": ("photo.heic", b"source-image", "image/heic")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"mosaic-jpeg")
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-face-count"], "2")


class FaceMosaicS3Test(unittest.TestCase):
    def test_stores_result_only_under_configured_output_prefix(self):
        client = RecordingS3PutClient()
        environment = {
            "aws.region": "ap-northeast-2",
            "aws.s3.bucket": "image-test-bucket",
            "aws.s3.mosaic-prefix": "private/mosaics/",
        }
        result = FaceMosaicResult(
            image_bytes=b"mosaic-jpeg",
            content_type="image/jpeg",
            face_count=1,
            width=640,
            height=480,
        )

        with mock.patch.dict(os.environ, environment, clear=True):
            output_key = store_s3_mosaic_image(result, client=client)

        self.assertTrue(output_key.startswith("private/mosaics/"))
        self.assertTrue(output_key.endswith("-mosaic.jpg"))
        self.assertEqual(client.calls[0]["Bucket"], "image-test-bucket")
        self.assertEqual(client.calls[0]["Key"], output_key)
        self.assertEqual(client.calls[0]["Body"], b"mosaic-jpeg")
        self.assertEqual(client.calls[0]["ContentType"], "image/jpeg")


if __name__ == "__main__":
    unittest.main()
