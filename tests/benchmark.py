import pytest
import tensorflow as tf
import numpy as np
import cv2

from modules.utils import pad_input_image, recover_pad_output

WARMUP_ITERATIONS = 10


@pytest.fixture
def detector_model():
    loaded_model = tf.saved_model.load("saved_models/retinaface")
    return loaded_model


@pytest.fixture
def single_test_image():
    img_bgr = cv2.imread("tests/data/test_image_1.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


@pytest.fixture(params=[640, 854, 1280, 1920])
def run_detection(detector_model, request):
    detection_width = request.param

    def _run_detection(image_arr):
        img = np.float32(image_arr.copy())

        if detection_width > 0.0:
            scale = float(detection_width) / image_arr.shape[1]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR,)

        img, pad_params = pad_input_image(img, max_steps=32)
        outputs = detector_model(np.expand_dims(img, axis=0)).numpy()
        outputs = recover_pad_output(outputs, pad_params)

        return outputs

    return _run_detection


@pytest.mark.benchmark(group="sequencial", warmup=True, warmup_iterations=WARMUP_ITERATIONS)
def test_detector(benchmark, run_detection, single_test_image):
    benchmark(run_detection, single_test_image)
