import pytest
import tensorflow as tf
import numpy as np
import cv2

from modules.utils import pad_input_image, recover_pad_output

WARMUP_ITERATIONS = 10


@pytest.fixture(params=["mobile-v2", "res50"])
def detector_model(request):
    backbone = request.param
    print(backbone)
    if backbone == "mobile-v2":
        loaded_model = tf.saved_model.load("saved_models/retinaface-mobile-v2")
    elif backbone == "res50":
        loaded_model = tf.saved_model.load("saved_models/retinaface-res50")

    return loaded_model


@pytest.fixture
def single_test_image():
    img_bgr = cv2.imread("tests/data/test_image_1.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# @pytest.fixture(params=[320, 640, 854, 1280])
# def run_detection(detector_model, request):
#     detection_width = request.param

#     def _run_detection(image_arr):
#         img = np.float32(image_arr.copy())

#         if detection_width > 0.0:
#             scale = float(detection_width) / image_arr.shape[1]
#             img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR,)

#         img, pad_params = pad_input_image(img, max_steps=32)
#         outputs = detector_model(np.expand_dims(img, axis=0)).numpy()
#         outputs = recover_pad_output(outputs, pad_params)

#         return outputs

#     return _run_detection


def _run_detection(model, image_arr, detection_width):
    img = np.float32(image_arr.copy())

    if detection_width > 0.0:
        scale = float(detection_width) / image_arr.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR,)

    img, pad_params = pad_input_image(img, max_steps=32)
    outputs = model(np.expand_dims(img, axis=0)).numpy()
    outputs = recover_pad_output(outputs, pad_params)

    return outputs


@pytest.mark.benchmark(group="sequencial", warmup=True, warmup_iterations=WARMUP_ITERATIONS)
@pytest.mark.parametrize("detection_width", [320, 640, 854, 1280])
def test_detector(benchmark, detector_model, single_test_image, detection_width):
    benchmark(_run_detection, detector_model, single_test_image, detection_width)
