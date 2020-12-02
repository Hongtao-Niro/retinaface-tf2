import pytest
import tensorflow as tf
import numpy as np
import cv2

from modules.utils import resize_and_pad_input_image, recover_pad_output

WARMUP_ITERATIONS = 10
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@pytest.fixture(params=["mobile-v2", "res50"])
def detector_model(request):
    backbone = request.param
    print(backbone)
    if backbone == "mobile-v2":
        loaded_model = tf.saved_model.load("saved_models/retinaface_mobile-v2_end2end")
    elif backbone == "res50":
        loaded_model = tf.saved_model.load("saved_models/retinaface_res50_end2end")

    return loaded_model


@pytest.fixture
def single_test_image():
    img_bgr = cv2.imread("tests/data/execs.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _run_detection(model, image_arr, score_thres, iou_thres, detection_width):
    img = np.float32(image_arr.copy())

    img, pad_params = resize_and_pad_input_image(
        img, padded_height=detection_width, padded_width=detection_width, max_steps=32, keep_aspect_ratio=True
    )
    outputs = model(
        [
            np.expand_dims(img, axis=0),
            tf.constant([score_thres], dtype=tf.float32),
            tf.constant([iou_thres], dtype=tf.float32),
        ]
    ).numpy()
    outputs = recover_pad_output(outputs, pad_params)

    return outputs


@pytest.mark.benchmark(group="sequencial", warmup=True, warmup_iterations=WARMUP_ITERATIONS)
@pytest.mark.parametrize("detection_width", [320, 480, 640, 960])
def test_detector(benchmark, detector_model, single_test_image, detection_width):
    benchmark(_run_detection, detector_model, single_test_image, 0.5, 0.4, detection_width)
