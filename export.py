from pathlib import Path

import tensorflow as tf

from modules.utils import load_yaml, pad_input_image, recover_pad_output


def export_to_saved_model(ckpt_path, output_path, config, image_size: int = None):
    from modules.models import RetinaFaceModel, RetinaFaceModel_

    """
    Export a training checkpoint to a tensorflow savedModel.
    """
    model = RetinaFaceModel_(config, training=False, image_size=image_size)

    # load checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path)

    tf.saved_model.save(model, output_path)


def convert_to_tflite(saved_model_dir, output_path, ckpt_path, config):
    if not Path(saved_model_dir).joinpath("saved_model.pb").exists() and ckpt_path is not None:
        export_to_saved_model(ckpt_path, saved_model_dir, config, image_size=320)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    with open(output_path, "wb") as fh:
        fh.write(tflite_model)

    return tflite_model


if __name__ == "__main__":
    config = load_yaml("configs/retinaface_mbv2.yaml")
    tflite_model = convert_to_tflite(
        "saved_models/retinaface_mobile-v2_end2end_fixed-shape",
        "retinaface_mobile-v2.tflite",
        "checkpoints/retinaface_mbv2/ckpt-81",
        config,
    )

    import cv2
    import numpy as np

    # interpreter = tf.lite.Interpreter(model_path="retinaface_mobile-v2.tflite")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)

    img_bgr = cv2.imread("/home/hongtao/Downloads/Face_Mask.jpg")
    scale = 320 / max(img_bgr.shape[0], img_bgr.shape[1])
    img_rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), None, fx=scale, fy=scale)
    img_pad_h = 320 - img_rgb.shape[0]
    img_pad_w = 320 - img_rgb.shape[1]

    img_rgb = cv2.copyMakeBorder(img_rgb, 0, img_pad_h, 0, img_pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # breakpoint()

    interpreter.set_tensor(input_details[0]["index"], np.expand_dims(img_rgb.astype(np.float32), axis=0))
    interpreter.set_tensor(input_details[1]["index"], np.array([0.5], dtype=np.float32))
    interpreter.set_tensor(input_details[2]["index"], np.array([0.4], dtype=np.float32))
    interpreter.invoke()

    outputs = interpreter.get_tensor(output_details[0]["index"])
    # outputs = recover_pad_output(outputs, pad_params)
    print(outputs)

