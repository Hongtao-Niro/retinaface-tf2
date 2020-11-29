from pathlib import Path

import click
import tensorflow as tf
import numpy as np
import cv2

from modules.utils import set_memory_growth, load_yaml, draw_bbox_landm, pad_input_image, recover_pad_output
from export import export_to_saved_model


def _run_detection(detector_model, image_arr, score_thres, iou_thres, detection_width):
    img = np.float32(image_arr.copy())

    if detection_width > 0.0:
        scale = float(detection_width) / image_arr.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR,)

    img, pad_params = pad_input_image(img, max_steps=32)
    outputs = detector_model(
        [
            np.expand_dims(img, axis=0),
            tf.constant([score_thres], dtype=tf.float32),
            tf.constant([iou_thres], dtype=tf.float32),
        ]
    ).numpy()
    outputs = recover_pad_output(outputs, pad_params)

    return outputs


@click.command()
@click.option("--image_path", type=str, required=True)
@click.option("--config_path", type=str, default="configs/retinaface_res50.yaml")
@click.option("--export_path", type=str, default="saved_models/retinaface_res50")
@click.option("--ckpt_path", type=str, default="checkpoints/retinaface_res50/ckpt-81")
@click.option("--score_thres", type=float, default=0.5)
@click.option("--iou_thres", type=float, default=0.4)
@click.option("--detection_width", type=int, default=0)
@click.option("--result_save_path", type=str, default="results")
def main(
    image_path,
    config_path,
    export_path,
    ckpt_path,
    score_thres,
    iou_thres,
    detection_width,
    result_save_path,
):
    config = load_yaml(config_path)

    if not Path(export_path).joinpath("saved_model.pb").exists() and ckpt_path is not None:
        export_to_saved_model(ckpt_path, export_path, config)
    elif not Path(export_path).joinpath("saved_model.pb").exists() and ckpt_path is None:
        raise ValueError(f"Must provide a checkpoint to export model.")

    loaded_model = tf.saved_model.load(export_path)
    print("model_loaded")

    img_raw = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_height_raw, img_width_raw, _ = img_rgb.shape
    outputs = _run_detection(loaded_model, img_rgb, score_thres, iou_thres, detection_width)

    # draw and save results
    result_save_path = Path(result_save_path)
    result_save_path.mkdir(exist_ok=True, parents=True)
    save_img_path = result_save_path.joinpath("result_" + Path(image_path).name)
    for prior_index in range(len(outputs)):
        draw_bbox_landm(
            img_raw, outputs[prior_index], img_height_raw, img_width_raw, draw_score=True, draw_lm=True
        )
        cv2.imwrite(str(save_img_path), img_raw)
    print(f"Results saved at {save_img_path}")


if __name__ == "__main__":
    main()
