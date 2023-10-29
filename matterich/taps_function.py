import cv2
import gradio as gr
from metaseg import (
    SahiAutoSegmentation,
    SegAutoMaskPredictor,
    SegManualMaskPredictor,
    sahi_sliced_predict,
)


def sahi_autoseg_app(
        image_path,
        sam_model_type,
        detection_model_type,
        detection_model_path,
        conf_th,
        image_size,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
):
    boxes = sahi_sliced_predict(
        image_path=image_path,
        # yolov8, detectron2, mmdetection, torchvision
        detection_model_type=detection_model_type,
        detection_model_path=detection_model_path,
        conf_th=conf_th,
        image_size=image_size,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    SahiAutoSegmentation().image_predict(
        source=image_path,
        model_type=sam_model_type,
        input_box=boxes,
        multimask_output=False,
        random_color=False,
        show=False,
        save=True,
    )

    return "output.png"


def extract_frames(video_path):
    vid = cv2.VideoCapture(video_path)
    success, image = vid.read()
    frames = []
    while success:
        frames.append(cv2.imencode('.png', image)[1].tobytes())
        success, image = vid.read()
    return frames


def automask_video_app_with_frames(
        video_path, model_type, points_per_side, points_per_batch, min_area
):
    output_video_path = "output.mp4"
    SegAutoMaskPredictor().video_predict(
        source=video_path,
        model_type=model_type,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
        output_path=output_video_path,
    )
    return output_video_path, extract_frames(output_video_path)


def automask_image_app(
        image_path, model_type, points_per_side, points_per_batch, min_area
):
    SegAutoMaskPredictor().image_predict(
        source=image_path,
        model_type=model_type,  # vit_l, vit_h, vit_b
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
        output_path="output.png",
        show=False,
        save=True,
    )
    return "output.png"


# For video


def automask_video_app(
        video_path, model_type, points_per_side, points_per_batch, min_area
):
    SegAutoMaskPredictor().video_predict(
        source=video_path,
        model_type=model_type,  # vit_l, vit_h, vit_b
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
        output_path="output.mp4",
    )
    return "output.mp4"
