import gradio as gr
from metaseg import (
    SahiAutoSegmentation,
    SegAutoMaskPredictor,
    SegManualMaskPredictor,
    sahi_sliced_predict,
)


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


# For manuel box and point selection


def manual_app(
        image_path,
        model_type,
        input_point,
        input_label,
        input_box,
        multimask_output,
        random_color,
):
    SegManualMaskPredictor().image_predict(
        source=image_path,
        model_type=model_type,  # vit_l, vit_h, vit_b
        input_point=input_point,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
        random_color=random_color,
        output_path="output.png",
        show=False,
        save=True,
    )
    return "output.png"


# For sahi sliced prediction


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


def image_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                seg_automask_image_file = gr.Image(type="filepath").style(height=260)
                with gr.Row():
                    with gr.Column():
                        seg_automask_image_model_type = gr.Dropdown(
                            choices=[
                                "vit_h",
                                "vit_l",
                                "vit_b",
                            ],
                            value="vit_l",
                            label="Model Type",
                        )

                        seg_automask_image_min_area = gr.Number(
                            value=0,
                            label="Min Area",
                        )
                    with gr.Row():
                        with gr.Column():
                            seg_automask_image_points_per_side = gr.Slider(
                                minimum=0,
                                maximum=32,
                                step=2,
                                value=16,
                                label="Points per Side",
                            )

                            seg_automask_image_points_per_batch = gr.Slider(
                                minimum=0,
                                maximum=64,
                                step=2,
                                value=64,
                                label="Points per Batch",
                            )

                seg_automask_image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image()

        seg_automask_image_predict.click(
            fn=automask_image_app,
            inputs=[
                seg_automask_image_file,
                seg_automask_image_model_type,
                seg_automask_image_points_per_side,
                seg_automask_image_points_per_batch,
                seg_automask_image_min_area,
            ],
            outputs=[output_image],
        )


def video_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                seg_automask_video_file = gr.Video().style(height=260)
                with gr.Row():
                    with gr.Column():
                        seg_automask_video_model_type = gr.Dropdown(
                            choices=[
                                "vit_h",
                                "vit_l",
                                "vit_b",
                            ],
                            value="vit_l",
                            label="Model Type",
                        )
                        seg_automask_video_min_area = gr.Number(
                            value=1000,
                            label="Min Area",
                        )

                    with gr.Row():
                        with gr.Column():
                            seg_automask_video_points_per_side = gr.Slider(
                                minimum=0,
                                maximum=32,
                                step=2,
                                value=16,
                                label="Points per Side",
                            )

                            seg_automask_video_points_per_batch = gr.Slider(
                                minimum=0,
                                maximum=64,
                                step=2,
                                value=64,
                                label="Points per Batch",
                            )

                seg_automask_video_predict = gr.Button(value="Generator")
            with gr.Column():
                output_video = gr.Video()

        seg_automask_video_predict.click(
            fn=automask_video_app,
            inputs=[
                seg_automask_video_file,
                seg_automask_video_model_type,
                seg_automask_video_points_per_side,
                seg_automask_video_points_per_batch,
                seg_automask_video_min_area,
            ],
            outputs=[output_video],
        )


def sahi_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                sahi_image_file = gr.Image(type="filepath").style(height=260)
                sahi_autoseg_model_type = gr.Dropdown(
                    choices=[
                        "vit_h",
                        "vit_l",
                        "vit_b",
                    ],
                    value="vit_l",
                    label="Sam Model Type",
                )

                with gr.Row():
                    with gr.Column():
                        sahi_model_type = gr.Dropdown(
                            choices=[
                                "yolov5",
                                "yolov8",
                            ],
                            value="yolov5",
                            label="Detector Model Type",
                        )
                        sahi_image_size = gr.Slider(
                            minimum=0,
                            maximum=1600,
                            step=32,
                            value=640,
                            label="Image Size",
                        )

                        sahi_overlap_width = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.2,
                            label="Overlap Width",
                        )

                        sahi_slice_width = gr.Slider(
                            minimum=0,
                            maximum=640,
                            step=32,
                            value=256,
                            label="Slice Width",
                        )

                    with gr.Row():
                        with gr.Column():
                            sahi_model_path = gr.Dropdown(
                                choices=[
                                    "yolov5l.pt",
                                    "yolov5l6.pt",
                                    "yolov8l.pt",
                                    "yolov8x.pt",
                                ],
                                value="yolov5l6.pt",
                                label="Detector Model Path",
                            )

                            sahi_conf_th = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Confidence Threshold",
                            )
                            sahi_overlap_height = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.2,
                                label="Overlap Height",
                            )
                            sahi_slice_height = gr.Slider(
                                minimum=0,
                                maximum=640,
                                step=32,
                                value=256,
                                label="Slice Height",
                            )
                sahi_image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image()

        sahi_image_predict.click(
            fn=sahi_autoseg_app,
            inputs=[
                sahi_image_file,
                sahi_autoseg_model_type,
                sahi_model_type,
                sahi_model_path,
                sahi_conf_th,
                sahi_image_size,
                sahi_slice_height,
                sahi_slice_width,
                sahi_overlap_height,
                sahi_overlap_width,
            ],
            outputs=[output_image],
        )


def matterich_app():
    app = gr.Blocks(title="title",
                    outputs='video',
                    description="description",
                    examples_per_page=4,
                    allow_flagging=False,
                    article="article",
                    )
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Image"):
                    image_app()
                with gr.Tab("Video"):
                    video_app()
                with gr.Tab("SAHI"):
                    sahi_app()

    app.queue(concurrency_count=1)
    try:
        app.launch(
            inbrowser=True,
            debug=True,
            enable_queue=True,
            server_name="localhost",
            server_port=7860,

        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    matterich_app()
