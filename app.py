import gradio as gr

import argparse
import json
import os
from statics.create_json import load_json, save_json
import numpy as np
# from metaseg import SegMultiAutoMaskPredictor
from statics.models import download_codeformer
from matterich.taps import image_app
from matterich.taps import video_app
from matterich.taps import sahi_app


def matterich_app():
    app = gr.Blocks(title="title",
                    mode="blocks"
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
            server_name="localhost",
            server_port=7860,

        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Automated video masking using metaseg.')
    config_file = "config.json"
    config = load_json(config_file)
    download_codeformer()
    if config is None:
        parser.add_argument('--src', type=str, default="video.mp4",
                            help='Source path of the video file to be processed.')
        parser.add_argument('--model', type=str, default="vit_h", help='The SAM model to use for mask prediction.')
        parser.add_argument('--model_type', type=str, default="vit_l", choices=['vit_l', 'vit_h', 'vit_b'],
                            help='Type of model to be used for prediction.')

        parser.add_argument('--points_per_side', type=int, default=16,
                            help='Number of points per side for the algorithm. If GPU memory is not enough, reduce this value.')

        parser.add_argument('--points_per_batch', type=int, default=64,
                            help='Number of points per batch for the algorithm. If GPU memory is not enough, reduce this value.')

        parser.add_argument('--min_area', type=int, default=1000,
                            help='Minimum area to be considered for segmentation.')

        parser.add_argument('--output_path', type=str, default="output.mp4",
                            help='Output path where the processed video will be saved.')

        parser.add_argument('--pred_iou_thresh', type=float, default=0.5,
                            help='A filtering threshold using the model\'s predicted mask quality.')

        parser.add_argument('--stability_score_thresh', type=float, default=0.5,
                            help='A filtering threshold using the stability of the mask.')

        parser.add_argument('--stability_score_offset', type=float, default=0.1,
                            help='The amount to shift the cutoff when calculated the stability score.')

        parser.add_argument('--box_nms_thresh', type=float, default=0.5,
                            help='The box IoU cutoff used by non-maximal suppression.')

        parser.add_argument('--crop_n_layers', type=int, default=0,
                            help='Sets the number of layers to run for cropped mask prediction.')

        parser.add_argument('--crop_nms_thresh', type=float, default=0.5,
                            help='The box IoU cutoff for non-maximal suppression between different crops.')

        parser.add_argument('--crop_overlap_ratio', type=float, default=0.2,
                            help='Sets the degree to which crops overlap.')

        parser.add_argument('--crop_n_points_downscale_factor', type=int, default=2,
                            help='The number of points-per-side sampled in layer n is scaled down by this factor.')

        parser.add_argument('--point_grids', type=lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32),
                            help='A list over explicit grids of points used for sampling, normalized to [0,1].')

        parser.add_argument('--min_mask_region_area', type=int, default=0,
                            help='If >0, postprocessing will remove disconnected regions and holes in masks smaller than this area.')

        parser.add_argument('--output_mode', type=str, default="binary_mask",
                            choices=['binary_mask', 'uncompressed_rle', 'coco_rle'],
                            help='The form masks are returned in.')

        # ... (add remaining arguments here)
        args = parser.parse_args()
        config = vars(args)
        save_json(config_file, config)
    else:
        args = argparse.Namespace(**config)
    matterich_app()

    # main(args)
