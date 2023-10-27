import argparse
import json
import os
import numpy as np
from metaseg import SegMultiAutoMaskPredictor


def load_json(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None
    return None


def save_json(file_path, data):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON: {e}")


def main(args) -> None:
    try:
        SegMultiAutoMaskPredictor().video_predict(
            source=args.src,
            model_type=args.model_type,
            points_per_side=8,
            points_per_batch=args.points_per_batch,
            min_area=args.min_area,
            output_path=args.output_path,
            checkpoint=args.model,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            stability_score_offset=args.stability_score_offset,
            box_nms_thresh=args.box_nms_thresh,
            crop_n_layers=args.crop_n_layers,
            crop_nms_thresh=args.crop_nms_thresh,
            crop_overlap_ratio=args.crop_overlap_ratio,
            crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
            point_grids=None,
            min_mask_region_area=args.min_mask_region_area,
            output_mode=args.output_mode,
            save_layers_interval=5
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated video masking using metaseg.')
    config_file = "config.json"
    config = load_json(config_file)

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

    main(args)
