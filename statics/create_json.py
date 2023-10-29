import json
import os


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


def create_init_model(path=""):
    print("no models found")


def create_standard_model():
    return {
        'url_key': 'codeformer',
        'model_dir': 'CodeFormer/weights/CodeFormer',
        'file_name': 'codeformer.pth',
        'colour': 'green',
        'size': 'green',
        'class_name': ''
    },


def create_standard_options():
    return {
        "src": "video.mp4",
        "model": "vit_h",
        "model_type": "vit_l",
        "points_per_side": 16,
        "points_per_batch": 64,
        "min_area": 1000,
        "output_path": "output.mp4",
        "pred_iou_thresh": 0.5,
        "stability_score_thresh": 0.5,
        "stability_score_offset": 0.1,
        "box_nms_thresh": 0.5,
        "crop_n_layers": 0,
        "crop_nms_thresh": 0.5,
        "crop_overlap_ratio": 0.2,
        "crop_n_points_downscale_factor": 2,
        "point_grids": 0.4,
        "min_mask_region_area": 0,
        "output_mode": "binary_mask"
    }
