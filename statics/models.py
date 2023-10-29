import os

from basicsr.utils.download_util import load_file_from_url
import settings.globals

import os
from functools import partial
from hashlib import md5
from pathlib import Path
from shutil import copyfileobj
import json
from requests import Response, get
from tqdm.auto import tqdm

print("statics.STANDARD_PATHES:" + settings.globals.STANDARD_PATHES)
MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "yolov8": "Yolov8DetectionModel",
    "mmdet": "MmdetDetectionModel",
    "yolov5": "Yolov5DetectionModel",
    "detectron2": "Detectron2DetectionModel",
    "huggingface": "HuggingfaceDetectionModel",
    "torchvision": "TorchVisionDetectionModel",
    "yolov5sparse": "Yolov5SparseDetectionModel",
    "yolonas": "YoloNasDetectionModel",
}

pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'yolov5': '',
    'yolov8': '',
    'yolov5l': '',
    'yolov5l6': '',
    'yolov8l': '',
    'yolov8x': ''
}

weights_info = [
    {'url_key': 'codeformer', 'model_dir': 'CodeFormer/weights/CodeFormer', 'file_name': 'codeformer.pth',
     'colour': 'green'},
    {'url_key': 'detection', 'model_dir': 'CodeFormer/weights/facelib', 'file_name': 'detection_Resnet50_Final.pth',
     'colour': 'green'},
    {'url_key': 'parsing', 'model_dir': 'CodeFormer/weights/facelib', 'file_name': 'parsing_parsenet.pth',
     'colour': 'green'},
    {'url_key': 'realesrgan', 'model_dir': 'CodeFormer/weights/realesrgan', 'file_name': 'RealESRGAN_x2plus.pth',
     'colour': 'green'},
    {'url_key': 'vit_b', 'model_dir': 'segment_anything/weights/realesrgan', 'file_name': 'vit_b.pth',
     'colour': 'green', 'class_name': 'YoloNasDetectionModel'},
    {'url_key': 'vit_l', 'model_dir': 'segment_anything/weights/realesrgan', 'file_name': 'vit_l.pth',
     'colour': 'green'},
    {'url_key': 'vit_h', 'model_dir': 'segment_anything/weights/realesrgan', 'file_name': 'vit_h.pth',
     'colour': 'green'}

]


def create_standard_model():
    return {
        "name": "model_name",  # Name of the model
        "type": "model_type",  # Type or category of the model
        "model_dir": "model/directory",  # Directory where the model's weight files are stored
        "file_name": "model.pth",  # File name of the model's weights
        "colour": "green",  # Colour label, if applicable
        "size": "",  # Size label, if applicable
        "class_name": "GenericClassName",  # Class name for instantiation, if applicable
        "url": "https://some_url/model_type.pth",  # URL for downloading the model weights, if applicable
        "repro": "https://some_url/model_type.pth"  # URL for downloading the model weights, if applicable

    }


def create_single_model_json(dict):
    complete_models = []
    for d in dict:
        url_key = d.get('url_key')
        class_name = d.get('class_name', MODEL_TYPE_TO_MODEL_CLASS_NAME.get(url_key, ''))
        model_info = {
            'name': url_key,
            'type': url_key,
            'model_dir': d.get('model_dir', ''),
            'file_name': d.get('file_name', ''),
            'colour': d.get('colour', ''),
            'size': '',
            'class_name': class_name,
            'url': pretrain_model_url.get(url_key, '')
        }
        complete_models.append(model_info)
    return complete_models


def create_complete_model_json():
    return json.dumps(
        {'standard_model': create_standard_model(), 'complete_models': create_single_model_json(weights_info)},
        indent=4)


if __name__ == '__main__':
    json_result = create_complete_model_json()

    with open('models.json', 'w') as json_file:
        json_file.write(json_result)


# md5 check function
def _check_md5(filename: str, orig_md5: str) -> bool:
    """
    filename: str, A string representing the path to the file.
    orig_md5: str, A string representing the original md5 hash.
    """
    if not os.path.exists(filename):
        return False
    with open(filename, "rb") as file_to_check:
        # read contents of the file
        data = file_to_check.read()
        # pipe contents of the file through
        md5_returned = md5(data).hexdigest()
        # Return True if the computed hash matches the original one
        if md5_returned == orig_md5:
            return True
        return False


def download_codeformer(path="models"):
    for weight in weights_info:
        full_model_dir = os.path.join(path, weight['model_dir'])
        file_path = os.path.join(full_model_dir, weight['file_name'])
        if not os.path.exists(file_path):
            if not os.path.exists(full_model_dir):
                os.makedirs(full_model_dir)
            load_file_from_url(
                url=pretrain_model_url[weight['url_key']],
                model_dir=full_model_dir,
                progress=True,
                file_name=weight['file_name']
            )


def download_model(model_type):
    """
    model_type: str, A string representing the model type.
    """

    # Check if the model file already exists and model_type is in MODEL_URLS
    filename = f"{model_type}.pth"
    if not os.path.exists(filename) and model_type in MODEL_URLS:
        print(f"Downloading {filename} model \n")
        res: Response = get(
            MODEL_URLS[model_type][0], stream=True, allow_redirects=True
        )
        if res.status_code != 200:
            res.raise_for_status()
            raise RuntimeError(
                f"Request to {MODEL_URLS[model_type][0]} "
                f"returned status code {res.status_code}"
            )

        file_size: int = int(res.headers.get("Content-Length", 0))
        folder_path: Path = Path(filename).expanduser().resolve()
        folder_path.parent.mkdir(parents=True, exist_ok=True)

        desc = "(Unknown total file size)" if file_size == 0 else ""
        res.raw.read = partial(
            res.raw.read, decode_content=True
        )  # Decompress if needed
        with tqdm.wrapattr(
                res.raw,
                "read",
                total=file_size,
                desc=desc,
                colour=MODEL_URLS[model_type][1],
        ) as r_raw:
            with folder_path.open("wb") as f:
                copyfileobj(r_raw, f)

    elif os.path.exists(filename):
        if not _check_md5(filename, MODEL_URLS[model_type][2]):
            print("File corrupted. Re-downloading... \n")
            os.remove(filename)
            download_model(model_type)

        print(f"{filename} model download complete. \n")
    else:
        raise ValueError(
            "Invalid model type. It should be 'vit_h', 'vit_l', or 'vit_b'."
        )

    return filename
