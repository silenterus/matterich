import multiprocessing
import os
import re
import settings

from os.path import exists
import json

REGEX_FOR_ID = re.compile('([0-9a-zA-Z]){12}$')

main_path = ""
model_path = "model"
STANDARD_PATHES = {
    'models': 'models',
    'models_json': [],
}




prompt_styles = []
sd_models = []
hypernetworks = []
face_restorers = []
realesrgan_models = []
artists_categories = []
artists = []

manager = None

themes = dict()
threads = dict()
messages = dict()
progress = dict()
Users = dict()

Files = dict()
PromptHistory = dict()

running_threads = dict()
discordAccess = dict()
modules = dict()
cogs = dict()
APP_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'settings', 'data')
APP_SETTINGS_JSON = os.path.join(APP_SETTINGS_PATH, 'app_settings.json')

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models')
MODELS_JSON = os.path.join(APP_SETTINGS_PATH, 'models.json')

DEBUG = True
CURRENT_ID = 0

import os
import pathlib


def create_path_recursive(path):
    try:
        p = pathlib.Path(path)
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
        return (path)
    except Exception as e:
        print(f"An error occurred: {e}")
        return (path)


def check_json(path, standard_creator):
        print("loading json:"+path)
        if not (exists(f"{create_path_recursive(path)}")):
                with open(f"{path}", "w") as write:
                        dict = standard_creator()
                        json.dump(dict, write, indent=2)
                        return dict
        else:
                f = open(f"{APP_SETTINGS_JSON}")
                return json.load(f)
                        







print("statics.STANDARD_PATHES:" + STANDARD_PATHES)
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


def reload_app_settings():
    f = open(f"{app_settings_path}")
    settings.globals.app_settings = json.load(f)


def save_app_settings():
    with open(f"{app_settings_path}", "w") as f:
        json.dump(settings.globals.app_settings, f, indent=2)
        f.close()


def get_next_id():
    print("get_next_id:" + str(settings.globals.app_settings['current_id']))
    settings.globals.app_settings['current_id'] += 1;
    save_app_settings()
    return settings.globals.app_settings['current_id']


class GlobalVars():

    def __init__(self, manager: multiprocessing.Manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # from multiprocessing import Manager
        print("multiprocessing")

        tm = manager
        self.themes = tm.dict(settings.globals.themes)
        self.threads = tm.dict(settings.globals.threads)
        self.messages = tm.dict(settings.globals.messages)
        self.progress = tm.dict(settings.globals.progress)
        self.Users = tm.dict(settings.globals.Users)

        self.Files = tm.dict(settings.globals.Files)
        self.PromptHistory = tm.dict(settings.globals.PromptHistory)

        self.running_threads = tm.dict(settings.globals.running_threads)
        self.discordAccess = tm.dict(settings.globals.discordAccess)
        self.modules = tm.dict(settings.globals.modules)
        self.cogs = tm.dict(settings.globals.cogs)
        print("")
