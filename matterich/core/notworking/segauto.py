from typing import Union, List, Optional, Any
import cv2
import numpy as np
import torch
from cv2 import Mat
from tqdm import tqdm

from typing import Union, List, Optional
import cv2
import numpy as np
import torch
from tqdm import tqdm

from metaseg.generator.automatic_mask_generator import SamAutomaticMaskGenerator
from metaseg.generator.build_sam import sam_model_registry
from metaseg.generator.predictor import SamPredictor
from metaseg.utils import (
    download_model,
    load_box,
    load_image,
    load_mask,
    load_video,
    multi_boxes,
    show_image,
)


class SegMaskPredictorBase:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type: str):
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)
        return self.model


class SegAutoMaskPredictor2(SegMaskPredictorBase):
    def __init__(self):
        super().__init__()

    def image_predict(
            self,
            source: Union[str, np.ndarray],
            model_type: str,
            points_per_side: int,
            points_per_batch: int,
            min_area: int,
            output_path: str = "output2.png",
            show: bool = True,
            save: bool = True,
    ):
        read_image = load_image(source)
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            min_mask_region_area=min_area,
        )
        masks = mask_generator.generate(read_image)
        mask_image = self._generate_colored_mask(masks)
        combined_mask = cv2.add(read_image, mask_image)

        if show:
            show_image(combined_mask)
        if save:
            cv2.imwrite(output_path, combined_mask)
        return masks

    def video_predict(
            self,
            source: str,
            model_type: str,
            points_per_side: int,
            points_per_batch: int,
            min_area: int,
            output_path: str = "output.mp4",
    ):
        cap, out = load_video(source, output_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            min_mask_region_area=min_area,
        )

        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            if not ret:
                break

            masks = mask_generator.generate(frame)
            mask_image = self._generate_colored_mask(masks)
            combined_mask = cv2.add(frame, mask_image)
            out.write(combined_mask)

        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def _generate_colored_mask(self, masks: List[dict]) -> np.ndarray:
        if not masks:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        sorted_anns = sorted(masks, key=lambda x: x["area"], reverse=True)
        mask_image = np.zeros(
            (masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3),
            dtype=np.uint8,
        )
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        for i, ann in enumerate(sorted_anns):
            m = ann["segmentation"]
            img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            color = colors[i % 256]
            img[:, :] = color
            img = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))
            img = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
            mask_image = cv2.add(mask_image, img)
        return mask_image


# Importing all required modules (update these as needed)
from metaseg.generator.automatic_mask_generator import SamAutomaticMaskGenerator
from metaseg.generator.build_sam import sam_model_registry
from metaseg.generator.predictor import SamPredictor
from metaseg.utils import (
    download_model,
    load_box,
    load_image,
    load_mask,
    load_video,
    multi_boxes,
    show_image,
)


class BaseMaskPredictor:
    def __init__(self) -> None:
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type: str) -> Any:
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)
        return self.model


class SegAutoMaskPredictor(BaseMaskPredictor):
    def image_predict(
            self,
            source: Union[str, Mat],
            model_type: str,
            points_per_side: int,
            points_per_batch: int,
            min_area: int,
            output_path: str = "output2.png",
            show: bool = True,
            save: bool = True,
    ) -> List[dict]:
        image = load_image(source)
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            min_mask_region_area=min_area,
        )
        masks = mask_generator.generate(image)
        sorted_anns = sorted(masks, key=lambda x: x["area"], reverse=True)
        self._draw_and_save_masks(sorted_anns, image, output_path, show, save)
        return masks

    def video_predict(
            self,
            source: str,
            model_type: str,
            points_per_side: int,
            points_per_batch: int,
            min_area: int,
            output_path: str = "output.mp4",
    ) -> str:
        cap, out = load_video(source, output_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            min_mask_region_area=min_area,
        )
        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            if not ret:
                break
            masks = mask_generator.generate(frame)
            sorted_anns = sorted(masks, key=lambda x: x["area"], reverse=True)
            self._draw_and_save_masks(sorted_anns, frame, None, False, False, out)
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        return output_path

    def _draw_and_save_masks(
            self,
            sorted_anns: List[dict],
            image: np.ndarray,
            output_path: Optional[str],
            show: bool,
            save: bool,
            out: Optional[cv2.VideoWriter] = None,
    ) -> None:
        # Draw and save logic here
        pass


class SegManualMaskPredictor(BaseMaskPredictor):
    def image_predict(
            self,
            source: Union[str, Mat],
            model_type: str,
            input_box: Optional[List[Union[List[int], int]]] = None,
            input_point: Optional[List[int]] = None,
            input_label: Optional[List[int]] = None,
            multimask_output: bool = False,
            output_path: str = "output.png",
            random_color: bool = False,
            show: bool = False,
            save: bool = False,
    ) -> List[dict]:
        image = load_image(source)
        model = self.load_model(model_type)
        predictor = SamPredictor(model)
        predictor.set_image(image)
        # Prediction and drawing logic here
        return []

    def video_predict(
            self,
            source: str,
            model_type: str,
            input_box: Optional[List[Union[List[int], int]]] = None,
            input_point: Optional[List[int]] = None,
            input_label: Optional[List[int]] = None,
            multimask_output: bool = False,
            output_path: str = "output.mp4",
            random_color: bool = False,
    ) -> str:
        cap, out = load_video(source, output_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        model = self.load_model(model_type)
        predictor = SamPredictor(model)
        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            if not ret:
                break
            predictor.set_image(frame)
            # Prediction and drawing logic here
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        return output_path

    def _draw_and_save_masks(
            self,
            sorted_anns: List[dict],
            image: np.ndarray,
            output_path: Optional[str],
            show: bool,
            save: bool,
            out: Optional[cv2.VideoWriter] = None,
    ) -> None:
        # Draw and save logic here
        pass
