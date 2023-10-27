from typing import Union, List, Optional, Any
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


class BaseMaskPredictor:
    def __init__(self, device: Optional[str] = None) -> None:
        self.model = None
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_type: str) -> Any:
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)
        return self.model


class SegAutoMaskPredictor(BaseMaskPredictor):
    def image_predict(
            self,
            source: Union[str, np.ndarray],
            model_type: str,
            points_per_side: int = 32,
            points_per_batch: int = 64,
            min_mask_region_area: int = 0,
            output_path: str = "output2.png",
            show: bool = True,
            save: bool = True,
    ) -> List[dict]:
        try:
            image = load_image(source)
            model = self.load_model(model_type)
            mask_generator = SamAutomaticMaskGenerator(
                model,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                min_mask_region_area=min_mask_region_area,
            )
            masks = mask_generator.generate(image)
            sorted_anns = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
            self._draw_and_save_masks(sorted_anns, image, output_path, show, save)
            return masks
        except Exception as e:
            print(f"Exception occurred: {e}")
            return []

    def video_predict(
            self,
            source: str,
            model_type: str,
            points_per_side: int = 32,
            points_per_batch: int = 64,
            min_mask_region_area: int = 0,
            output_path: str = "output.mp4",
    ) -> str:
        try:
            cap, out = load_video(source, output_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            model = self.load_model(model_type)
            mask_generator = SamAutomaticMaskGenerator(
                model,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                min_mask_region_area=min_mask_region_area,
            )
            for _ in tqdm(range(length)):
                ret, frame = cap.read()
                if not ret:
                    break
                masks = mask_generator.generate(frame)
                sorted_anns = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
                self._draw_and_save_masks(sorted_anns, frame, None, False, False, out)
            out.release()
            cap.release()
            cv2.destroyAllWindows()
            return output_path
        except Exception as e:
            print(f"Exception occurred: {e}")
            return "Error"

    def _draw_and_save_masks(
            self,
            sorted_anns: List[dict],
            image: np.ndarray,
            output_path: Optional[str],
            show: bool,
            save: bool,
            out: Optional[cv2.VideoWriter] = None,
    ) -> None:
        try:
            print("")
        except Exception as e:
            print(f"Exception occurred: {e}")


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from metaseg.generator.predictor import SamPredictor
from metaseg.modeling import Sam
from metaseg.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:

    def __init__(
            self,
            model: Any,  # Update type hint as per your Sam model
            points_per_side: Optional[int] = 32,
            points_per_batch: Optional[int] = 64,
            pred_iou_thresh: Optional[float] = 0.88,
            stability_score_thresh: Optional[float] = 0.95,
            stability_score_offset: Optional[float] = 1.0,
            box_nms_thresh: Optional[float] = 0.7,
            crop_n_layers: Optional[int] = 0,
            crop_nms_thresh: Optional[float] = 0.7,
            crop_overlap_ratio: Optional[float] = 512 / 1500,
            crop_n_points_downscale_factor: Optional[int] = 1,
            point_grids: Optional[List[np.ndarray]] = None,
            min_mask_region_area: Optional[int] = 0,
            output_mode: Optional[str] = "binary_mask",
    ) -> None:

        self.model = model
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.point_grids = point_grids
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        points_per_side = 8
        # assert ( is None) != (
        #        point_grids is None
        # assert (points_per_side is None) != (
        #    point_grids is None
        # ), "Exactly one of points_per_side or point_grid must be provided."
        point_grids = None
        points_per_side = 8
        self.point_grids = None
        self.points_per_side = 8
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:

        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros(len(data["boxes"])),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
            self,
            image: np.ndarray,
            crop_box: List[int],
            crop_layer_idx: int,
            orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros(len(data["boxes"])),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
            self,
            points: np.ndarray,
            im_size: Tuple[int, ...],
            crop_box: List[int],
            orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"],
            self.predictor.model.mask_threshold,
            self.stability_score_offset,
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
            mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:

        if len(mask_data["rles"]) == 0:
            return mask_data

        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))

            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros(len(boxes)),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    "