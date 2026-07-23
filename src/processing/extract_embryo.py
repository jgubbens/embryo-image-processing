from pathlib import Path
import sys

import cv2
import numpy as np
import tifffile
from cellpose import models
from cellpose.models import MODEL_DIR
from cellpose.transforms import convert_image


CELLPOSE_MODEL = str(MODEL_DIR / "embryomodel")
OUTPUT_SIZE = (800, 800)

MASK_FILL_FRACTION = 0.9
MASK_PADDING_FRACTION = 0.05
CONFIDENCE_THRESHOLD = 0.5
MAX_SIZE_DEVIATION = 0.5 # reject masks whose area differs from the median by more than this fraction
MAX_CENTROID_DEVIATION = 0.2 # reject masks whose centroid differs from the median by more than this fraction of the frame size
TRANSFORM_SIMILARITY_THRESHOLD = 0.97

class EmbryoExtractor:
    def __init__(self):
        self.load_model()
        self.mask = None
        self.transform = None

    def _segment_frame(self, model, frame: np.ndarray, segment_size=(100, 100)) -> tuple[np.ndarray, float]:
        norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(norm)
        small = cv2.resize(enhanced, segment_size)
        transformed = convert_image(small)
        masks, flows, _ = model.eval(transformed, normalize=True)

        if masks.max() == 0:
            print('No mask found')
            return np.zeros(frame.shape[:2], dtype=bool), 0.0

        labels, counts = np.unique(masks[masks > 0], return_counts=True)
        biggest = labels[np.argmax(counts)]
        big_mask_small = masks == biggest
        confidence = float(flows[2][big_mask_small].mean())
        big_mask = cv2.resize(big_mask_small.astype(np.float32), (frame.shape[1], frame.shape[0])) > 0.5
        return big_mask, confidence

    def _filter_masks(self, masks: list[np.ndarray], confidences: list[float]) -> list[int]:
        candidates = [i for i, (m, c) in enumerate(zip(masks, confidences))
                    if m.any() and c >= CONFIDENCE_THRESHOLD]
        if not candidates:
            return []

        areas = np.array([masks[i].sum() for i in candidates], dtype=float)
        centroids = np.array([np.mean(np.nonzero(masks[i]), axis=1) for i in candidates])

        median_area = np.median(areas)
        median_centroid = np.median(centroids, axis=0)

        h, w = masks[0].shape
        size_ok = np.abs(areas - median_area) / median_area <= MAX_SIZE_DEVIATION
        centroid_ok = np.linalg.norm(centroids - median_centroid, axis=1) / max(h, w) <= MAX_CENTROID_DEVIATION

        return [candidates[i] for i in range(len(candidates)) if size_ok[i] and centroid_ok[i]]

    def _pad_mask(self, mask: np.ndarray, fraction: float = MASK_PADDING_FRACTION) -> np.ndarray:
        _, _, w, h = cv2.boundingRect(mask.astype(np.uint8))
        margin = max(1, round(fraction * max(w, h)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    def _orientation_matrix(self, mask: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        points = np.column_stack((xs, ys)).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points, mean=None)
        centroid = (float(mean[0, 0]), float(mean[0, 1]))
        major_axis = eigenvectors[0]
        angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
        rotation = cv2.getRotationMatrix2D(centroid, angle - 90.0, 1.0)

        rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation).reshape(-1, 2)
        ymin, ymax = rotated_points[:, 1].min(), rotated_points[:, 1].max()
        cx = (rotated_points[:, 0].min() + rotated_points[:, 0].max()) / 2.0

        out_w, out_h = output_size
        scale = (MASK_FILL_FRACTION * out_h) / (ymax - ymin)
        margin = (1.0 - MASK_FILL_FRACTION) / 2.0 * out_h
        tx = out_w / 2.0 - cx * scale
        ty = margin - ymin * scale
        scale_translate = np.array([[scale, 0.0, tx], [0.0, scale, ty]])

        rotation_h = np.vstack([rotation, [0.0, 0.0, 1.0]])
        scale_translate_h = np.vstack([scale_translate, [0.0, 0.0, 1.0]])
        return (scale_translate_h @ rotation_h)[:2]

    def _transform_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        ref = np.linalg.norm(b)
        if ref == 0:
            return 1.0 if np.linalg.norm(a) == 0 else 0.0
        return 1.0 - np.linalg.norm(a - b) / ref

    def load_model(self, gpu: bool = True) -> models.CellposeModel:
        self.model = models.CellposeModel(gpu=gpu, pretrained_model=CELLPOSE_MODEL)

    def extract_full_video(self, input_path: Path | np.ndarray, output_path: Path | None = None) -> np.ndarray:
        if isinstance(input_path, np.ndarray):
            stack = input_path
        else:
            stack = tifffile.imread(input_path)
            print(f"Loaded {input_path}: {stack.shape}, dtype={stack.dtype}")

        single_frame = stack.ndim == 2
        if single_frame:
            stack = stack[np.newaxis]

        out_frames = []
        for i, frame in enumerate(stack):
            out_frames.append(self.extract_frame(frame))
        resized_result = np.stack(out_frames, axis=0)
        
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(output_path, resized_result, imagej=True)
            print(f"Saved at {output_path} shape={resized_result.shape}")

        return resized_result
    
    
    def extract_frame(self, frame) -> np.ndarray:
        mask, conf = self._segment_frame(self.model, frame)
        padded = self._pad_mask(mask)
        transform = self._orientation_matrix(padded, OUTPUT_SIZE)
        if self.mask is None:
            self.transform = transform
            warped_mask = cv2.warpAffine(padded.astype(np.uint8), transform, OUTPUT_SIZE) > 0
            x, y, w, h = cv2.boundingRect(warped_mask.astype(np.uint8))
            out_mask = np.zeros(warped_mask.shape, dtype=bool)
            out_mask[y:y + h, x:x + w] = True
            self.mask = out_mask
            out_frame = cv2.warpAffine(frame, transform, OUTPUT_SIZE) * out_mask
        else:
            if self._transform_similarity(transform, self.transform) >= TRANSFORM_SIMILARITY_THRESHOLD:
                transform = self.transform
            else:
                self.transform = transform
            out_frame = cv2.warpAffine(frame, transform, OUTPUT_SIZE) * self.mask

        return out_frame


if __name__ == "__main__":
    inp = r'data/training_data/brightfield/0629_pos010.tif'
    out = r'data/training_data/processed_tifs/0629_pos010.tif'

    extractor = EmbryoExtractor()

    extractor.extract_full_video(inp, out)
