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

def _segment_frame(model, frame: np.ndarray, segment_size=(100, 100)) -> np.ndarray:
    norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    small = cv2.resize(enhanced, segment_size)
    transformed = convert_image(small)
    masks = model.eval(transformed, normalize=True)[0]

    if masks.max() == 0:
        print('No mask found')
        return np.zeros(frame.shape[:2], dtype=bool)
    
    labels, counts = np.unique(masks[masks > 0], return_counts=True)
    biggest = labels[np.argmax(counts)]
    big_mask = cv2.resize((masks == biggest).astype(np.float32), (frame.shape[1], frame.shape[0])) > 0.5
    return big_mask

def _pad_mask(mask: np.ndarray, fraction: float = MASK_PADDING_FRACTION) -> np.ndarray:
    _, _, w, h = cv2.boundingRect(mask.astype(np.uint8))
    margin = max(1, round(fraction * max(w, h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0

def _orientation_matrix(mask: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
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

def extract_embryo(input_path: Path | np.ndarray, output_path: Path | None = None, gpu: bool = True) -> np.ndarray:
    if isinstance(input_path, np.ndarray):
        stack = input_path
    else:
        stack = tifffile.imread(input_path)
        print(f"Loaded {input_path.name}: {stack.shape}, dtype={stack.dtype}")

    single_frame = stack.ndim == 2
    if single_frame:
        stack = stack[np.newaxis]

    print("Loading Cellpose model")
    model = models.CellposeModel(gpu=gpu, pretrained_model=CELLPOSE_MODEL)

    mask = None
    for i, frame in enumerate(stack):
        print(f"Segmenting frame {i}")
        candidate = _segment_frame(model, frame)
        if not candidate.any():
            continue
        mask = candidate
        break

    if mask is None:
        raise RuntimeError("Cellpose found no embryo in any frame.")

    mask = _pad_mask(mask)
    transform = _orientation_matrix(mask, OUTPUT_SIZE)
    out_mask = cv2.warpAffine(mask.astype(np.uint8), transform, OUTPUT_SIZE) > 0
    out_frames = [cv2.warpAffine(frame, transform, OUTPUT_SIZE) * out_mask for frame in stack]

    resized_result = np.stack(out_frames, axis=0)
    if single_frame:
        resized_result = resized_result[0]

    if output_path is not None:
        tifffile.imwrite(output_path, resized_result, imagej=True)
        print(f"Saved at {output_path} shape={resized_result.shape}")

    return resized_result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_stem(inp.stem + "_extracted")

    extract_embryo(inp, out)
