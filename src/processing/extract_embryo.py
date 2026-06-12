from pathlib import Path
import sys

import cv2
import numpy as np
import tifffile
from cellpose import models
from cellpose.transforms import convert_image


CELLPOSE_MODEL = "embryomodel"
OUTPUT_SIZE = (800, 800)

def _segment_frame(model, frame: np.ndarray, segment_size=(100, 100)) -> np.ndarray:
    norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    # small = cv2.resize(frame, segment_size)
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

    out_frames = [(frame * mask) for frame in stack]

    result = np.stack(out_frames, axis=0)
    if single_frame:
        resized_result = cv2.resize(result[0], OUTPUT_SIZE)
    else:
        resized_result = np.stack([cv2.resize(f, OUTPUT_SIZE) for f in result], axis=0)

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
