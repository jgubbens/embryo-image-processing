from pathlib import Path
import nd2
import tifffile as tiff
import numpy as np

if __name__ == '__main__':

    input_file = Path("data/nd2_videos/run1007.nd2")
    output_dir = Path("data/nd2_videos/tif")
    output_dir.mkdir(exist_ok=True, parents=True)

    BIN_FACTOR = 4

    def bin_frame(frame: np.ndarray, factor: int) -> np.ndarray:
        h, w = frame.shape[-2], frame.shape[-1]
        h_crop = (h // factor) * factor
        w_crop = (w // factor) * factor
        cropped = frame[..., :h_crop, :w_crop]
        
        return cropped.reshape(
            *frame.shape[:-2],
            h_crop // factor, factor,
            w_crop // factor, factor
        ).mean(axis=(-3, -1)).astype(frame.dtype)

    with nd2.ND2File(input_file) as f:
        axes = f.sizes
        print("Dimensions:", axes)

        positions = axes.get("P", 1)
        channels = axes.get("C", 1)
        timepoints = axes.get("T", 1)

        try:
            channel_names = [ch.channel.name for ch in f.metadata.channels]
        except Exception:
            channel_names = [f"ch{i}" for i in range(channels)]

        for p in range(positions):
            for c in range(channels):
                frames = []

                for t in range(timepoints):
                    frame_index = p * timepoints + t

                    frame = f.read_frame(frame_index)
                    frame = bin_frame(frame, BIN_FACTOR)
                    frames.append(frame)

                stack = np.stack(frames, axis=0)

                channel_name = (
                    channel_names[c]
                    if c < len(channel_names)
                    else f"ch{c}"
                )

                out_name = (
                    f"{input_file.stem}"
                    f"_pos{p:03d}"
                    f"_{channel_name}"
                    f"_bin{BIN_FACTOR}.tif"
                )

                tiff.imwrite(output_dir / out_name, stack, imagej=True)
                print(f"Saved {out_name}")