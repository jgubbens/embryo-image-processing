from pathlib import Path
import nd2
import tifffile as tiff
import numpy as np

input_dir = Path("data/nd2_videos")
output_dir = Path("data/unlabeled_tifs/nd2_converted")
output_dir.mkdir(exist_ok=True, parents=True)

for nd2_file in input_dir.glob("*.nd2"):
    print(f"Processing {nd2_file.name}")
    with nd2.ND2File(nd2_file) as f:
        # Load image data
        data = f.asarray()
        axes = f.sizes
        print("Dimensions:", axes)

        positions = axes.get("P", 1)
        channels = axes.get("C", 1)

        try:
            channel_names = [ch.channel.name for ch in f.metadata.channels]
        except Exception:
            channel_names = [f"ch{i}" for i in range(channels)]

        dim_order = list(axes.keys())

        for p in range(positions):
            for c in range(channels):

                subset = data

                if "P" in dim_order:
                    subset = np.take(
                        subset,
                        indices=p,
                        axis=dim_order.index("P")
                    )

                current_dims = [d for d in dim_order if d != "P"]

                if "C" in current_dims:
                    subset = np.take(
                        subset,
                        indices=c,
                        axis=current_dims.index("C")
                    )

                channel_name = (
                    channel_names[c]
                    if c < len(channel_names)
                    else f"ch{c}"
                )

                out_name = (
                    f"{nd2_file.stem}"
                    f"_pos{p:03d}"
                    f"_{channel_name}.tif"
                )

                out_path = output_dir / out_name

                tiff.imwrite(
                    out_path,
                    subset,
                    imagej=True
                )

                print(f"Saved {out_name}")