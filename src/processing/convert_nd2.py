"""Convert an ND2 video to per-position, per-channel ImageJ TIFF stacks.

Robust to arbitrary acquisition loop orders (P/T/Z can be stored in any order)
because all indexing is done by *axis name* via nd2's xarray interface rather
than by manually raveling a flat frame index.
"""

from pathlib import Path

import nd2
import numpy as np
import tifffile as tiff


def bin_frame(arr: np.ndarray, factor: int) -> np.ndarray:
    """Mean-bin the last two axes (Y, X) by `factor`. Leading axes are kept.

    Cropping discards the remainder rows/cols that don't divide evenly.
    Rounding (not truncation) is used before casting back to the source dtype.
    """
    if factor <= 1:
        return arr

    h, w = arr.shape[-2], arr.shape[-1]
    h_crop = (h // factor) * factor
    w_crop = (w // factor) * factor
    if h_crop == 0 or w_crop == 0:
        raise ValueError(f"Frame {h}x{w} too small to bin by {factor}")

    cropped = arr[..., :h_crop, :w_crop]
    binned = cropped.reshape(
        *arr.shape[:-2],
        h_crop // factor, factor,
        w_crop // factor, factor,
    ).mean(axis=(-3, -1))

    # round-then-cast for integer dtypes to avoid a systematic downward bias
    if np.issubdtype(arr.dtype, np.integer):
        binned = np.round(binned)
    return binned.astype(arr.dtype)


def isel_existing(da, **selectors):
    """xarray .isel that silently ignores dims not present in the array."""
    return da.isel(**{k: v for k, v in selectors.items() if k in da.dims})


def main():
    input_file = Path("/Volumes/toettcherlab/Justin/nd2_videos/0613_01.nd2")
    output_dir = Path("data/training_data/nd2_tifs")
    output_dir.mkdir(exist_ok=True, parents=True)

    BIN_FACTOR = 2 # FURTHER BINNING - STACKS WITH MICROSCOPE BINNING

    with nd2.ND2File(input_file) as f:
        print("Dimensions:", dict(f.sizes))

        # delayed=True -> dask-backed, so each (position, channel) slice below
        # only pulls the chunks it needs instead of loading the whole file.
        # squeeze=False -> singleton dims (e.g. a single channel) stay labeled,
        # so name-based indexing works uniformly regardless of file contents.
        xarr = f.to_xarray(delayed=True, squeeze=False)

        # Pixel/voxel size (microns) for ImageJ calibration, if available.
        try:
            voxel = f.voxel_size()  # VoxelSize(x, y, z)
        except Exception:
            voxel = None

        # Channel names: nd2 puts them on the 'C' coordinate when known.
        if "C" in xarr.coords:
            channel_names = [str(v) for v in np.atleast_1d(xarr.coords["C"].values)]
        else:
            channel_names = ["ch0"]

        n_positions = xarr.sizes.get("P", 1)
        n_channels = xarr.sizes.get("C", 1)

        for p in range(n_positions):
            for c in range(n_channels):
                sub = isel_existing(xarr, P=p, C=c)

                # Canonical ImageJ-friendly order; keep only dims that exist.
                order = [d for d in ("T", "Z", "Y", "X") if d in sub.dims]
                sub = sub.transpose(*order)

                data = sub.to_numpy()          # computes just this slice
                data = bin_frame(data, BIN_FACTOR)

                axes = "".join(order)          # e.g. "TZYX", "TYX", or "YX"

                channel_name = (
                    channel_names[c] if c < len(channel_names) else f"ch{c}"
                )
                # sanitize for use in a filename
                channel_name = "".join(
                    ch if ch.isalnum() else "_" for ch in channel_name
                ).strip("_") or f"ch{c}"

                out_name = (
                    f"{input_file.stem}"
                    f"_pos{p:03d}"
                    f"_{channel_name}"
                    f"_bin{BIN_FACTOR}.tif"
                )

                # ImageJ calibration metadata
                metadata = {"axes": axes}
                resolution = None
                if voxel is not None:
                    metadata["unit"] = "um"
                    if "Z" in order and getattr(voxel, "z", None):
                        metadata["spacing"] = float(voxel.z)
                    vx, vy = getattr(voxel, "x", None), getattr(voxel, "y", None)
                    if vx and vy:
                        # binning enlarges the effective pixel by BIN_FACTOR
                        resolution = (
                            1.0 / (vx * BIN_FACTOR),
                            1.0 / (vy * BIN_FACTOR),
                        )

                tiff.imwrite(
                    output_dir / out_name,
                    data,
                    imagej=True,
                    resolution=resolution,
                    metadata=metadata,
                )
                print(f"Saved {out_name}  shape={data.shape} axes={axes}")


if __name__ == "__main__":
    main()