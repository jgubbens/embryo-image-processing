from augmentation.elastic_deformation import ElasticDeformer
import cv2
from pathlib import Path

# Histogram Equalization
def equalize_color_image(img):
    # Split the image into its color channels
    channels = cv2.split(img)

    # Apply histogram equalization to each channel
    equalized_channels = [cv2.equalizeHist(channel) for channel in channels]

    # Merge the equalized channels back into a color image
    equalized_img = cv2.merge(equalized_channels)

    return equalized_img

UNPROCESSED_PATH = 'data/unprocessed_pre_post_nc10'

samples = []

for embryo_dir in sorted(Path(UNPROCESSED_PATH).iterdir()):
    if not embryo_dir.is_dir():
        continue
    for cls_name, label in [('pre-nc10', 0), ('post-nc10', 1)]:
        cls_dir = embryo_dir / cls_name
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob('*.tif*')):
            samples.append((f, label, embryo_dir.name))
        if not samples:
            raise RuntimeError(f'No TIFs found under {UNPROCESSED_PATH}')



deformer = ElasticDeformer(10, 5)
deformer.run_transform('data/unprocessed_pre_post_nc10/inner.i04channel_638_patterns/post-nc10/inner.i04channel_638_patterns_frame_014.tif')



