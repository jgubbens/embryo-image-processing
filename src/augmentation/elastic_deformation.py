import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import os
import shutil
import random

class ElasticDeformer:
    def __init__(self, displacement_strength, displacement_density):
        self.displacement_strength = displacement_strength
        self.displacement_density = displacement_density

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state):
        if random_state is None:
            random_state = np.random.RandomState(None)
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        shape = image.shape
        shape_size = shape[:2]  # (height, width)

        # Random affine transformation
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, 
                        [center_square[0] + square_size, center_square[1] - square_size], 
                        center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        # Reduce the randomness and increase the deformation intensity
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)  # No deformation along the color axis for RGB images

        # 3D meshgrid for RGB images
        if len(shape) == 3:
            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        else:
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            z = np.zeros_like(x)
        
        # Generate sparse displacement field: fewer large displacements
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma * self.displacement_density) * alpha * self.displacement_strength
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma * self.displacement_density) * alpha * self.displacement_strength
        
        # Create new coordinates for the deformation
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        # Map the coordinates to the original image and apply the transformation
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    def run_transform(self, image_path):
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Apply elastic transformation with larger, fewer deformations
        transformed_image = self.elastic_transform(image, alpha=100, sigma=9, alpha_affine=10, random_state=None)

        #cv2.imwrite('elastic_deformation/transformed.png', transformed_image)

        # Display the original and transformed images
        cv2.imshow("Original Image", image)
        cv2.imshow("Elastic Transformed Image", transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transform_dir(self, dir):
        images_dir = os.path.join(dir, 'images')
        labels_dir = os.path.join(dir, 'labels')
        output_path = 'ElasticDeformation/output'
        
        # Create output directories for transformed images and labels
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)

        # List all image files in the 'images' subdirectory
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.tif'))]

        # Iterate through all image files
        for idx, image_file in enumerate(image_files):
            seed = random.randint(0, 2**32 - 1)

            # Construct the full path to the image and label
            image_path = os.path.join(images_dir, image_file)
            label_file = image_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(labels_dir, label_file)

            # Load the image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # Save the original image with a numbered name
            transformed_image_path = os.path.join(output_path, 'images', f'transformed{idx+1}.png')
            transformed_image = self.elastic_transform(image, alpha=100, sigma=9, alpha_affine=10, random_state=seed)
            cv2.imwrite(transformed_image_path, transformed_image)
            print(f'Transformed {transformed_image_path}')

            # Read the corresponding YOLO label file
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.readlines()

                # Iterate over each label and extract the bounding box
                for label_idx, label in enumerate(labels):
                    class_id, x_center, y_center, width, height = map(float, label.split())

                    # Convert YOLO format to pixel coordinates
                    xmin = int((x_center - width / 2) * image.shape[1])
                    xmax = int((x_center + width / 2) * image.shape[1])
                    ymin = int((y_center - height / 2) * image.shape[0])
                    ymax = int((y_center + height / 2) * image.shape[0])

                    # Create a blank white image for the bounding box
                    label_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255  # White background

                    # Draw the bounding box on the white image (black box on white background)
                    cv2.rectangle(label_image, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)

                    # Apply elastic deformation to the bounding box
                    deformed_bbox = self.elastic_transform(label_image, alpha=100, sigma=9, alpha_affine=10, random_state=seed)

                    # Save the deformed bounding box as a separate image
                    os.makedirs(os.path.join(output_path, 'bbox_transformed'), exist_ok=True)
                    bbox_output_path = os.path.join(output_path, 'bbox_transformed', f'transformed_{idx+1}_bbox_{label_idx+1}.png')
                    cv2.imwrite(bbox_output_path, deformed_bbox)

                    # Save the metadata (class_id, original coordinates) for each bounding box in a separate label file
                    #bbox_label_path = os.path.join(output_path, 'labels', f'transformed_{idx+1}_bbox_{label_idx+1}.txt')
                    bbox_label_path = os.path.join(output_path, 'labels', f'transformed{idx+1}.txt')
                    bbox_label_text = self.transformed_box_label(bbox_output_path, image, class_id)
                    with open(bbox_label_path, 'a') as label_out:
                        label_out.write(f"{bbox_label_text}\n")
                    print(f'Updated label for {bbox_label_path}')

        print(f"Transformation complete. Transformed images saved to {os.path.join(output_path, 'images')}")