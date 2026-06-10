import cv2

def equalize_color_image(img):
    # Split the image into its color channels
    channels = cv2.split(img)

    # Apply histogram equalization to each channel
    equalized_channels = [cv2.equalizeHist(channel) for channel in channels]

    # Merge the equalized channels back into a color image
    equalized_img = cv2.merge(equalized_channels)

    return equalized_img

def equalize_bw_image(img):
    equalized_img = cv2.equalizeHist(img)

    return equalized_img

def run_equalization_color(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Apply histogram equalization
    equalized_image = equalize_color_image(image)

    # cv2.imwrite('histogram_equalization/equalized.png', equalized_image)

    # Display the original and equalized images
    cv2.imshow("Original Image", image)
    cv2.imshow("Equalized Image", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_equalization_bw(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization
    equalized_image = equalize_bw_image(image)

    # cv2.imwrite('histogram_equalization/equalized.png', equalized_image)

    # Display the original and equalized images
    cv2.imshow("Original Image", image)
    cv2.imshow("Equalized Image", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run_equalization_bw('data/unprocessed_nc_binned/NC13/NCEmbryo3_NC13.tif')