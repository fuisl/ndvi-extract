import cv2
import numpy as np
# import matplotlib.pyplot as plt

# Load images
nir_image = cv2.imread("nir.jpg")
rgb_image = cv2.imread("rgb.jpg")

# Function to manually select points and compute homography
def select_points_and_compute_homography():
    points_nir = []
    points_rgb = []

    def click_event_nir(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_nir.append((x, y))
            cv2.circle(temp_nir, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("NIR Image", temp_nir)

    def click_event_rgb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_rgb.append((x, y))
            cv2.circle(temp_rgb, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("RGB Image", temp_rgb)

    # Create copies for display
    temp_nir = nir_image.copy()
    temp_rgb = rgb_image.copy()

    cv2.namedWindow("NIR Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NIR Image", 640, 480)
    cv2.imshow("NIR Image", temp_nir)
    cv2.setMouseCallback("NIR Image", click_event_nir)

    cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB Image", 640, 480)
    cv2.imshow("RGB Image", temp_rgb)
    cv2.setMouseCallback("RGB Image", click_event_rgb)

    print("Select at least 4 corresponding points on both images.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points_nir) < 4 or len(points_rgb) < 4:
        raise ValueError("At least 4 points are required to compute the homography matrix.")

    points_nir = np.array(points_nir, dtype=np.float32)
    points_rgb = np.array(points_rgb, dtype=np.float32)

    H, _ = cv2.findHomography(points_rgb, points_nir, cv2.RANSAC)
    return H

# Compute homography matrix
homography_matrix = select_points_and_compute_homography()
print("Homography matrix:\n", homography_matrix)

# Save homography matrix to file
np.savetxt("homography_matrix.txt", homography_matrix)

# homography_matrix = [1.228061464845214390e+00, 1.378023440776908470e-01, -3.112631673481545036e+01,
# 1.176856204065238228e-02, 1.221398345195136237e+00, -9.387910005087371701e+01,
# 6.320283444446293136e-05, 1.128010490367590890e-04, 1.000000000000000000e+00,
# ]

# Wrap RGB image using the homography matrix
wrapped_rgb = cv2.warpPerspective(rgb_image, homography_matrix, (nir_image.shape[1], nir_image.shape[0]))
cv2.imwrite("wrapped_rgb.jpg", wrapped_rgb)

# Compute NDVI image
red_channel_rgb = wrapped_rgb[:, :, 2].astype(np.float32)
cv2.imwrite("red_rgb.jpg",red_channel_rgb)
red_channel_nir = nir_image[:, :, 2].astype(np.float32)
blue_channel_nir = nir_image[:, :, 0].astype(np.float32)
cv2.imwrite("red_nir.jpg", red_channel_nir)
cv2.imwrite("blue_nir.jpg", blue_channel_nir)



nir_data = red_channel_nir
red_data = red_channel_rgb

# Clip data to 0 -> 1
# nir_data = np.clip(nir_data / 255.0, 0, 1)
# red_data = np.clip(red_data / 255.0, 0, 1)
nir_data = nir_data / 255.0
red_data = red_data / 255.0

# Compute NDVI
ndvi = (nir_data - red_data) / (nir_data + red_data + 1e-6)  # Add small value to avoid division by zero

# Map NDVI to colormap
ndvi_colormap = cv2.applyColorMap(((ndvi + 1) / 2 * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Save and display results
cv2.imwrite("ndvi_image.jpg", ndvi_colormap)
# cv2.imshow("NIR Image", nir_image)
# cv2.imshow("Wrapped RGB", wrapped_rgb)
# cv2.imshow("NDVI Image", ndvi_colormap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
