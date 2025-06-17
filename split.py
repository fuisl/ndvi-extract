import cv2
import numpy as np

nir_image = cv2.imread("RGB/070525_1522_RGB.jpg")

nir_channel = cv2.split(nir_image)

red_nir = nir_channel[2].astype("float32")
green_nir = nir_channel[1].astype("float32")
blue_nir = nir_channel[0].astype("float32")

cv2.imwrite("red_nir.jpg", red_nir)
cv2.imwrite("green_nir.jpg", red_nir)
cv2.imwrite("blue_nir.jpg", red_nir)
