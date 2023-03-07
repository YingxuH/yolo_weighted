import cv2
import rawpy
from pathlib import Path

import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127


def preprocess(gray, clip_limit=40.0, grid_size=(180, 90), sigma=10, adap_area=21):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    gray_clahe = clahe.apply(gray)
    filtered = highpass(gray_clahe, sigma)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, adap_area, 0.0)
    return thresh


def extract_blocks(image, block_size):
    """
    Function to extract blocks from images
    """
    x_origin, y_origin = image.shape[:2]
    x_step, y_step = block_size

    x_splits = np.arange(0, x_origin, x_step)
    y_splits = np.arange(0, y_origin, y_step)
    blocks = []

    for x_id in range(len(x_splits)):
        for y_id in range(len(y_splits)):
            x_start = x_splits[x_id]
            y_start = y_splits[y_id]

            if x_id == len(x_splits) - 1:
                x_end = x_origin
            else:
                x_end = x_splits[x_id + 1]

            if y_id == len(y_splits) - 1:
                y_end = y_origin
            else:
                y_end = y_splits[y_id + 1]
            blocks.append(image[x_start: x_end, y_start: y_end])

    return blocks


def adaptive_morph(thresh):
    mask = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), np.uint8)
    for k_size in tqdm(range(3, 15)):

        _, thresh_twice = cv2.threshold(cv2.blur(thresh,(k_size, k_size)), int(255 * 0.999), 255.0, cv2.THRESH_BINARY)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        thresh_twice = cv2.dilate(thresh_twice, dilate_kernel)

        # thresh_twice_close = cv2.medianBlur(thresh_twice, int(np.ceil(k_size/2) * 2 - 1))
        thresh_twice_close = cv2.medianBlur(thresh_twice, 5)
        thresh_twice = cv2.bitwise_or(thresh_twice, thresh_twice_close)

        threshInv = cv2.bitwise_not(thresh_twice)

        params = cv2.SimpleBlobDetector_Params()

        # opencv tutorial: https://learnopencv.com/blob-detection-using-opencv-python-c/
        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 1000

        # Set Circularity filtering parameters
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # # Set inertia filtering parameters
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        # Set Convexity filtering parameters
        params.filterByConvexity = True
        params.minConvexity = 0.7

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(threshInv)
        blank_ = np.full_like(threshInv, 255)

        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))

        for i in range(len(keypoints)):
            if cv2.bitwise_not(thresh_twice)[int(keypoints[i].pt[1]), int(keypoints[i].pt[0])] == 0:
                _, _, mask, _ = cv2.floodFill(cv2.bitwise_not(thresh_twice), mask, seedPoint=(int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), newVal=255, flags=cv2.FLOODFILL_MASK_ONLY)
    return mask


def detect_blobs(threshInv):
    #     threshInv = cv2.bitwise_not(mask[1:-1, 1:-1] * 255)

    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1000

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.4

    # # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(threshInv)
    # blank_ = np.full_like(threshInv, 255)
    return keypoints

