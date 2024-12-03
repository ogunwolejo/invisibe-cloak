import time

import cv2
import numpy as np

kernel_size = np.array([3, 3], np.uint8)
lower_bound = np.array([100, 150, 50], np.uint8)
upper_bound = np.array([140, 255, 255], np.uint8)

window_name = "invisible cloak"

def create_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converting the video frame to hsv from BGR
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    """We perform morphological operations on both the background and the frame"""
    # we perform an erosion and then dilation (opening) and then we now dilate again
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel_size, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel=kernel_size, iterations=1)
    return mask


def apply_mask(mask, bgr, frame: np.ndarray) -> np.ndarray:
    mask_inv = cv2.bitwise_not(mask) # we get the inverse of our blue_mask so that where-ever the blue color is detected it will show black while the background will be white
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(bgr, bgr, mask=mask)
    return cv2.addWeighted(fg, 1, bg, 1, 0)


def start_cam():
    live_cam = cv2.VideoCapture(0)
    if not live_cam.isOpened():
        print("Error: Could not open camera.")
        return

    while live_cam.isOpened():
        ret, frame = live_cam.read()
        backgrounds = []
        if not ret:
            raise ValueError("Cannot not read next camera frame.")
        else:
            # getting frame background
            median_bg: np.uint8 # getting the background median
            backgrounds.append(frame)
            if backgrounds:
                median_bg = np.mean(backgrounds, axis=0).astype(np.uint8)

            # we need to create a mask to detect blue colors
            blue_mask = create_mask(frame)

            # apply the morphological operations
            output = apply_mask(mask=blue_mask, bgr=np.flip(median_bg, axis=1), frame=frame)

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, output)

            if cv2.waitKey(1) & 0xFF == 27:
                break


    live_cam.release()
    cv2.destroyAllWindows()