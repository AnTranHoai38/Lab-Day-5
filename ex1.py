import cv2
import numpy as np
def find_line(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss_img = cv2.GaussianBlur(gray_img, (5,5), 0 )
    canny_img = cv2.Canny(gauss_img, 75, 150)
    roi_img = find_roi(canny_img)
    contours, _ = cv2.findContours(roi_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,0,255), 2)
    return img
def find_roi(img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    shape = img.shape
    center = [shape[1] // 2, shape[0] // 2]
    left_point = [0, shape[0]]
    right_point = [shape[1], shape[0]]
    points = np.array([center, left_point, right_point])
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], color=(255, 255, 255))
    roi_img = cv2.bitwise_and(img, img, mask=mask)
    return roi_img

img_path = "Lane.png"
img = cv2.imread(img_path)
result = find_line(img)
cv2.imshow("Find line", result)
cv2.waitKey(0)
