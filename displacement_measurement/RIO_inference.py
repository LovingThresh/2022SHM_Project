from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

pts = np.array([[105, 222], [86, 963], [51, 1534], [1, 1549], [8, 1655], [850, 1723], [904, 1694], [1010, 1697],
                [1073, 1629], [1073, 1508], [980, 1497], [882, 235], [727, 4], [382, 125]], np.int32)
mask_1 = np.zeros((1920, 1080), np.uint8)
cv.fillPoly(mask_1, [pts], 255)

pts_2 = np.array([[147, 1363], [147, 1318], [186, 1318], [196, 1551], [157, 1544]], np.int32)
mask_2 = np.zeros((1920, 1080), np.uint8)
cv.fillPoly(mask_2, [pts_2], 255)

pts_3 = np.array([[4, 1278], [102, 1271], [102, 1620], [0, 1936]], np.int32)
mask_3 = np.zeros((1920, 1080), np.uint8)
cv.fillPoly(mask_3, [pts_3], 255)

mask_ROI = np.uint8((mask_2 != mask_1) & (mask_3 != mask_1))
mask_ROI = cv.rotate(mask_ROI, cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imwrite('a.png', mask_ROI * 255)
n = 0

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frame = cv.bitwise_and(frame, frame, mask=mask_ROI)

    # frame = cv.medianBlur(frame, 7)
    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imwrite('V:/2022SHM-dataset/project2/mask1_images/mask{}.png'.format(n), fgMask)
    n = n + 1
    print(fgMask.sum())
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
