import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

################################################################
brushThickness = 15
eraserThickness = 100
color = (0, 0, 255)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
################################################################

folderPath = "images"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    print(imPath)
    image = cv.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
header = overlayList[0]

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv.flip(img, 1)  # flip image

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]  # index finger
        x2, y2 = lmList[12][1:]  # middle finger

        # 3. Check which fingers are up (Draw with index finger and select the mode with two fingers)
        fingers = detector.fingerUp()
        # print(fingers)

        # 4. if selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection Mode")
            xp, yp = 0, 0

            # check for the click
            if y1 < 125:
                if 60 < x1 < 260:
                    header = overlayList[0]
                    color = (0, 0, 255)
                elif 380 < x1 < 580:
                    header = overlayList[2]
                    color = (255, 0, 0)
                elif 700 < x1 < 900:
                    header = overlayList[3]
                    color = (0, 255, 0)
                elif 1020 < x1 < 1220:
                    header = overlayList[1]
                    color = (0, 0, 0)
            cv.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), color, cv.FILLED)

        # 5. if drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 15, color, cv.FILLED)
            print("Drawing Mode")
            # for the first frame
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if color == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), color, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), color, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), color, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), color, brushThickness)

            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    # Set the header image
    img[0:125, 0:1280] = header
    # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv.imshow("Image", img)
    # cv.imshow("Cavas", imgCanvas)

    cv.waitKey(1)
