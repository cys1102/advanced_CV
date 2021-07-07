import cv2 as cv
import numpy as np
import time
import PoseModule as pm

# cap = cv.VideoCapture("videos/curls.mp4")
cap = cv.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0
while True:
    success, img = cap.read()
    img = cv.resize(img, (640, 480))
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # # Right arm
        # detector.findAngle(img, 12, 14, 16)

        # Left arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (300, 100))
        # print(per)

        color = (255, 0, 255)
        # check for the dumbbell curls
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:  # the direction going up
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw a bar
        cv.rectangle(img, (570, 100), (590, 300), color, 3)
        cv.rectangle(img, (570, int(bar)), (590, 300), color, cv.FILLED)
        cv.putText(img, f"{int(per)}%", (550, 75), cv.FONT_HERSHEY_PLAIN, 2, color, 2)

        # Draw curl count
        cv.rectangle(img, (0, 330), (150, 480), (0, 255, 0), cv.FILLED)
        cv.putText(img, f"{int(count)}", (25, 450), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv.imshow("Image", img)
    cv.waitKey(1)
