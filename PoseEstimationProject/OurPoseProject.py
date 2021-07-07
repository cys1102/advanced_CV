import cv2 as cv
import time
import PoseModule as pm


cap = cv.VideoCapture("videos/4.mp4")
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList)
        cv.circle(img, (lmList[14][1], lmList[14][2]), 10, (255, 0, 0), cv.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow("Image", img)
    cv.waitKey(1)
