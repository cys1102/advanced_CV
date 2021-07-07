import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import autopy

################################################################
wCam, hCam = 640, 480
frameR = 100  # frame reduction
smoothening = 7
################################################################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1, detectionCon=0.75)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    success, img = cap.read()
    # 1. Find hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # 2. Get the tip of the index and middel fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingerUp()
        # print(fingers)
        cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only index finger: Moving mode
        if fingers[1] and not fingers[2]:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # 7. Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both index and middel fingers are up: Clicking mode
        if fingers[1] and fingers[2]:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 25:
                cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), 3)
                autopy.mouse.click()
            # 9. Find distance between fingers
    # 10. Click mouse if distance is short

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (20, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # 12. Display
    cv.imshow("Image", img)
    cv.waitKey(1)
