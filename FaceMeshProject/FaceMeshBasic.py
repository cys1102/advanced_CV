import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("videos/7.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLandmark in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                img, faceLandmark, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec
            )
            for id, lm in enumerate(faceLandmark.landmark):
                # print(id, lm)
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                print(id, x, y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv.imshow("image", img)
    cv.waitKey(1)
