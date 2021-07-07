import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("videos/3.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location.relative_bounding_box)
            boundingBox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bBox = (
                int(boundingBox.xmin * w),
                int(boundingBox.ymin * h),
                int(boundingBox.width * w),
                int(boundingBox.height * h),
            )
            cv.rectangle(img, bBox, (255, 0, 255), 2)
            cv.putText(
                img,
                f"{int(detection.score[0]*100)}%",
                (bBox[0], bBox[1] - 20),
                cv.FONT_HERSHEY_PLAIN,
                3,
                (200, 0, 255),
                2,
            )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    ptime = cTime
    cv.putText(img, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
    cv.imshow("Image", img)

    cv.waitKey(1)
