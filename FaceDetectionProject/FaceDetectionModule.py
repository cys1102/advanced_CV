import cv2 as cv
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bBoxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingBox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bBox = (
                    int(boundingBox.xmin * w),
                    int(boundingBox.ymin * h),
                    int(boundingBox.width * w),
                    int(boundingBox.height * h),
                )
                bBoxes.append([id, bBox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bBox)
                    cv.putText(
                        img,
                        f"{int(detection.score[0]*100)}%",
                        (bBox[0], bBox[1] - 20),
                        cv.FONT_HERSHEY_PLAIN,
                        3,
                        (200, 0, 255),
                        2,
                    )
        return img, bBoxes

    def fancyDraw(self, img, bBox, l=30, t=10, rt=1):
        x, y, w, h = bBox
        x1, y1 = x + w, y + h

        cv.rectangle(img, bBox, (255, 0, 255), rt)
        # Top left x, y
        cv.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top right x1, y
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom left x, y1
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom right x1, y1
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv.VideoCapture("videos/1.mp4")
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bBoxes = detector.findFaces(img)
        print(bBoxes)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        ptime = cTime
        cv.putText(img, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
        cv.imshow("Image", img)

        cv.waitKey(1)


if __name__ == "__main__":
    main()
