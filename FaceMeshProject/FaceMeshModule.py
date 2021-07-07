import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, mode=False, max_num_faces=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.max_num_faces = max_num_faces
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.mode, self.max_num_faces, self.detectionCon, self.trackingCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)

    def findMeshFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLandmark in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLandmark,
                        self.mpFaceMesh.FACE_CONNECTIONS,
                        self.drawSpec,
                        self.drawSpec,
                    )
                face = []
                for id, lm in enumerate(faceLandmark.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    # cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv.VideoCapture("videos/5.mp4")
    pTime = 0
    detector = FaceMeshDetector(max_num_faces=3)

    while True:
        success, img = cap.read()
        img, faces = detector.findMeshFaces(img)
        if len(faces) != 0:
            print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow("image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
