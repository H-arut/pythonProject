import cv2
import mediapipe as mp
import time
import imutils
import scipy.spatial

distanceModule = scipy.spatial.distance


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
        #                                          self.minDetectionCon, self.minTrackCon)
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        # face_parts = [81,178,13,14,311,402]
        face_parts = [x for x in range(468)]

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # if draw:
                #     self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                #                            self.drawSpec, self.drawSpec)

                face = []

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    face.append([x, y])

                for id,lm in enumerate(faceLms.landmark):

                    if id in face_parts:
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        cv2.circle(img, (x, y), 2, (0,0, 255), -1)
                faces.append(face)

        return img, faces


def mesh_detector_main_face_part(face_part):
    cap = cv2.VideoCapture(0)
    pTime = 0
    frames = 0
    detector = FaceMeshDetector(face_part,maxFaces=2)
    while True:
        frames +=1
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!= 0:
                # print(faces[0])
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
            #             3, (0, 255, 0), 3)
            resized = imutils.resize(img, width=1200)

            cv2.imshow("Image", resized)
            # cv2.waitKey(0)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break




def mesh_detector_main():
    # cap = cv2.VideoCapture("./recordings/video.mp4")
    cap = cv2.VideoCapture(0)
    pTime = 0
    frames = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        frames +=1
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
                # print(faces[0])
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 255, 0), 3)
            cv2.imshow("Image", img)
            # cv2.waitKey(1)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

if __name__ == "__main__":
    mesh_detector_main()
