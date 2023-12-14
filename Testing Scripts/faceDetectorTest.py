from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()
while True:

        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        if bboxs:
            center = bboxs[0]["center"]

        # Display the image in a window named 'Image'
        cv2.imshow("Image", img)
        # Wait for 1 millisecond, and keep the window open
        cv2.waitKey(1)

