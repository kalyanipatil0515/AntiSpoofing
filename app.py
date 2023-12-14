import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
import time  # Import the time module

def main():
    confidence = 0.8
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("../models/l_version_1_300.pt")

    classNames = ["fake", "real"]

    prev_frame_time = 0

    st.title("Real vs. Fake Object Detection")

    while True:
        new_frame_time = time.time()  # Use the time module for tracking time
        success, img = cap.read()
        results = model(img, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])
                if conf > confidence:
                    if classNames[cls] == 'real':
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=2, colorR=color, colorB=color)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        st.image(img, channels="BGR", caption=f"FPS: {fps}", use_column_width=True)
        st.write(f"Frames Per Second: {fps}")

if __name__ == "__main__":
    main()
