import cv2


def main() -> None:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise FileNotFoundError("Unable to load the face cascade classifier.")

    video_cap = cv2.VideoCapture(0)
    if not video_cap.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    print("Face detection started. Press 'a' or 'q' to stop.")

    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("Failed to read frame from the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "Press 'a' to stop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face Detection App", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("a"), ord("q")):
            break

    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
