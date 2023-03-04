import cv2
import time

# Load the cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("path/to/haarcascade_eye.xml")

# Start the timer
start_time = time.time()

# capture video from camera
cap = cv2.VideoCapture(0)

while True:
    # read the frame
    ret, frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # check if eyes are closed
    closed_eyes = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            closed_eyes += 1

    # check if closed_eyes for more than 5 sec
    if closed_eyes > 0:
        current_time = time.time()
        if current_time - start_time > 5:
            print("Eyes have been closed for more than 5 seconds")
            break
    else:
        start_time = time.time()

    # display the frame
    cv2.imshow("Frame", frame)

    # exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()
