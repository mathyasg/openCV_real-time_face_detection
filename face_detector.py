# face_eye_detection.py
# ========================================================
# Real-time Face & Eye Detection using OpenCV + Haar Cascades
# Author: MathyasG
# Created: February 2026
# Description:
#   - Detects faces and eyes from your webcam in real-time
#   - Improved eye detection (upper-half only + anti-flicker)
#   - Press 'c' to capture faces, 'f' for fullscreen, 'e' to toggle eyes
# ========================================================

import cv2
import datetime

# ====================== CONFIGURATION ======================
# Window settings (user-friendly size)
WINDOW_NAME = 'Face & Eye Detection'
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600

# Path to classifier files (must be in the same folder as this script)
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = 'haarcascade_eye.xml'

# ====================== LOAD CLASSIFIERS ======================
# Load pre-trained Haar Cascade models (machine learning based)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# Critical safety check for face detector
if face_cascade.empty():
    print("ERROR: Could not load face cascade classifier!")
    print(f"Please make sure '{FACE_CASCADE_PATH}' is in the project folder.")
    exit()

# Warning if eye detector is missing
if eye_cascade.empty():
    print(f"WARNING: Eye classifier '{EYE_CASCADE_PATH}' not found. Eye detection will be disabled.")

# ====================== WINDOW SETUP ======================
# Create resizable window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DEFAULT_WIDTH, DEFAULT_HEIGHT)

# ====================== CAMERA SETUP ======================
# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Make sure your camera is not being used by another application.")
    exit()

# ====================== RUNTIME VARIABLES ======================
detect_eyes = True      # Toggle with 'e' key
fullscreen = False      # Toggle with 'f' key

print("🎥 Face & Eye Detection started!")
print("Controls:  q = quit  |  c = capture  |  f = fullscreen  |  e = toggle eyes")

# ====================== MAIN LOOP ======================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert to grayscale (required for Haar detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw face rectangle + label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # ==================== IMPROVED EYE DETECTION ====================
        if detect_eyes and not eye_cascade.empty():
            # Search ONLY in upper half of face → eliminates nose false positives
            eye_roi_h = int(h * 0.55)
            roi_gray = gray[y:y + eye_roi_h, x:x + w]
            roi_color = frame[y:y + eye_roi_h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray,
                                                scaleFactor=1.1,
                                                minNeighbors=10,   # Higher = less flicker
                                                minSize=(25, 25))

            # Limit to max 2 eyes per face
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye Detected", (ex, ey - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ====================== ON-SCREEN INSTRUCTIONS ======================
    cv2.putText(frame, "q:quit | c:capture | f:fullscreen | e:toggle eyes",
                (15, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow(WINDOW_NAME, frame)

    # ====================== KEYBOARD CONTROLS ======================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Exiting application...")
        break
    elif key == ord('c'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"face_detection_{timestamp}.jpg", frame)

        for i, (x, y, w, h) in enumerate(faces):
            face_crop = frame[y:y + h, x:x + w]
            cv2.imwrite(f"detected_face_{timestamp}_{i+1}.jpg", face_crop)

        print(f"✅ Saved! Full image + {len(faces)} face crop(s)")
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    elif key == ord('e'):
        detect_eyes = not detect_eyes
        print(f"Eyes detection: {'ON' if detect_eyes else 'OFF'}")

# ====================== CLEANUP ======================
cap.release()
cv2.destroyAllWindows()