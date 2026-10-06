"""
Real-time phrase recognition from webcam.

Builds keypoint sequences from your webcam feed, predicts a phrase once
30 frames have been collected, and:
  - shows the recognized phrase + confidence on screen
  - speaks it out loud (text-to-speech) the first time it's recognized
  - lets you build up a running "sentence" of recognized phrases

Controls:
  q  - quit
  c  - clear the on-screen sentence
"""

import cv2
import numpy as np
import threading
from keras.models import model_from_json

from function import (
    mediapipe_detection, draw_styled_landmarks, extract_keypoints,
    mp_holistic, actions,
)

# ---------- Text-to-speech setup ----------
# pyttsx3 runs offline (no internet needed). We speak on a background
# thread so it never freezes the video feed.
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 165)
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Text-to-speech unavailable ({e}). Continuing without audio.")
    TTS_AVAILABLE = False


def speak(text):
    if not TTS_AVAILABLE:
        return
    def _run():
        tts_engine.say(text.replace('_', ' '))
        tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


# ---------- Load trained model ----------
with open("model.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("model.h5")

THRESHOLD = 0.8          # minimum confidence to accept a prediction
STABLE_FRAMES = 10       # how many recent predictions must agree before accepting
SEQUENCE_LEN = 30

sequence = []
sentence = []
predictions = []
last_spoken = None

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LEN:]

        if len(sequence) == SEQUENCE_LEN:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = int(np.argmax(res))
            predictions.append(predicted_idx)
            predictions = predictions[-STABLE_FRAMES:]

            confidence = res[predicted_idx]
            stable = len(set(predictions)) == 1  # last N predictions all agree

            if stable and confidence > THRESHOLD:
                phrase = actions[predicted_idx]
                if len(sentence) == 0 or sentence[-1] != phrase:
                    sentence.append(phrase)
                    sentence = sentence[-5:]  # keep last 5 phrases on screen
                    if phrase != last_spoken:
                        speak(phrase)
                        last_spoken = phrase

            # Live confidence bar for the top prediction
            cv2.rectangle(image, (0, 60), (int(confidence * 200), 90), (245, 117, 16), -1)
            cv2.putText(image, f'{actions[predicted_idx]} {confidence*100:.0f}%', (5, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Display running sentence
        display_text = ' '.join(w.replace('_', ' ') for w in sentence)
        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (245, 117, 16), -1)
        cv2.putText(image, display_text, (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Recognition', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            last_spoken = None

cap.release()
cv2.destroyAllWindows()
