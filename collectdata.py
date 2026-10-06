"""
Records your own training data for each phrase in `actions` (see function.py).

For every phrase, it records `no_sequences` separate clips, each
`sequence_length` frames long. Between clips you get a short pause so you
can reset your hands/posture before the next one starts.

TIPS FOR GOOD ACCURACY:
- Vary your position slightly between recordings (don't sit frozen in the
  exact same spot every time) so the model learns the *gesture*, not your
  exact pixel position.
- Keep consistent lighting and try to stay inside the camera frame.
- Do the sign at a natural, comfortable speed - not rushed.
- 40 recordings per phrase is a reasonable starting point. More is better.
"""

import cv2
import numpy as np
import os
from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from function import mp_holistic, DATA_PATH, actions, no_sequences, sequence_length

# Create the folder structure: MP_Data/<phrase>/<sequence_number>/
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Could not read from webcam. Check your camera index/permissions.")
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    # Pause screen before each new recording so you can get ready
                    cv2.putText(image, f'Get ready: "{action}"', (15, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Recording {sequence + 1}/{no_sequences}', (15, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Collecting Data', image)
                    cv2.waitKey(1500)  # ~1.5 second pause
                else:
                    cv2.putText(image, f'Recording: "{action}" ({sequence + 1}/{no_sequences})', (15, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Collecting Data', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
print("Data collection complete. Run trainmodel.py next.")
