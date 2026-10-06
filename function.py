import cv2
import numpy as np
import os
import mediapipe as mp

# Mediapipe Holistic tracks pose + both hands + face in one pass.
# Using both hands and pose (instead of just one hand) gives the model
# much more signal, which matters a lot once you move from single
# letters to full phrases/gestures that involve arm movement.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def mediapipe_detection(image, model):
    """Run a single frame through the mediapipe holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draw pose + both hands on the frame so you can see what the model sees."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style())


def extract_keypoints(results):
    """
    Flatten pose + left hand + right hand landmarks into one feature vector.
    Missing landmarks (e.g. a hand goes off-screen) are filled with zeros so
    the vector length is always the same: 33*4 + 21*3 + 21*3 = 258 values.
    """
    pose = (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    lh = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3))
    rh = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, lh, rh])


# ---- CONFIG: edit this list to add/remove phrases ----
DATA_PATH = os.path.join('MP_Data')
# actions = np.array(['hello', 'how_are_you', 'see_you_later', 'i_need_help', 'thank_you', 'yes', 'no'])
actions = np.array(['hello', 'see_you_later'])

no_sequences = 40       # how many recordings you make per phrase
sequence_length = 30    # how many frames per recording (~1 second at 30fps)
