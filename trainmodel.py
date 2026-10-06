"""
Trains an LSTM model on the keypoint sequences you recorded with collectdata.py.
Run this only after collectdata.py has finished for every phrase.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from function import DATA_PATH, actions, no_sequences, sequence_length

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(npy_path):
                window.append(np.load(npy_path, allow_pickle=True))
            else:
                window.append(np.zeros(258))  # 33*4 pose + 21*3 + 21*3 hands
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

if len(X) == 0:
    raise SystemExit("No training data found in MP_Data/. Run collectdata.py first.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])),
    Dropout(0.3),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax'),
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(
    X_train, y_train,
    epochs=300,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, early_stop],
)

model.summary()

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nValidation accuracy: {acc * 100:.1f}%  |  Validation loss: {loss:.3f}")
if acc < 0.8:
    print("Accuracy is on the lower side - consider recording more samples per phrase, "
          "with more variation in position/angle, then re-running this script.")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
print("Saved model.json and model.h5")
