import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Path to the dataset directory

data = []
labels = []

# Iterate over directories in the DATA_DIR (assuming these represent classes)
for dir_ in os.listdir(DATA_DIR):
    # Only process directories (ignore non-directory files)
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):
        print(f"Processing class {dir_}...")
        
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                print(f"Skipping {img_path}, could not read image.")
                continue  # Skip files that cannot be read

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe Hands to get landmarks
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Normalize landmarks by subtracting the minimum x and y
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Add the processed data and the corresponding label (class)
                data.append(data_aux)
                labels.append(int(dir_))  # Assuming the directory name is the label (e.g., 0, 1, 2, ..., 10)

# Save the collected data and labels as a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection completed and saved to 'data.pickle'.")
