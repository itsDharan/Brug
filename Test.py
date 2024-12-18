import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the trained model from the pickle file
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Open the camera (use 0 or 2 depending on your setup)
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Updated labels_dict for 11 alphabet classes
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L'
}

# Map each letter to its corresponding sound file
sound_files = {
    'A': 'Sound/A.wav',
    'B': 'Sound/B.wav',
    'C': 'Sound/C.wav',
    'D': 'Sound/D.wav',
    'E': 'Sound/E.wav',
    'F': 'Sound/F.wav',
    'G': 'Sound/G.wav',
    'H': 'Sound/H.wav',
    'I': 'Sound/I.wav',
    'K': 'Sound/K.wav',
    'L': 'Sound/L.wav',
}

last_predicted_character = None  # To track the last predicted character
sound_played_time = 0  # To track when the sound was last played

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame from webcam
    ret, frame = cap.read()

    # If failed to capture frame, continue the loop
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    H, W, _ = frame.shape

    # Convert frame to RGB as Mediapipe expects RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe hands model
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Image to draw
                hand_landmarks,  # Model output
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Collect the landmark data (x, y positions)
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize the coordinates by subtracting the min value
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Get bounding box for drawing rectangle around hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])
        # Get the predicted label (character) from the model prediction
        predicted_character = labels_dict[int(prediction[0])]

        # Play unique sound for each letter after 3 seconds
        current_time = time.time()
        if predicted_character != last_predicted_character or current_time - sound_played_time >= 3:
            last_predicted_character = predicted_character
            sound_played_time = current_time
            sound_file = sound_files.get(predicted_character)
            if sound_file:
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()

        # Draw a rectangle around the hand and the predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the frame with hand landmarks and prediction
    cv2.imshow('frame', frame)

    # Exit the loop if the user presses the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()




cv2.putText(frame, predicted_character, (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
