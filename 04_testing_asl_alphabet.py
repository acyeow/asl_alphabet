# Imports
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Create mapping for letters a-z (0-25)
char_to_num = {chr(i + 97): i for i in range(26)}
# Add special mappings
char_to_num.update({
    'del': 26,
    'space': 27
})
# Create reverse mapping
num_to_char = {v: k for k, v in char_to_num.items()}

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Set up the Hands function with static image mode and a minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the model
model_dict = pickle.load(open('./xgboost_baseline.pkl', 'rb'))
model = model_dict['model']

# Open the camera
cap = cv2.VideoCapture(0)
while True:
    data_aux, x_, y_ = [], [], []
    ret, frame = cap.read()
    H, W, _ = frame.shape
    # Convert the BGR image to RGB
    frame_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get the landmarks
    results = hands.process(frame_rbg)
    
    if results.multi_hand_landmarks:
        
        # Draw the landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
        # Extract the landmarks for prediction (data_aux) and bounding box (x_ and y_)
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x)
                data_aux.append(y)
                
        # Ensure that the data has all landmarks and ignore if not
        if len(data_aux) != 42:
            continue
        
        # Predict the letter
        prediction = model.predict([np.asarray(data_aux)])
        predicted_char = num_to_char[int(prediction[0])]
        
        # Define the bounding box and the predicted letter
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Draw the bounding box and the predicted letter
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)   
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)    
    
    # Show the frame  
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()