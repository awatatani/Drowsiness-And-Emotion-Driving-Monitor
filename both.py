import cv2
import torch
from torchvision import transforms
from grad_cam import BackPropagation
from PIL import Image
from model import Model
import model
import matplotlib.pyplot as plt
from IPython import display
import os
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Initialize the face detection cascade for emotion detection
faceCascade_emotion = cv2.CascadeClassifier('haarcascade_frontalface.xml')

# Initialize the face detection cascade for drowsiness detection
faceCascade_drowsiness = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the eye detection cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Emotion detection variables
emotion_shape = (48, 48)
emotion_classes = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprised',
    'Neutral'
]

# Drowsiness detection variables
drowsiness_shape = (24, 24)
drowsiness_classes = ['Close', 'Open']
eyess = []
cface = 0
sensi = 20

# Preprocess function for real-time frames in emotion detection
def preprocess_emotion(frame):
    global faceCascade_emotion
    global emotion_shape
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade_emotion.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    flag = 0
    if len(faces) == 0:
        print('No face found for emotion detection')
        face = cv2.resize(frame, emotion_shape)
    else:
        (x, y, w, h) = faces[0]
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, emotion_shape)
        flag = 1

    img = Image.fromarray(face).convert('L')
    inputs = transform_test(img)
    return inputs, face, flag

# Preprocess function for real-time frames in drowsiness detection
def preprocess_drowsiness(frame):
    global eyess
    global cface
    global sensi

    transform_test = transforms.Compose([transforms.ToTensor()])
    image = frame.copy()

    faces = faceCascade_drowsiness.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        cface = 0
    else:
        cface = 1
        eyess = []  # Clear the eyess list for each frame
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face, 1.3, sensi)

            for (ex, ey, ew, eh) in eyes:
                eye = face[ey:ey + eh, ex:ex + ew]
                eye = cv2.resize(eye, drowsiness_shape)
                eyess.append([transform_test(Image.fromarray(eye).convert('L')).unsqueeze(1), eye])

def eye_status(image, name, net):
    img = torch.unsqueeze(image[name], 0)
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:, 1]
    class_index = actual_status.data.item()
    confidence = prob.data.item()   # Multiply by 100 to get percentage
    return drowsiness_classes[class_index], confidence

# Perform real-time emotion and drowsiness detection
def detect_emotion_and_drowsiness_realtime(emotion_model_name, drowsiness_model_name):
    # Load emotion detection model
    emotion_checkpoint = torch.load(emotion_model_name, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    emotion_net = Model(num_classes=len(emotion_classes))
    emotion_net.load_state_dict(emotion_checkpoint['net'])
    emotion_net.eval()
    emotion_net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load drowsiness detection model
    drowsiness_net = model.Model(num_classes=len(drowsiness_classes))
    drowsiness_checkpoint = torch.load(drowsiness_model_name, map_location=torch.device('cpu'))
    drowsiness_net.load_state_dict(drowsiness_checkpoint['net'])
    drowsiness_net.eval()

    cap = cv2.VideoCapture(0)  # Open camera
    emotion_threshold = 0.8  # Confidence threshold for emotion detection
    drowsiness_threshold = 0.5  # Confidence threshold for drowsiness detection
    emotion_label = ""  # Current emotion label
    emotion_confidence = 0.0  # Current emotion confidence

    # Initialize drowsiness_display
    drowsiness_display = ""
    pred_drowsiness_display = ""
    confidence = 0.0



    while True:
        ret, frame = cap.read()  # Read frame from camera
        if not ret:
            break

        # Emotion detection
        target_emotion, raw_image_emotion, flag_emotion = preprocess_emotion(frame)  # Preprocess frame for emotion detection
        img_emotion = torch.stack([target_emotion]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        with torch.no_grad():
            outputs_emotion = emotion_net(img_emotion)  # Perform emotion inference
            _, predicted_emotion = torch.max(outputs_emotion, 1)
            emotion = emotion_classes[predicted_emotion.item()]
            confidence_emotion = torch.softmax(outputs_emotion, dim=1)[0][predicted_emotion.item()].item() * 100

        if flag_emotion:
            if confidence_emotion > emotion_threshold:
                emotion_label = emotion
                emotion_confidence = confidence_emotion

            emotion_display = f"Emotion: {emotion_label} ({emotion_confidence:.2f}%)"
        else:
            emotion_display = "No face found for emotion detection"

        # Drowsiness detection
        preprocess_drowsiness(frame)  # Preprocess frame for drowsiness detection

        if cface == 0:
            drowsiness_label = "No face detected"
            drowsiness_confidence = 0.0
        else:
            if len(eyess) > 0:
                pred_drowsiness, confidence_drowsiness = eye_status(eyess[0][0], 0, drowsiness_net)
                confidence = confidence_drowsiness
                pred_drowsiness_display = f"Drowsiness: {pred_drowsiness} ({drowsiness_confidence:.2f}%)"
                drowsiness_label = pred_drowsiness
                drowsiness_confidence = confidence * 100
            else:
                drowsiness_label = "No eyes detected"
                drowsiness_confidence = 0.0


        # Display the results on the frame
        cv2.putText(frame, emotion_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, pred_drowsiness_display, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion and Drowsiness Detection', frame)

        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time emotion and drowsiness detection
emotion_model = 'emotions.t7'
drowsiness_model = 'model_4_64.t7'
detect_emotion_and_drowsiness_realtime(emotion_model, drowsiness_model)

