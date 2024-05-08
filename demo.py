import cv2
import os
import numpy as np
from mtcnn import MTCNN

# Create a face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Generate a face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


label = {'Garvit':0,'Vaibhav':1,'Anurag':2,'Abhinav':3}
# label

mtcnn_detector = MTCNN()

# Function to capture images and store in dataset folder
def capture_images(User):
    # Create a directory to store the captured images
    if not os.path.exists('Faces'):
        os.makedirs('Faces')

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Set the image counter as 0
    count = 0

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to RGB (MTCNN requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces = mtcnn_detector.detect_faces(rgb_frame)

        # Draw rectangles around the faces and store the images
        for result in faces:
            x, y, w, h = result['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Store the captured face images in the Faces folder
            cv2.imwrite(f'Faces/{User}_{count}.jpg', frame[y:y + h, x:x + w])

            count += 1

        # Display the frame with face detection
        cv2.imshow('Capture Faces', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop after capturing a certain number of images
        if count >= 250:
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


# Create the dataset of faces
# capture_images('Abhinav')
# import cv2
# print(cv2.__version__)



import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC

# Function to preprocess images, extract HOG features, and train SVM classifier
def train_model(directory):
    X_train = []
    y_train = []
    label_map = {'Garvit': 0, 'Vaibhav': 1, 'Anurag': 2, 'Abhinav': 3}

    # Initialize Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Loop through the images in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.jpg'):
            # Read the image
            image = cv2.imread(os.path.join(directory, file_name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extract the label from the file name
            label = file_name.split('_')[0]

            # Map the label to its corresponding integer value
            y_label = label_map[label]

            # Detect faces using Haar cascades
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Resize the detected face to a fixed size
                face = cv2.resize(gray[y:y+h, x:x+w], (64, 64))

                # Extract HOG features for the resized face
                features = hog(face, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

                # Append the HOG features and label to X_train and y_train
                X_train.append(features)
                y_train.append(y_label)

    # Train SVM classifier
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm


# Train the model using preprocessed images and HOG features
trained_model = train_model('Faces')

# Function to recognize faces in live video
# Reverse mapping from IDs to names
label_map_reverse = {0: 'Garvit', 1: 'Vaibhav', 2: 'Anurag', 3: 'Abhinav'}

# Function to recognize faces in live video
def recognize_faces(model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haar cascades
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Resize the detected face to a fixed size
            face = cv2.resize(gray[y:y+h, x:x+w], (64, 64))

            # Extract HOG features for the resized face
            features = hog(face, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

            # Predict the label using the trained SVM classifier
            label = model.predict([features])[0]

            # Look up the name corresponding to the predicted label
            name = label_map_reverse[label]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Recognize Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Recognize faces in live video using the trained model
recognize_faces(trained_model)
