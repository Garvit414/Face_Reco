import cv2
import os
import numpy as np
import paddlehub as hub

# Create a face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Generate a face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


label = {'Garvit':0,'Vaibhav':1,'Abhinav':2,'Anurag':3}
# label




# Use the default webcam (index 0)
camera_url = 0

face_detector = hub.Module(name="pyramidbox_lite_mobile")

def draw_bounding_boxes(image, faces):
    # Draw bounding boxes on the image
    for face in faces:
        left = int(face['left'])
        right = int(face['right'])
        top = int(face['top'])
        bottom = int(face['bottom'])
        confidence = face['confidence']

        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        label = f": {confidence:.2f}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    return image

def capture_faces(User):
    # Create a directory to store the captured faces
    if not os.path.exists('Faces'):
        os.makedirs('Faces')

    # Open the camera
    cap = cv2.VideoCapture(camera_url)

    # Set the image counter as 0
    count = 0

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        if ret:
            # Perform face detection on the frame
            result = face_detector.face_detection(images=[frame])
            box_list = result[0]['data']
            
            # Draw bounding boxes on the frame
            img_with_boxes = draw_bounding_boxes(frame, box_list)
            
            # Display the frame with bounding boxes
            cv2.imshow("Capture Faces", img_with_boxes)
            
            # Store the captured face images in the Faces folder
            for face in box_list:
                left = int(face['left'])
                right = int(face['right'])
                top = int(face['top'])
                bottom = int(face['bottom'])
                cv2.imwrite(f'Faces/{User}_{count}.jpg', frame[top:bottom, left:right])
                count += 1
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Break the loop after capturing a certain number of images
        if count >= 200:
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# capture_faces('Anurag')





def train_model(label):
    # Create lists to store the face samples and their corresponding labels
    faces = []
    labels = []

    # Load the images from the 'Faces' folder
    for file_name in os.listdir('Faces'):
        if file_name.endswith('.jpg'):
            # Extract the label (person's name) from the file name
            name = file_name.split('_')[0]

            # Read the image
            image = cv2.imread(os.path.join('Faces', file_name))

            # Perform face detection on the image
            result = face_detector.face_detection(images=[image])
            box_list = result[0]['data']

            # Check if a face is detected
            if len(box_list) > 0:
                # Get the coordinates of the first detected face
                left = int(box_list[0]['left'])
                top = int(box_list[0]['top'])
                right = int(box_list[0]['right'])
                bottom = int(box_list[0]['bottom'])

                # Crop the detected face region and convert to grayscale
                face_crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)

                # Append the face sample and label to the lists
                faces.append(face_crop)
                labels.append(label[name])
    # Train the face recognition model using the faces and labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # Save the trained model to a file
    recognizer.save('trained_model.xml')
    return recognizer            

# Train the model

Recognizer =train_model(label)
Recognizer



# Function to recognize faces
def recognize_faces(recognizer, label):
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Reverse keys and values in the dictionary
    label_name = {value: key for key, value in label.items()}
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if ret:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            result = face_detector.face_detection(images=[frame])
            box_list = result[0]['data']

            # Recognize and label the faces
            for face in box_list:
                # Get the coordinates of the detected face
                left = int(face['left'])
                top = int(face['top'])
                right = int(face['right'])
                bottom = int(face['bottom'])

                # Recognize the face using the trained model
                label_id, confidence = recognizer.predict(gray[top:bottom, left:right])

                if confidence > 50:
                    # Display the recognized label and confidence level
                    cv2.putText(frame, label_name[label_id], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    print('Unrecognized')

            # Display the frame with face recognition
            cv2.imshow('Recognize Faces', frame)

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Recognize the live faces
recognize_faces(Recognizer, label)
