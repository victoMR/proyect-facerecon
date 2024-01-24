import face_recognition
import cv2
import numpy as np
import time

# Open the camera; adjust the number according to your configuration
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error")
    exit()

# File to store performance data
performance_file = open('performance_data_Recon.txt', 'w')

images = []  # List to store the images
print(images)
names = ["2022143009", "2022143069"]

for i in range(5):
    img_path = f"/home/pi/Documents/face_recon/img/img{i+1}.jpeg"
    print(img_path)  # Add this line to check the image path
    img = face_recognition.load_image_file(img_path)

    # Asignar nombres según el rango de imágenes
    if i < 2:
        name = "2022143009"
    else:
        name = "2022143069"

    images.append((img, name))


# List to store the encodings and names of the images
encodings_and_names = [(face_recognition.face_encodings(img)[0], name) for img, name in images]

print("Images loaded")

frame_count = 0  # Counter to control face recognition frequency

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # If the frame is not captured correctly or end of video stream
    if not ret:
        print("Error capturing frame or end of video stream")
        break

    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Resize the frame to a smaller size
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

    # Calculate and store the current time in milliseconds
    timestamp = int(time.time() * 100)

    # Write the timestamp to the file
    performance_file.write(f"{timestamp},{time.process_time()}\n")

    # Get the faces and their encodings from the current frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # List to store the names of the people detected in the current frame
    face_names = []

    # Perform face recognition every frame
    for face_encoding in face_encodings:
        # Compare the encoding of the current face with the encodings of the images
        matches = face_recognition.compare_faces([enc for enc, _ in encodings_and_names], face_encoding)

        # If there are no matches
        if not any(matches):
            face_names.append("Desconocido")
            continue

        # If there are matches
        # Get the name of the person using enumerate to get the index directly
        index, _ = next((i, name) for i, (_, name) in enumerate(encodings_and_names) if matches[i])

        # Add the name to the list
        face_names.append(encodings_and_names[index][1])
        
    print("Nombres detectados:", face_names)

    # For each detected face in the current frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)

        # Draw a label with the name of the person
        cv2.rectangle(frame, (left * 4, (bottom * 4) - 35), (right * 4, bottom * 4), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left * 4 + 6, (bottom * 4) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Camera', frame)
    cv2.waitKey(1)  # Add a small delay to allow the window to refresh

    # Press 'q' to exit the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Increment the frame count
    frame_count += 1

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Close the performance file
performance_file.close()
