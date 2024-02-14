import face_recognition
import cv2
import time

# Open the camera; adjust the number according to your configuration
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error")
    exit()

# List to store the images and names for facial recognition
images = []
names = ["2022143009_Abril", "2022143069_Vic", "2022143063_Mau", "2022143015_Palo"]

for i in range(20):
    img_path = f"/home/pi/Documents/face_recon/img/img{i+1}.jpeg"
    img = face_recognition.load_image_file(img_path)

    # Assign names based on the range of images
    if i < 6:
        name = names[0]
    elif i < 11:
        name = names[1]
    elif i < 16:
        name = names[2]
    else:
        name = names[3]

    # Try to get face encodings; print a message if no face is found
    try:
        encoding = face_recognition.face_encodings(img)[0]
    except IndexError:
        print(f"No face detected in image {img_path}")
        continue

    images.append((img, name))

# List to store the encodings and names of the images
encodings_and_names = [(face_recognition.face_encodings(img)[0], name) for img, name in images]

print("Imagenes cargadas")

frame_count = 1  # Counter to control face recognition frequency
recognition_frequency = 10  # Adjust this value based on your needs

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

    # Perform face recognition every 'recognition_frequency' frames
    if frame_count % recognition_frequency == 0:
        # Get the faces and their encodings from the current frame
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        # List to store the names of the people detected in the current frame
        face_names = []

        # Perform face recognition for each face in the frame
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
