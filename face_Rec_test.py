import face_recognition
import cv2
import numpy as np
import time

class Node:
    def __init__(self, name):
        self.name = name
        self.images = []
        self.left = None
        self.right = None

def create_image_tree():
    # Create nodes for each person
    abril_node = Node("2022143009_Abril")
    vic_node = Node("2022143069_Vic")
    mau_node = Node("2022143063_Mau")
    palo_node = Node("2022143015_Palo")

    # Organize images into sub-trees
    abril_node.images = load_images("2022143009_Abril", 1, 5)
    vic_node.images = load_images("2022143069_Vic", 6, 10)
    mau_node.images = load_images("2022143063_Mau", 11, 15)
    palo_node.images = load_images("2022143015_Palo", 16, 20)

    # Build the image tree
    root = Node("root")
    root.left = abril_node
    root.right = Node("Unknown")  # Node for images not assigned to any person

    abril_node.left = vic_node
    abril_node.right = Node("Unknown")

    vic_node.left = mau_node
    vic_node.right = Node("Unknown")

    mau_node.left = palo_node
    mau_node.right = Node("Unknown")

    return root

def load_images(name, start, end):
    images = []
    for i in range(start, end + 1):
        img_path = f"/home/pi/Documents/face_recon/img/img{i}.jpeg"
        img = face_recognition.load_image_file(img_path)
        images.append((img, name))
    return images

# Create the image tree
image_tree = create_image_tree()

# List all images for comparison
all_images = [image for node in [image_tree] for image in node.images]

# List the encodings and names of the images
encodings_and_names = [(face_recognition.face_encodings(img)[0], name) for img, name in all_images]

print("Images loaded")

frame_count = 0  # Counter to control face recognition frequency

cap = cv2.VideoCapture(0)  # Open the default camera

# Create a file to store performance data
performance_file = open("performance.txt", "w")

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
    timestamp = int(time.time() * 1000)

    # Write the timestamp to the file
    performance_file.write(f"{timestamp},{time.process_time()}\n")

    # Get the faces and their encodings from the current frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # List to store the names of the people detected in the current frame
    face_names = []

    # Perform face recognition for each face in the current frame
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
