import cv2
import numpy as np
import face_recognition
import tflite_runtime.interpreter as tflite

# Cargar el modelo TensorFlow Lite (reemplaza 'modelo.tflite' con tu modelo)
interpreter = tflite.Interpreter(model_path="modelo.tflite")
interpreter.allocate_tensors()

# Obtener detalles del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lista para almacenar las imágenes y nombres para el reconocimiento facial
images = []
names = ["Desconocido", "2022143009_Abril", "2022143069_Vic", "2022143063_Mau", "2022143015_Palo"]

# Cargar imágenes y nombres
for i in range(40):
    img_path = f"/home/pi/Documents/face_recon/img/img{i+1}.JPG"
    img = cv2.imread(img_path)

    # Asignar nombres basados en el rango de imágenes
    if i < 11:
        name = names[1]
    elif i < 21:
        name = names[2]
    elif i < 31:
        name = names[3]
    elif i < 41:
        name = names[4]
    else:
        name = names[0]

    images.append((img, name))

# Inicializar la cámara en la Raspberry Pi; ajusta el número según tu configuración
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    # Capturar fotograma por fotograma
    ret, frame = cap.read()

    # Mostrar "Cargando..." hasta que se detecte la cámara
    cv2.putText(frame, "Cargando...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('Camera', frame)
    cv2.waitKey(1)

    # Convertir la imagen a escala de grises si es necesario
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionar la imagen al tamaño requerido por el modelo
    resized_frame = cv2.resize(gray_frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))

    # Expandir las dimensiones para cumplir con el formato de entrada del modelo
    input_data = np.expand_dims(np.expand_dims(resized_frame, axis=0), axis=3)

    # Alimentar la entrada del modelo y obtener la salida
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Modificar la tolerancia dinámicamente
    tolerance = 0.6  # Valor de tolerancia inicial, puedes ajustarlo según sea necesario

    # Realizar el reconocimiento facial para cada cara en el fotograma
    for embedding in output_data:
        # Comparar la incrustación facial actual con las incrustaciones de las imágenes
        distances = np.linalg.norm(images[:, 0] - embedding, axis=1)
        min_distance_index = np.argmin(distances)

        if distances[min_distance_index] < tolerance:
            name = images[min_distance_index, 1]
        else:
            name = "Desconocido"

        # Obtener las ubicaciones de las caras
        face_locations = face_recognition.face_locations(frame)

        # Mostrar las ubicaciones de las caras
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Mostrar el nombre en el fotograma
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Mostrar el fotograma
    cv2.imshow('Camera', frame)

    # Presiona 'q' para salir del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
