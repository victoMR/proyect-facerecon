import face_recognition
import cv2
import dlib
import time
from tqdm import tqdm  # Importa la clase tqdm
from pyomo.environ import *

# Cargar el modelo de puntos faciales
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Abrir la cámara; ajusta el número según tu configuración
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Lista para almacenar las imágenes y nombres para el reconocimiento facial
images = []
names = ["Desconocido", "2022143009_Abril", "2022143069_Vic", "2022143063_Mau", "2022143015_Palo"]

# Utiliza tqdm para mostrar una barra de progreso en la consola
for i in tqdm(range(20), desc="Cargando datos"):
    img_path = f"/home/pi/Documents/face_recon/img/img{i+1}.JPG"
    img = face_recognition.load_image_file(img_path)

    # Asignar nombres basados en el rango de imágenes
    if i < 11:
        name = names[2]
    elif i < 21:
        name = names[3]
    elif i < 31:
        name = names[4]
    elif i < 41:
        name = names[5]
    else:
        name = names[1]

    # Intentar obtener codificaciones faciales; imprimir un mensaje si no se encuentra ninguna cara
    try:
        encoding = face_recognition.face_encodings(img)[0]
    except IndexError:
        print(f"No se detectó ninguna cara en la imagen {img_path}")
        continue

    images.append((img, name))

# Lista para almacenar las codificaciones y nombres de las imágenes
encodings_and_names = [(face_recognition.face_encodings(img)[0], name) for img, name in images]
tqdm.write("Imágenes cargadas")

frame_count = 1  # Contador para controlar la frecuencia de reconocimiento facial
recognition_frequency = 6  # Ajusta este valor según tus necesidades

# Variable para almacenar los nombres de las caras detectadas
current_face_names = []

# Definir face_locations fuera del bucle
face_locations = []

# Crear un modelo Pyomo
modelo = ConcreteModel()

# Definir el conjunto de imágenes
imagenes = range(1, 21)

# Variable binaria para representar si una persona es reconocida incorrectamente (Falso Positivo)
modelo.fp = Var(imagenes, within=Binary)

# Variable para representar la similitud entre imágenes
modelo.similitud = Var(imagenes, imagenes, within=Binary)

# Objetivo: Minimizar el número de falsos positivos y la falta de similitud entre imágenes
modelo.obj = Objective(expr=sum(modelo.fp[i] for i in imagenes) + sum((1 - modelo.similitud[i, j]) for i in imagenes for j in imagenes), sense=minimize)

# Restricción: Asegurar que "Desconocido" no sea reconocido como alguien específico
modelo.con = Constraint(expr=modelo.fp[1] == 0)

# Restricción: Asegurar que las personas conocidas sean reconocidas correctamente
for i in range(2, 21):
    modelo.add_component(f"con_{i}", Constraint(expr=modelo.fp[i] == 0))

# Restricción: Definir la relación de similitud entre imágenes (puedes ajustar según tus necesidades)
modelo.sim_constraint = ConstraintList()
for i in imagenes:
    for j in imagenes:
        if i != j:
            # Ejemplo: Asumimos que si ambas imágenes tienen falsos positivos, son similares
            modelo.sim_constraint.add(modelo.similitud[i, j] >= modelo.fp[i] + modelo.fp[j] - 1)

# Crear un solucionador
solucionador = SolverFactory('glpk')

while True:
    # Capturar fotograma por fotograma
    ret, frame = cap.read()

    # Si el fotograma no se captura correctamente o el final del flujo de video
    if not ret:
        print("Error capturando el fotograma o el final del flujo de video")
        break

    # Convertir el fotograma de BGR a RGB
    rgb_frame = frame[:, :, ::-1]

    # Cambiar el tamaño del fotograma a un tamaño más pequeño
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.23, fy=0.23)

    # Realizar el reconocimiento facial cada 'recognition_frequency' fotogramas
    if frame_count % recognition_frequency == 0:
        # Obtener las caras y sus codificaciones del fotograma actual
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        # Limpiar la lista de nombres de caras
        current_face_names = []

        # Realizar el reconocimiento facial para cada cara en el fotograma
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Obtener la caja delimitadora de la cara
            top, right, bottom, left = face_location
            face_box = dlib.rectangle(left, top, right, bottom)

            # Predecir los puntos faciales
            landmarks = predictor(small_frame, face_box)

            # Dibujar puntos faciales más finos
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x * 4, y * 4), 1, (0, 255, 0), -1)

            # Comparar la codificación de la cara actual con las codificaciones de las imágenes
            matches = face_recognition.compare_faces([enc for enc, _ in encodings_and_names], face_encoding, tolerance=0.5)

            # Si no hay coincidencias
            if not any(matches):
                current_face_names.append("Desconocido")
                continue

            # Si hay coincidencias
            # Obtener el nombre de la persona usando enumerate para obtener el índice directamente
            index = matches.index(True)
            current_face_names.append(encodings_and_names[index][1])

        try:
            # Resolver el problema de optimización con Pyomo
            solucionador.solve(modelo)

            # Mostrar los resultados de la optimización
            print("Solución óptima:")
            for i in imagenes:
                print(f"Imagen {i}: Falso Positivo = {value(modelo.fp[i])}")
        except Exception as e:
            print(f"Error en la optimización: {e}")

    # Mostrar los nombres de las caras detectadas
    for (top, right, bottom, left), name in zip(face_locations, current_face_names):
        # Dibujar un rectángulo alrededor de la cara
        cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)

        # Dibujar una etiqueta con el nombre de la persona
        cv2.rectangle(frame, (left * 4, (bottom * 4) - 35), (right * 4, bottom * 4), (0, 100, 255), cv2.FILLED)
        cv2.putText(frame, name, (left * 4 + 6, (bottom * 4) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Mostrar el fotograma
    cv2.imshow('Camera', frame)
    cv2.waitKey(1)  # Agregar una pequeña demora para permitir que la ventana se actualice

    # Presiona 'q' para salir del programa
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Incrementar el contador de fotogramas
    frame_count += 1

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
