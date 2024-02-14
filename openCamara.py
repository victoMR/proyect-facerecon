import cv2
import dlib

# Inicializar el detector de rostros de dlib
detector_rostros = dlib.get_frontal_face_detector()

# Descargar el modelo preentrenado de puntos faciales
predictor_facial = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Ancho
cap.set(4, 480)  # Altura

# Capturar una imagen cada 0.1 segundos
captura_cada_segundo = 0.1
ultimo_tiempo = 0

while True:
    tiempo_actual = cv2.getTickCount()

    # Capturar un frame de la cámara cada segundo
    if (tiempo_actual - ultimo_tiempo) / cv2.getTickFrequency() > captura_cada_segundo:
        ret, frame = cap.read()

        # Convertir la imagen a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        rostros = detector_rostros(gris)

        # Dibujar puntos y líneas más finas
        for rostro in rostros:
            puntos_faciales = predictor_facial(gris, rostro)

            # Dibujar puntos faciales más finos
            for punto in puntos_faciales.parts():
                cv2.circle(frame, (punto.x, punto.y), 1, (0, 255, 0), -1)

            # Dibujar líneas más finas entre puntos faciales
            for i in range(1, 17):  # Líneas para la ceja izquierda
                cv2.line(frame, (puntos_faciales.part(i - 1).x, puntos_faciales.part(i - 1).y),
                         (puntos_faciales.part(i).x, puntos_faciales.part(i).y), (0, 255, 0), 1)
            for i in range(18, 26):  # Líneas para la ceja derecha
                cv2.line(frame, (puntos_faciales.part(i - 1).x, puntos_faciales.part(i - 1).y),
                         (puntos_faciales.part(i).x, puntos_faciales.part(i).y), (0, 255, 0), 1)
            for i in range(27, 35):  # Líneas para la nariz
                cv2.line(frame, (puntos_faciales.part(i - 1).x, puntos_faciales.part(i - 1).y),
                         (puntos_faciales.part(i).x, puntos_faciales.part(i).y), (0, 255, 0), 1)
            for i in range(36, 47):  # Líneas para el ojo izquierdo
                cv2.line(frame, (puntos_faciales.part(i - 1).x, puntos_faciales.part(i - 1).y),
                         (puntos_faciales.part(i).x, puntos_faciales.part(i).y), (0, 255, 0), 1)
            for i in range(48, 59):  # Líneas para el ojo derecho
                cv2.line(frame, (puntos_faciales.part(i - 1).x, puntos_faciales.part(i - 1).y),
                         (puntos_faciales.part(i).x, puntos_faciales.part(i).y), (0, 255, 0), 1)
            for i in range(60, 67):  # Líneas para la boca
                cv2.line(frame, (puntos_faciales.part(i - 1).x, puntos_faciales.part(i - 1).y),
                         (puntos_faciales.part(i).x, puntos_faciales.part(i).y), (0, 255, 0), 1)

        # Mostrar la imagen en una ventana
        cv2.imshow('Escaneo Facial', frame)

        ultimo_tiempo = tiempo_actual

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
