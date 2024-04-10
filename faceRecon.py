import face_recognition
import cv2
import dlib
import serial
import boto3
import numpy as np
from tqdm import tqdm
import time
import random
import os
import logging

from paho.mqtt import client as mqtt_client

BROKER = 'mf631bef.ala.us-east-1.emqxsl.com'
PORT = 8883
TOPIC_BASE = "historial/alumno"  # Tema base para MQTT
CLIENT_ID = f'python-mqtt-tls-sub-{random.randint(0, 1000)}'
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60

FLAG_EXIT = False

def on_connect(client, userdata, flags, rc):
    if rc == 0 and client.is_connected():
        print("Connected to MQTT Broker!")
    else:
        print(f'Failed to connect, return code {rc}')

def on_disconnect(client, userdata, rc):
    logging.info("Disconnected with result code: %s", rc)
    reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
    while reconnect_count < MAX_RECONNECT_COUNT:
        logging.info("Reconnecting in %d seconds...", reconnect_delay)
        time.sleep(reconnect_delay)

        try:
            client.reconnect()
            logging.info("Reconnected successfully!")
            return
        except Exception as err:
            logging.error("%s. Reconnect failed. Retrying...", err)

        reconnect_delay *= RECONNECT_RATE
        reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
        reconnect_count += 1
    logging.info("Reconnect failed after %s attempts. Exiting...", reconnect_count)
    global FLAG_EXIT
    FLAG_EXIT = True

def on_message(client, userdata, msg):
    # Extraer el nombre del estudiante del mensaje recibido
    topic_parts = msg.topic.split('/')
    student_name_with_id = topic_parts[-1]  # Último elemento de la lista
    student_name = student_name_with_id.split('_')[0]  # Obtener el nombre antes del guión bajo
    # Construir el nuevo topic sin el identificador
    new_topic = '/'.join(topic_parts[:-1]) + '/' + student_name
    # Imprimir el nuevo topic y el mensaje recibido
    print(f'Received `{msg.payload.decode()}` from `{new_topic}` topic')

def connect_mqtt():
    client = mqtt_client.Client(CLIENT_ID)
    client.tls_set(ca_certs='./emqxsl-ca.cert')
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, keepalive=120)
    client.on_disconnect = on_disconnect
    return client

def run_face_recognition():
    # Cargar el modelo de puntos faciales
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Configurar la conexión a DynamoDB
    try:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        list_tables = dynamodb.meta.client.list_tables()
        print(f"Conexión exitosa con DynamoDB: {list_tables['TableNames']}")
        if not list_tables['TableNames']:
            print("No hay tablas en la base de datos")
        else:
            print(f"Tablas en la base de datos: {list_tables['TableNames']}")
        table_name = 'faceReconMetadata'
        table = dynamodb.Table(table_name)
    except Exception as e:
        print(f"Error al conectar con DynamoDB: {e}")
        exit()

    # Obtener las codificaciones y nombres de las imágenes almacenadas en DynamoDB
    encodings_by_name = {}
    response = table.scan()
    for item in response['Items']:
        # Convertir la codificación facial a lista de numpy array y tipo float
        encoding = np.array([float(coord) for coord in item['Codificacion']], dtype=float)
        name = item['Nombre']

        # Agrupar las codificaciones por nombre
        if name not in encodings_by_name:
            encodings_by_name[name] = []
        encodings_by_name[name].append(encoding)

    # Transformar a la lista de tuplas (nombre, lista de codificaciones)
    encodings_and_names = [(name, encodings) for name, encodings in encodings_by_name.items()]

    tqdm.write("Imágenes cargadas desde DynamoDB")

    # Abrir la cámara; ajusta el número según tu configuración
    cap = cv2.VideoCapture(0)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir la cámara")
        exit()

    tqdm.write("Imágenes cargadas")

    frame_count = 1
    recognition_frequency = 6

    current_face_names = []
    face_locations = []
    detection_counter = {name: 0 for name, _ in encodings_and_names}
    detection_threshold = 8 ## fotos que se tomaron para que se abra el torniquete

    arduino_serial = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    entered_students = set()

    client = connect_mqtt()
    client.loop_start()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error capturando el fotograma o el final del flujo de video")
            break

        if frame_count % recognition_frequency == 0:
            rgb_frame = frame[:, :, ::-1]
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.23, fy=0.23)

            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            current_face_names = []

            for face_location, face_encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = face_location
                face_box = dlib.rectangle(left, top, right, bottom)

                landmarks = predictor(small_frame, face_box)

                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x * 4, y * 4), 1, (0, 255, 0), -1)

                tolerance = 0.4
                face_encoding = np.array(face_encoding, dtype=float)

                match_found = False
                for name, encodings in encodings_and_names:
                    for known_encoding in encodings:
                        match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
                        if any(match):
                            current_face_names.append(name)
                            match_found = True
                            break
                    if match_found:
                        break

                if not match_found:
                    current_face_names.append("Desconocido")

        for (top, right, bottom, left), name in zip(face_locations, current_face_names):
            cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
            cv2.rectangle(frame, (left * 4, (bottom * 4) - 35), (right * 4, bottom * 4), (0, 100, 255), cv2.FILLED)
            cv2.putText(frame, name, (left * 4 + 6, (bottom * 4) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

            # Verificar si el nombre no es 'Desconocido' antes de incrementar el contador
            if name != 'Desconocido':
                # Verificar si la clave existe en el diccionario
                if name in detection_counter:
                    detection_counter[name] += 1
                else:
                    detection_counter[name] = 1

                if detection_counter[name] >= detection_threshold:
                    #time.sleep(0.5)
                    print(f"Persona {name} detectada y servo activado!")
                    
                    # Construir el tema MQTT para el alumno
                    #crear un nuevo nombre ya que este esta con _ este necesitamos que sean los puros numeros 
                    nameWitdNo = name.split('_')[0]
                    # quitamos el _ 
                    nameWitdNo = nameWitdNo.replace("_", "")
                    topic = f"{TOPIC_BASE}/{nameWitdNo}"

                    # Publicar el nombre del estudiante y la hora de ingreso
                    msg = f"{name} ingresó a las {time.strftime('%H:%M:%S')}"
                    result = client.publish(topic, msg, qos=0)
                    status = result[0]
                    if status == 0:
                        print(f"Mensaje publicado exitosamente: {msg}")
                    else:
                        print(f"Fallo al publicar mensaje con estado: {status}")

                    if name not in entered_students:
                        #time.sleep(0.2)
                        arduino_serial.write(b'M')  # Enviar comando para abrir el torniquete
                        entered_students.add(name)
                        ## imprimir el comando que le manda a la arduino
                        print("M")

                    else:
                        #time.sleep(0.2)
                        arduino_serial.write(b'R')  # Enviar comando para cerrar el torniquete
                        entered_students.remove(name)
                        print("R")  

                    detection_counter[name] = 0

        cv2.imshow('Camera', frame)
        cv2.waitKey(1)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_face_recognition()
