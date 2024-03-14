import pymongo
from dotenv import load_dotenv
import os
from paho.mqtt import client as mqtt_client

broker = 'xee4876e.us-east-1.emqx.cloud'
port = 15280
topic = "alumnos/entradas"
client_id = f'python-mqtt-subscriber'
username = 'pi'
password = '123'

load_dotenv()

# Configuración de MongoDB
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["alumnos"]
collection = db["entradas"]

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Conectado al broker MQTT")
            client.subscribe(topic)
        else:
            print("Fallo al conectar al broker MQTT")

    def on_message(client, userdata, msg):
        message = msg.payload.decode()
        print(f"Mensaje recibido: {message}")
        parse_and_save_message(message)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(broker, port)
    except Exception as e:
        print(f"Fallo al conectar al broker MQTT: {e}")
        exit()
    return client

def test_connection_mongo():
    try:
        mongo_client.server_info()
        print("Conexión exitosa con MongoDB")
    except Exception as e:
        print(f"Error al conectar con MongoDB: {e}")
        exit()

def parse_and_save_message(message):
    parts = message.split("_")
    if len(parts) == 2:
        student_id = parts[0]
        name_time = parts[1].split(" ingresó a las ")
        if len(name_time) == 2:
            name = name_time[0]
            time = name_time[1]

            # Crear un diccionario con la información extraída
            data = {"student_id": student_id, "name": name, "time": time}
            
            # Guardar la entrada en MongoDB
            save_entry_to_mongo(data)
        else:
            print("Formato de mensaje incorrecto: falta la hora de ingreso")
    else:
        print("Formato de mensaje incorrecto: falta el ID del estudiante o el nombre")

def save_entry_to_mongo(data):
    try:
        collection.insert_one(data)
        print("Entrada guardada en MongoDB")
    except Exception as e:
        print(f"Error al guardar entrada en MongoDB: {e}")

def run():
    test_connection_mongo()
    client = connect_mqtt()
    client.loop_forever()

if __name__ == '__main__':
    run()
