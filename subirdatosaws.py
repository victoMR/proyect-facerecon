import os
import face_recognition
import dlib
import boto3
from tqdm import tqdm
from decimal import Decimal

# Cargar el modelo de puntos faciales
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Configurar la conexión a DynamoDB
try:
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    # Intenta realizar una operación básica para verificar la conexión
    list_tables = dynamodb.meta.client.list_tables()
    print(f"Conexión exitosa con DynamoDB: {list_tables['TableNames']}")
    # Si la conexión es exitosa, pero no hay tablas, se mostrará un mensaje
    if not list_tables['TableNames']:
        print("No hay tablas en la base de datos")
    else:
        print(f"Tablas en la base de datos: {list_tables['TableNames']}")
except Exception as e:
    print(f"Error al conectar con DynamoDB: {e}")
    exit()

# Nombres correspondientes a cada rango de imágenes
names = ["Desconocido", "2022143009_Abril", "2022143069_Vic", "2022143063_Mau", "2022143015_Palo"]

# Ruta al directorio que contiene las imágenes
img_directory = '/home/pi/Documents/face_recon/img'

# Obtener la tabla de DynamoDB
table_name = 'faceReconMetadata'
table = dynamodb.Table(table_name)

# Utiliza tqdm para mostrar una barra de progreso en la consola
for i in tqdm(range(40), desc="Cargando datos"):
    img_path = f"{img_directory}/img{i+1}.JPG"
    img = face_recognition.load_image_file(img_path)

    # Asignar nombres basados en el rango de imágenes
    if i < 10:
        name = names[1]
    elif i < 20:
        name = names[2]
    elif i < 30:
        name = names[3]
    elif i < 40:
        name = names[4]

    # Intentar obtener codificaciones faciales
    try:
        encoding = face_recognition.face_encodings(img)[0]
    except IndexError:
        print(f"No se detectó ninguna cara en la imagen {img_path}")
        continue

    # Generar una clave única para cada elemento
    item_id = f"{name}_{i+1}"

    # Almacenar en DynamoDB
    item = {
        'Id': item_id,
        'Nombre': name,
        'Codificacion': [Decimal(num) for num in encoding.tolist()]
    }

    # Almacenar en DynamoDB
    with table.batch_writer() as batch:
        print(f"Almacenando {item['Id']} en DynamoDB")
        batch.put_item(Item=item)
