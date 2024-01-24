# import cv2
# import time

# # Abrir la cámara; ajusta el número según tu configuración
# cap = cv2.VideoCapture(0)

# # Checar si la cámara se abrió correctamente
# if not cap.isOpened():
#     print("Error")
#     exit()

# # Archivo para almacenar datos de rendimiento
# performance_file = open('performance_data.txt', 'w')

# while True:
#     # Capturar frame por frame
#     ret, frame = cap.read()

#     # Si el frame no se captura correctamente
#     if not ret:
#         print("Error al capturar el frame")
#         break

#     # Mostrar el frame capturado
#     cv2.imshow('Camera', frame)

#     # Calcular y almacenar el tiempo actual en milisegundos
#     timestamp = int(time.time() * 1000)
    
#     # Escribir el timestamp en el archivo
#     performance_file.write(f"{timestamp},{time.process_time()}\n")

#     # Presionar 'q' para salir del programa
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     # Esperar 10ms para que se pueda ver el frame capturado en la ventana 
#     time.sleep(0.01)

# # Liberar la cámara y cerrar todas las ventanas
# cap.release()
# cv2.destroyAllWindows()

# # Cerrar el archivo de rendimiento
# performance_file.close()

# programa que abra las imagenes de img1 y img2 y las muestre en pantalla
import cv2

def display_images(image_paths):
    for path in image_paths:
        # Lee la imagen desde la ruta especificada
        img = cv2.imread(path)

        # Verifica si la imagen se ha leído correctamente
        if img is None:
            print(f"No se pudo leer la imagen en {path}")
            continue

        # Muestra la imagen en una ventana
        cv2.imshow('Imagen', img)

        # Espera a que se presione una tecla (0 significa esperar indefinidamente)
        cv2.waitKey(0)

        # Cierra la ventana después de presionar una tecla
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Lista de rutas de las imágenes que deseas abrir
    image_paths = [
        "/home/pi/Documents/face_recon/img/img1.jpeg",
         "/home/pi/Documents/face_recon/img/img2.jpeg",
        # Agrega más rutas según sea necesario
    ]

    # Llama a la función para mostrar las imágenes
    display_images(image_paths)
