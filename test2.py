import serial
import time

# Configura la conexión serial
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Ajusta el puerto serial según tu configuración

try:
    while True:
        # Espera a que ingreses un comando desde la consola
        command = input("Ingrese el comando (0 para 0 grados, 90 para 90 grados, 'exit' para salir): ")

        # Verifica si el comando es 'exit' para salir del bucle
        if command.lower() == 'exit':
            break

        # Envía el comando al Arduino a través del puerto serie
        ser.write(command.encode())
        time.sleep(1)  # Espera un segundo antes de volver a aceptar un comando

except KeyboardInterrupt:
    print("\nPrograma cerrado por el usuario.")

finally:
    ser.close()  # Cierra la conexión serial
