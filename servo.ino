#include <Servo.h>

Servo myservo;  // Objeto servo para controlar un servo motor

void setup() {
  myservo.attach(9);  // El pin del servo está conectado al pin 9
  Serial.begin(9600); // Inicializa la comunicación serie a 9600 baudios
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();  // Lee el comando desde el puerto serie
    if (command == 'M') {
      // Gira el servo a la posición deseada para abrir el torniquete
      myservo.write(90);
    } else if (command == 'R') {
      // Gira el servo a la posición deseada para cerrar el torniquete
      myservo.write(0);
    }
  }
}