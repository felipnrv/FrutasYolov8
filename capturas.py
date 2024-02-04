import cv2
import os
# Abre el video en vivo
cap = cv2.VideoCapture(1)

# Inicializa el contador de capturas
capturas = 0

directorio_capturas = 'D:\Felipe\Codigo\Img'

        

while True:
    # Lee el frame del video
    ret, frame = cap.read()

    # Muestra el frame en una ventana
    cv2.imshow('Video en vivo', frame)

    # Guarda el frame como una imagen
    cv2.imwrite(os.path.join(directorio_capturas, f'captura_{capturas}.jpg'), frame)

    # Incrementa el contador de capturas
    capturas += 1

    # Si se han tomado 300 capturas, sale del bucle
    if capturas == 150:
        break

    # Espera 1 milisegundo y verifica si se presiona la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
