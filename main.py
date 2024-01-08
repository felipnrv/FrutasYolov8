
###
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


model = YOLO('D:/Felipe/Codigo/Model/best.pt')
cap = cv2.VideoCapture(0)

counter = object_counter.ObjectCounter()#crea un objeto contador

points=[(0,300),(1080,300)]#puntos de la region de conteo, se puede cambiar por [(200,600),(200,600),(400,600),(400,600)
counter.set_args(view_img=True,reg_pts=points,classes_names=model.names,draw_tracks=False)#configura el objeto contador


while cap.isOpened():
    ret,frame=cap.read()

    if not ret:
        print("No se pudo obtener la imagen")
        exit(0)
    tracks=model.track(frame,persist=True,show=False) #persist es para que se mantenga el objeto en pantalla,show es para que se muestre la imagen
    counter.start_counting(frame,tracks) #tracks es una lista de objetos detectados, cada objeto tiene un id, un rectangulo y una clase

    #Para cerrar la ventana se presiona la tecla k
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

