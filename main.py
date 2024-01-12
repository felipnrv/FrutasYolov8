
###
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


outfile = open('new3.0.txt', 'w')
model = YOLO('D:/Felipe/Codigo/Model/best.pt')
cap = cv2.VideoCapture(0)

counter = object_counter.ObjectCounter()#crea un objeto contador

points=[(0,300),(1080,300)]#puntos de la region de conteo, se puede cambiar por [(200,600),(200,600),(400,600),(400,600)
counter.set_args(view_img=True,reg_pts=points,classes_names=model.names,draw_tracks=False)#configura el objeto contador


while True:
    ret,frame=cap.read()

    if not ret:
        print("No se pudo obtener la imagen")
        exit(0)
    tracks=model.track(frame,persist=True,show=False) #persist es para que se mantenga el objeto en pantalla,show es para que se muestre la imagen
    counter.start_counting(frame,tracks) #tracks es una lista de objetos detectados, cada objeto tiene un id, un rectangulo y una clase
    
    results=model.predict(frame,save_txt=True,save_conf=True) #save_img es para guardar la imagen con los objetos detectados

    #Para cerrar la ventana se presiona la tecla k
    outfile.write(str(tracks)) 
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        outfile.close()
        break
    

cap.release()
cv2.destroyAllWindows()

