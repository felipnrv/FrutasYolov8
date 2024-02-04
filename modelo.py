import cv2
from ultralytics import YOLO
import supervision as sv
import pyrebase
from flask import Flask 
from flask import render_template
from flask import Response
import datetime as dt

app = Flask(__name__,template_folder='plantilla') #se inicializa la aplicacion

config = {
        "apiKey": "AIzaSyCC1C7eIcb6Q0-WeWKuzSrZNwVoSyVf8Lw",
        "authDomain": "dbfrutas-dd3ba.firebaseapp.com",
        "projectId": "dbfrutas-dd3ba",
        "databaseURL": "https://dbfrutas-dd3ba-default-rtdb.firebaseio.com/",
        "storageBucket": "dbfrutas-dd3ba.appspot.com",
        "messagingSenderId": "404825544625",
        "appId": "1:404825544625:web:efaa68356d01d1153ccf90",
        "measurementId": "G-EMQTJ61ZL6"
    }



#Global variables#--------------------------------------
LINEA_INICIO = sv.Point(450, 0) #punto de inicio de la linea,la coordenada x es 320 y la coordenada y es 0
LINEA_FINAL = sv.Point(450, 480) #punto final de la line, la coordenada x es 320 y la coordenada y es 480

fruta1_grand = 1 # granadilla
fruta2_mango = 2 # mango
fruta3_marac = 3 # maracuya
fruta4_pitah = 4 # pitahaya
fruta5_tomate = 5# tomate de arbol

child_nombre = dt.datetime.now()
child_nombre = child_nombre.strftime(format='%Y-%m-%d %H:%M:%S')
#una clase es un objeto que tiene atributos y metodos y una funcion es un bloque de codigo que se ejecuta cuando se llama
class linea_conteo_class():
    def __init__(self,id_clase,linea_conteo):
        self.id_clase = id_clase #es el numero de la clase que se quiere contar
        self.linea_conteo = linea_conteo #es el contador de la linea

    def detections(self,result):
        deteccion = sv.Detections.from_yolov8(result) #se obtienen las detecciones
        deteccion = deteccion[deteccion.class_id == self.id_clase] #se filtran las detecciones de la clase que se quiere contar

        if result.boxes.id is not None:#se asigna el id del tracker a las detecciones
            deteccion.tracker_id = result.boxes.id.cpu().numpy().astype(int) #se asigna el id del tracker a las detecciones,cpu para convertir a numpy y astype para convertir a int
            # se usa boxes.id.cpu().numpy() para obtener el id del tracker, es cpu porque se esta trabajando con un tensor de pytorch

        self.linea_conteo.trigger(detections=deteccion)#se cuentan las detecciones que cruzan la linea
        conteo_in = self.linea_conteo.in_count#se obtiene el numero de detecciones que cruzaron la linea,in de izquierda a derecha
        conteo_out = self.linea_conteo.out_count#se obtiene el numero de detecciones que cruzaron la linea, out de derecha a izquierda

        return conteo_in,conteo_out


def main():
    linea_2 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)#line counter es el contador de la linea
    linea_1 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)
    linea_3 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)
    linea_4 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)
    linea_5 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)

    #LinezoneAnnotator es donde se esconde el contador de la linea
    linea_anotador = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5) #se crea el contador de la linea
    caja_anotador = sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5) #se crea el objeto que permite anotar las cajas

    # Create class objects
    granadilla = linea_conteo_class(fruta1_grand, linea_1)
    mango = linea_conteo_class(fruta2_mango, linea_2)
    maracuya = linea_conteo_class(fruta3_marac, linea_3)
    pitahaya = linea_conteo_class(fruta4_pitah, linea_4)
    tomatearbol = linea_conteo_class(fruta5_tomate, linea_5)

    model = YOLO("D:/Felipe/Codigo/Model/frutasmdl.pt")

    for result in model.track(source=0, show=False, stream=True, agnostic_nms=True):
        
        frame = result.orig_img

        #Detect al objects
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id != 10]
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        #labels = create_labels(model,detections)
        labels = [
        f"# {class_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id,tracker_id
        in detections 
    ]

        #Detect specific objects
        mango_in,mango_out = mango.detections(result)
        granadilla_in,granadilla_out = granadilla.detections(result)
        maracuya_in,maracuya_out = maracuya.detections(result)
        pitahaya_in,pitahaya_out = pitahaya.detections(result)
        tomatearbol_in,tomatearbol_out = tomatearbol.detections(result)


        frame = caja_anotador.annotate(scene=frame, detections=detections, labels=labels)

        
        linea_2.trigger(detections=detections)
        linea_anotador.annotate(frame=frame, line_counter=linea_2)

        """	
        print('Mango_in',mango_in, 'Mango_out',mango_out)
        print('Granadilla_in',granadilla_in, 'Granadilla_out',granadilla_out)
        print('Maracuya_in',maracuya_in, 'Maracuya_out',maracuya_out)
        print('Pitahaya_in',pitahaya_in, 'Pitahaya_out',pitahaya_out)
        print('Tomatearbol_in',tomatearbol_in, 'Tomatearbol_out',tomatearbol_out)
        """
        print('Mango',mango_in)
        print('Granadilla',granadilla_in)
        print('Maracuya',maracuya_in)
        print('Pitahaya',pitahaya_in)
        print('Tomatearbol',tomatearbol_in)

        #cv2.imshow("yolov8", frame)

        if (cv2.waitKey(1) == 27):
            break

        firebase = pyrebase.initialize_app(config)
        db = firebase.database()

        """
        db.child(child_nombre).update(
            {"Mango_Entrada": mango_in, "Mango_Salida": mango_out,
            "Granadilla_Entrada": granadilla_in, "Granadilla_Salida": granadilla_out,
            "Maracuya_Entrada": maracuya_in, "Maracuya_Salida": maracuya_out,
            "Pitahaya_Entrada": pitahaya_in, "Pitahaya_Salida": pitahaya_out,
            "Tomatearbol_Entrada": tomatearbol_in, "Tomatearbol_Salida": tomatearbol_out})
        """
        db.child(child_nombre).update(
            {"Mango_Entrada": mango_in,
            "Granadilla_Entrada": granadilla_in,
            "Maracuya_Entrada": maracuya_in,
            "Pitahaya_Entrada": pitahaya_in,
            "Tomatearbol_Entrada": tomatearbol_in})
        

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

@app.route('/')#se crea una ruta
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    #main()
    app.run(debug=False)

