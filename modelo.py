import cv2
from ultralytics import YOLO
import supervision as sv
import pyrebase
from flask import Flask,render_template,Response,request,redirect,url_for 
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


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

firebase = pyrebase.initialize_app(config)
db = firebase.database()
auth = firebase.auth()

LINEA_INICIO = sv.Point(450, 0) #punto de inicio de la linea,la coordenada x es 320 y la coordenada y es 0
LINEA_FINAL = sv.Point(450, 480) #punto final de la line, la coordenada x es 320 y la coordenada y es 480

fruta1_grand = 0 # granadilla
fruta2_mango = 1 # mango
fruta3_marac = 2 # maracuya
fruta4_pitah = 3 # pitahaya
fruta5_tomate = 4# tomate de arbol

child_nombre = dt.datetime.now()
child_nombre = child_nombre.strftime(format='%Y-%m-%d')
#una clase es un objeto que tiene atributos y metodos y una funcion es un bloque de codigo que se ejecuta cuando se llama


class linea_conteo_class():
    def __init__(self,id_clase,linea_conteo):
        self.id_clase = id_clase #es el numero de la clase que se quiere contar
        self.linea_conteo = linea_conteo #es el contador de la linea

    def detections(self,result):
        deteccion = sv.Detections.from_yolov8(result) #se obtienen las detecciones
        deteccion = deteccion[deteccion.class_id == self.id_clase] #se filtran las detecciones de la clase que se quiere contar

        if result.boxes.id is not None:#se asigna el id del tracker a las detecciones
            deteccion.tracker_id = result.boxes.id.cpu().numpy().astype(int) 
            # se usa boxes.id.cpu().numpy() para obtener el id del tracker, es cpu porque se esta trabajando con un tensor de pytorch

        self.linea_conteo.trigger(detections=deteccion)#se cuentan las detecciones que cruzan la linea
        conteo_in = self.linea_conteo.in_count#in de derecha a izquierda
        conteo_out = self.linea_conteo.out_count#out de izquierda a derecha

        return conteo_in,conteo_out

#@app.route('/')

def main():

    mango_out=0
    maracuya_out=0

    linea = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL) #se crea el contador de la linea
    linea_1 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)
    linea_2 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)#line counter es el contador de la linea
    linea_3 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)
    linea_4 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)
    linea_5 = sv.LineZone(start=LINEA_INICIO, end=LINEA_FINAL)

    #LinezoneAnnotator es donde se esconde el contador de la linea
    linea_anotador = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5) #se crea el contador de la linea
    
    caja_anotador = sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5) #se crea el objeto que permite anotar las cajas

    #se crea un objeto de la clase linea_conteo_class
    granadilla = linea_conteo_class(fruta1_grand, linea_1)
    mango = linea_conteo_class(fruta2_mango, linea_2)
    maracuya = linea_conteo_class(fruta3_marac, linea_3)
    pitahaya = linea_conteo_class(fruta4_pitah, linea_4)
    tomatearbol = linea_conteo_class(fruta5_tomate, linea_5)

    model = YOLO("D:/Felipe/Codigo/Model/frutasmdl.pt")

    stop_detection = False
    for result in model.track(source=0, show=False, stream=True, agnostic_nms=True):
        
        frame = result.orig_img

        
        detections = sv.Detections.from_yolov8(result)

        detections = detections[detections.class_id != 10]
        detections1 = detections[detections.class_id == 0]
        detections2 = detections[detections.class_id == 1]
        detections3 = detections[detections.class_id == 2]
        detections4 = detections[detections.class_id == 3]
        detections5 = detections[detections.class_id == 4]
      
        if result.boxes.id is not None:#es para el conteo de las detecciones
            detections1.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        if result.boxes.id is not None: #se asigna el id del tracker a las detecciones
            detections2.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            
        if result.boxes.id is not None: 
            detections3.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        if result.boxes.id is not None: 
            detections4.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        if result.boxes.id is not None: 
            detections5.tracker_id = result.boxes.id.cpu().numpy().astype(int)
    
        
        labels = [ #se crea una lista con las etiquetas de las detecciones
        f"# {class_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id,tracker_id
        in detections 
        ]
        #Detect specific objects
        maracuya_in,maracuya_out = maracuya.detections(result)
        granadilla_in,granadilla_out = granadilla.detections(result)
        mango_in,mango_out = mango.detections(result)
        pitahaya_in,pitahaya_out = pitahaya.detections(result)
        tomatearbol_in,tomatearbol_out = tomatearbol.detections(result)
        
        frame = caja_anotador.annotate(scene=frame, detections=detections, labels=labels)

        linea_1.trigger(detections=detections1)
        linea_anotador.annotate(frame=frame, line_counter=linea)
            
        linea_2.trigger(detections=detections2)
        linea_anotador.annotate(frame=frame, line_counter=linea)

        linea_3.trigger(detections=detections3)
        linea_anotador.annotate(frame=frame, line_counter=linea)

        linea_4.trigger(detections=detections4)
        linea_anotador.annotate(frame=frame, line_counter=linea)

        linea_5.trigger(detections=detections5)
        linea_anotador.annotate(frame=frame, line_counter=linea)

        
        y_offset = 50  # Espacio vertical entre cada línea de texto

        mango_text = f"Mango: {mango_out}"
        cv2.putText(frame, mango_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)#color negro en rgb es (0,0,0)
        y_offset += 30  # Incrementar el desplazamiento vertical

        granadilla_text = f"Granadilla: {granadilla_out}"
        cv2.putText(frame, granadilla_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)#0.5 es el tamaño de la letra,1 es el grosor de la letra
        y_offset += 30

        maracuya_text = f"Maracuya: {maracuya_out}"
        cv2.putText(frame, maracuya_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        y_offset += 30

        pitahaya_text = f"Pitahaya: {pitahaya_out}"
        cv2.putText(frame, pitahaya_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        y_offset += 30

        tomatearbol_text = f"Tomate de arbol: {tomatearbol_out}"
        cv2.putText(frame, tomatearbol_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)#blanco en rgb es (255,255,255)
        y_offset += 30

        if mango_out != 0:
            print('Mango', mango_out)
        if granadilla_out != 0:
            print('Granadilla', granadilla_out)
        if maracuya_out != 0:
            print('Maracuya', maracuya_out)
        if pitahaya_out != 0:
            print('Pitahaya', pitahaya_out)
        if tomatearbol_out != 0:
            print('Tomatearbol', tomatearbol_out)

        fecha = pd.to_datetime("today")

        data={"Mango": [mango_out],"Maracuya":[maracuya_out],
              "Granadilla":[granadilla_out],
              "Pitahaya":[pitahaya_out],
              "Tomate de arbol":[tomatearbol_out]}
        
        df = pd.DataFrame(data)

        fecha_actual = pd.to_datetime("today").strftime("%d-%m-%Y")

            # Crear la tabla
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis('off')

            # Agregar título
        ax.text(0.5, 1.05, "Informe del conteo de frutas - " + fecha_actual, ha='center', fontsize=14)

            # Mostrar la tabla
        ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')

        

        plt.savefig('output1.png', dpi=300, bbox_inches='tight')


        

       
        db.child(child_nombre).update(
            {"Mango_Entrada": mango_out,
            "Granadilla_Entrada": granadilla_out,
            "Maracuya_Entrada": maracuya_out,
            "Pitahaya_Entrada": pitahaya_out,
            "Tomatearbol_Entrada": tomatearbol_out})
        

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
        
    return {"mango: ":mango_out,"maracuya: ":maracuya_out}
    #return render_template('index.html',mango_out_list=mango_out_list)

@app.route('/')#se crea una ruta
def main_page():
    return render_template('main.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Obtener datos del formulario
        nombre = request.form['nombre']
        apellido = request.form['apellido']
        email = request.form['email']
        password = request.form['password']

        # Enviar datos a la base de datos de Firebase
        try:
            db.child(child_nombre).child(nombre).update({
                "Nombre": nombre,
                "Apellido": apellido,
                "Correo": email,
                "Contraseña": password
            })
        except Exception as e:
            # Manejar errores si hay algún problema al actualizar la base de datos
            print("Error:", e)
            return "Error al registrar. Por favor, inténtalo de nuevo más tarde."
        
        
        user=auth.create_user_with_email_and_password(email, password)
        	
        # Redirigir a la página principal después del registro exitoso
        return redirect(url_for('main_page'))

    # Renderizar la plantilla de registro
    return render_template('registro.html')
    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Firebase Authentication
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            return redirect(url_for('video'))
        except Exception as e:
            print("Error:", e)
            return render_template('login.html', error=str(e))

    return render_template('login.html')

@app.route('/video')#se crea una ruta
def video():
    
    return render_template('video.html')


@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    #main()
    app.run(debug=True)

