import pyrebase
import subprocess

def check_terminal_salida():

    # Get terminal output
    child_nombre = input("Ingrese un nombre para guardar el registro: ")
    completed_process = subprocess.run(["python", "modelo.py"], capture_output=True, text=True)
    output_text = completed_process.stdout
    
    
    # Initialize Firebase
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
        
    frutas = ["granadilla", "arandanos", "tomatearbol", "pitahaya", "maracuya"]
    # Split output text into lines
    linea = output_text.split("\n")
    
    lineas_salida = set()
    #1 EJERCICIO DE BRAZOS 

    
    for line in linea: #para cada linea en lines
        if line not in lineas_salida: #si la linea no esta en existing_lines
            lineas_salida.add(line) #agregar la linea a existing_lines
            for fruta in frutas: #para cada fruta en fruits
                if fruta in line: #si la fruta esta en la linea
                    db.child(child_nombre).push({"Fruta detectada": line, "resolucion": "480x640"}) # se hace push a la base de datos
                    break
               
check_terminal_salida()

#registro
    #NJDIWDJOQDOI
        #frutas detectada : 1MARACUYA 
        #RESOLUCION 640X480
