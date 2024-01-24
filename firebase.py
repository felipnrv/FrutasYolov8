import pyrebase
import subprocess

def check_terminal_output():

    # Get terminal output
    child_name = input("Ingrese un nombre para guardar el registro: ")
    output = subprocess.check_output(["python", "main.py"])
    output_text = output.decode("utf-8")
    
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
        
    # Split output text into lines
    lines = output_text.split("\n")
        
    # Push each line to Firebase
    fruits = ["granadilla", "arandanos", "tomatearbol", "pitahaya", "maracuya"]
    
    for line in lines:
        for fruit in fruits:
            if fruit in line:
                db.child(child_name).push({"Fruta detectada": line, "resolucion": "480x640"})
                break
                
check_terminal_output()


