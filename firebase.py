import pyrebase
#import main as m
import subprocess

def check_terminal_output():
    # Get terminal output
    output = subprocess.check_output(["python", "main.py"])
    output_text = output.decode("utf-8")
    
    if "detections" in output_text:
        # Initialize Firebase
        config = {
            "apiKey": "AIzaSyCC1C7eIcb6Q0-WeWKuzSrZNwVoSyVf8Lw",
            "authDomain": "dbfrutas-dd3ba.firebaseapp.com",
            "databaseURL": "https://dbfrutas-dd3ba-default-rtdb.firebaseio.com",
            "projectId": "dbfrutas-dd3ba",
            "databaseURL": "https://dbfrutas-dd3ba-default-rtdb.firebaseio.com/",
            "storageBucket": "dbfrutas-dd3ba.appspot.com",
            "messagingSenderId": "404825544625",
            "appId": "1:404825544625:web:efaa68356d01d1153ccf90",
            "measurementId": "G-EMQTJ61ZL6"
        }
        firebase = pyrebase.initialize_app(config)
        db = firebase.database()
        
        # Push terminal output to Firebase
        db.push({"terminal_output": output_text})
        
check_terminal_output()

