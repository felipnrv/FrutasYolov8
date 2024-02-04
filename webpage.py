from flask import Flask 
from flask import render_template
from flask import Response
import cv2

app = Flask(__name__,template_folder='plantilla') #se inicializa la aplicacion

def modelo():


@app.route('/')#se crea una ruta
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True) #al activar el debug se reinicia el servidor cada vez que se hace un cambio en el codigo