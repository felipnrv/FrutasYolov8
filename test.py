
from roboflow import Roboflow
rf = Roboflow(api_key="8049iBONyQC4bosBrcaa")
project = rf.workspace("proyecto-detec-frutas").project("frutas-nzthy")
dataset = project.version(1).download("yolov8")

data = {"name": "daniel ","age":"21","?":False}

#db.push(data)

#para poner el nombre de la llave
#db.child("users").child("Usuario").set(data)

#leer 
#felipe=db.child("users").child("Usuario").get()
#print(felipe.val())

#actualizar
#db.child("users").child("Usuario").update({"name":"pablo"})

#eliminar
#db.child("users").child("Usuario").child("age").remove()

#eliminar todo el nodo
db.child("users").child("Usuario").remove()

#m.model()
#data1=m.model()



#firebase = pyrebase.initialize_app(config)
#db = firebase.database()

#data = {"name": "daniel", "age": "21", "?": False}

# Push data to Firebase
#db.push(data)

# Get terminal output
output = subprocess.check_output(["python", "main.py"])

# Push terminal output to Firebase
db.push({"terminal_output": output.decode("utf-8")})