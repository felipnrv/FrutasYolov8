
import cv2
import os

objname=input('Ingresa el nombre de la carpeta: ')
datapath = 'D:/Felipe/Codigo/Modelo'
path=datapath+ '/'+ objname

if not os.path.exists(path):
    print('Carpeta creada: ',objname)
    os.makedirs(path)

cap = cv2.VideoCapture(1)
count = 0
while True:
  
    ret,frame = cap.read()
    cv2.imwrite(path +'/obj_{}.jpg'.format(count),frame)     # save frame as JPEG file      
    
    print('Read a new frame: ', ret)
    count += 1
    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    if k == 27 or count >= 50:    
        break


cap.release()
cv2.destroyAllWindows()