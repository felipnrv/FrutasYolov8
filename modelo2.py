import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np


LINE_START = sv.Point(320, 0) #punto de inicio de la linea
LINE_END = sv.Point(320, 480) #punto final de la linea


def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END) #se crea una linea
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)#se crea un anotador de linea
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    model = YOLO("D:/Felipe/Codigo/Model/frutasmdl.pt")
    for result in model.track(source=1, show=True, stream=True, agnostic_nms=True): #es el modelo de deteccion de frutas
        
        frame = result.orig_img #es el frame de la camara,orig_img es el frame original
        detections = sv.Detections.from_yolov8(result) #es el resultado de la deteccion de frutas, y las detecciones de las frutas

        if result.boxes.id is not None: #si el id de las cajas no es nulo
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int) #se le asigna a las detecciones el id de las cajas
        
        detections = detections[(detections.class_id != 60) ]

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ] 
        # arriba solo es para mostrar el nombre de la fruta y el id del tracker
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        line_counter.trigger(detections=detections) #se activa el contador de la linea
        line_annotator.annotate(frame=frame, line_counter=line_counter) #se anota la linea

        cv2_imshow = True
        if cv2_imshow:
            cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()