import cv2
from ultralytics import YOLO
import supervision as sv

#Global variables#--------------------------------------
LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)
object1_id = 41 # Object ID of: cup
object2_id = 39 # Object ID of: bottle

def create_labels(model, detections):
    labels = [
        f"# {class_id}{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections 
    ]
    return labels
    

class object_line_counter():
    def __init__(self,class_id_number,line_counter):
        self.class_id_number = class_id_number
        self.line_counter = line_counter

    def detections(self,result):
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == self.class_id_number]

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)  
            
        self.line_counter.trigger(detections=detections)
        count_in = self.line_counter.in_count
        count_out = self.line_counter.out_count

        return count_in,count_out


def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_counter1 = sv.LineZone(start=LINE_START, end=LINE_END)
    line_counter2 = sv.LineZone(start=LINE_START, end=LINE_END)

    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5)

    # Create class objects
    object1 = object_line_counter(object1_id, line_counter1)
    object2 = object_line_counter(object2_id, line_counter2)

    model = YOLO("yolov8l.pt")

    for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
        
        frame = result.orig_img

        #Detect al objects
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id != 0]
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        labels = create_labels(model,detections)

        #Detect specific objects
        object1_in,object1_out = object1.detections(result)
        object2_in,object2_out = object2.detections(result)


        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        
        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        print('object1_in',object1_in, 'object1_out',object1_out)
        print('object2_in',object2_in, 'object2_out',object2_out)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()