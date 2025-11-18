from ultralytics import YOLO
import cv2

# Load YOLO model (people detection)
model = YOLO("yolov8n.pt")  # lightweight model, good for CPU

# Load image
img_path = "test.jpg"  # put your image name here
results = model(img_path)

# Count number of persons (class 0 = person)
person_count = 0

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # YOLO class ID for 'person'
            person_count += 1

# Print result
print("Estimated Crowd Count:", person_count)

# OPTIONAL: Draw boxes and show image
annotated = results[0].plot()
cv2.imshow("YOLO Crowd Counting", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
