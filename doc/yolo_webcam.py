from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")  # fast and works on CPU

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not detected.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    person_count = 0

    # Process results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # YOLO class ID for 'person'
                person_count += 1

        # Draw bounding boxes and labels
        annotated_frame = r.plot()

    # Display count on screen
    cv2.putText(annotated_frame,
                f"People Count: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3)

    # Show output
    cv2.imshow("YOLO Webcam Crowd Counting", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()
