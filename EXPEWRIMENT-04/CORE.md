from google.colab import files
uploaded = files.upload()

from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Perform object detection
results = model("input.jpg", save=True)

# Print detected objects and confidence
for r in results:
    for box in r.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf) * 100
        print(model.names[class_id], ":", round(confidence, 2), "%")

# Display output image
output_image = cv2.imread("runs/detect/predict/input.jpg")
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

plt.imshow(output_image)
plt.axis("off")

#Output:
<img width="522" height="339" alt="Screenshot 2026-02-18 095033" src="https://github.com/user-attachments/assets/254929a3-d3d3-4794-a993-81f39f1dfc65" />
