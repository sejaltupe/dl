pip install ultralytics opencv-python matplotlib

import zipfile
import os

zip_path = "archive (1).zip"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(".")

print("Dataset extracted successfully!")


from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("yolov8n.pt")

print(os.listdir("."))

print(os.listdir("dogscats\images"))

dataset_path = r"dogscats/images"

print("Categories:", os.listdir(dataset_path))

print("Cat images:",
      len(os.listdir(os.path.join(dataset_path, "cats"))))

print("Dog images:",
      len(os.listdir(os.path.join(dataset_path, "dogs"))))


dataset_path = r"dogscats/images"

cat_folder = os.path.join(dataset_path, "cats")
dog_folder = os.path.join(dataset_path, "dogs")

cat_images = [
    f for f in os.listdir(cat_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

dog_images = [
    f for f in os.listdir(dog_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_paths = []

if cat_images:
    image_paths.append(os.path.join(cat_folder, cat_images[0]))

if dog_images:
    image_paths.append(os.path.join(dog_folder, dog_images[0]))

print("Images selected:", len(image_paths))

for image_path in image_paths:
    results = model(image_path)

    result = results[0]
    output = result.plot()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 5))
    plt.imshow(output)
    plt.title(os.path.basename(image_path))
    plt.axis("off")
    plt.show()

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            "Object:", model.names[class_id],
            "Confidence:", round(confidence, 2)
        )
