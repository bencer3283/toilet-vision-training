from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train10/weights/best.pt')  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model('img/*.jpg')  # return a list of Results objects

# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="/Users/poshengcheng/Documents/Spring 25/Grad Studio/toilet-vision-training/img/predicted/{i}.jpg")  # save to disk