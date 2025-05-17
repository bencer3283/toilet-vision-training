from ultralytics import YOLO

toilet_model = YOLO('/Users/poshengcheng/Documents/Spring 25/Grad Studio/toilet-vision-training/runs/detect/single-led/weights/best.pt', task='detect')
toilet_model.train(data='/Users/poshengcheng/Documents/Spring 25/Grad Studio/toilet-vision-training/dataset-neopixel/data.yaml', epochs=180, device='mps')

# 1. Data quality:
#     - Ensure all images have proper labels
#     - Check that empty label files are intentional (no seams in those images)
#     - Consider more varied lighting conditions
#   2. Data augmentation:
#     - You're already using Gaussian blur and salt/pepper noise
#     - Add rotation, brightness adjustments, and perspective transforms
#     - Implement random cropping to improve detection at different scales
#   3. Training parameters:
#     - Increase epochs beyond 200 if validation loss continues improving
#     - Try different model sizes (yolov8s.pt or yolov8m.pt) for better accuracy
#     - Use early stopping with patience to prevent overfitting
#   4. Labeling improvements:
#     - Label seams consistently with tight bounding boxes
#     - Consider adding more data with varied seam appearances
#     - Ensure labels capture different angles and positions of seams
#   5. Post-processing:
#     - Experiment with confidence thresholds during inference
#     - Consider using test-time augmentation for improved detection