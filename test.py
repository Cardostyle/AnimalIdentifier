import os
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
    labels = [[int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
    return labels

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def test_model(model, test_images_dir, test_labels_dir, iou_threshold=0.5):
    total_predictions = 0
    correct_predictions = 0

    for filename in tqdm(os.listdir(test_images_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(test_images_dir, filename)
            label_path = os.path.join(test_labels_dir, os.path.splitext(filename)[0] + '.txt')

            img = Image.open(image_path)
            results = model(img)

            if isinstance(results, list):
                results = results[0]

            df = results.pandas().xyxy[0]
            gt_labels = load_labels(label_path)

            for _, row in df.iterrows():
                pred_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                pred_class = int(row['class'])
                pred_conf = row['confidence']

                for gt in gt_labels:
                    gt_class, x_center, y_center, width, height = gt
                    x1 = (x_center - width / 2) * img.width
                    y1 = (y_center - height / 2) * img.height
                    x2 = (x_center + width / 2) * img.width
                    y2 = (y_center + height / 2) * img.height
                    gt_box = [x1, y1, x2, y2]

                    iou = calculate_iou(pred_box, gt_box)
                    if iou > iou_threshold and pred_class == gt_class:
                        correct_predictions += 1
                        break
                total_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    return accuracy

# Laden des trainierten Modells
model = YOLO('models/best.pt')

# Verzeichnisse für Testbilder und Labels
test_images_dir = 'datasets/data/test/images'
test_labels_dir = 'test_dataset/labels'

# Modell testen und Genauigkeit berechnen
accuracy = test_model(model, test_images_dir, test_labels_dir)
print(f'Modellgenauigkeit: {accuracy:.2f}%')
