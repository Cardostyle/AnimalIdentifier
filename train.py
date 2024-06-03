from ultralytics import YOLO

# Laden des vortrainierten Modells
model = YOLO('yolov5s.pt')

# Training starten
results = model.train(data='data.yaml', epochs=2, imgsz=640)

# Optionale: Auswertung des Modells nach dem Training
metrics = model.val()

# Speichern des trainierten Modells
model.save('models/best.pt')
