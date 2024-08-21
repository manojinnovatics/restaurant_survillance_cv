import numpy as np
import cv2
from scipy.spatial import distance
import os
from ultralytics import YOLO
import tensorflow as tf
from keras.api.layers import Conv1D, MaxPooling1D, Flatten, Input
from keras.api.models import Model

def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    return Model(inputs=inputs, outputs=x)

# Create CNN models for each class
Dish_model = create_cnn_model((15*10,6))
Dish_model.summary()
Menu_Card_model = create_cnn_model((15*10,6))
Menu_Card_model.summary()
Mobile_model = create_cnn_model((15*5,6))
Mobile_model.summary()

model_dict = {
    'Dish': Dish_model,
    'Menu Card': Menu_Card_model,
    'Mobile': Mobile_model
}

dish_vectors = []
menu_card_vectors = []
ordering_device_vectors = []
class_names = ['Dish', 'Menu Card', 'Mobile']

model = YOLO('/home/manojinnovatics/work/projects/restaurant_cv/yolo8_custom/models/last_2.pt')

tables = [
    {'id': 1, 'x1': 47, 'y1': 322, 'x2': 367, 'y2': 479},
    {'id': 2, 'x1': 275, 'y1': 254, 'x2': 509, 'y2': 437},
    {'id': 3, 'x1': 433, 'y1': 206, 'x2': 583, 'y2': 334},
    {'id': 4, 'x1': 523, 'y1': 183, 'x2': 626, 'y2': 265},
    {'id': 5, 'x1': 9, 'y1': 135, 'x2': 261, 'y2': 319},
    {'id': 6, 'x1': 215, 'y1': 104, 'x2': 428, 'y2': 212},
    {'id': 7, 'x1': 398, 'y1': 83, 'x2': 532, 'y2': 155},
]

for table in tables:
    table['center_x'] = (table['x1'] + table['x2']) // 2
    table['center_y'] = (table['y1'] + table['y2']) // 2

max_lengths = {
    'Dish': 10,
    'Menu Card': 10,
    'Mobile': 5
}

# Initialize lists to store features and labels
X_dish = []
X_menu_card = []
X_mobile = []
y_activity = []
y_table_activity = []

# Load existing data if the file exists
dataset_path = "dataset.npz"
if os.path.exists(dataset_path):
    data = np.load(dataset_path)
    X_combined = list(data['X_combined'])
    y_activity = list(data['y_activity'])
    y_table_activity = list(data['y_table_activity'])
    
X_combined = []  # Final dataset to store combined features
y_activity = []
y_table_activity = []

# Paths to multiple videos
video_paths = [
    "/mnt/c/Users/pc/Downloads/Ordering_Table_2.mp4",
]

fps = 5
batch_size = 15
frame_count = 0
frame_index = 0

for vid_path in video_paths:
    cap = cv2.VideoCapture(vid_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    interval = int(fps / 5)  # Process frames at 5 fps

    while cap.isOpened():
        success, img = cap.read()
        
        if not success or img is None:
            break
        
        if frame_index % interval == 0:
            frame_count += 1
            frame_vectors = {
                'Dish': np.zeros((max_lengths['Dish'], 6)),  # [x1, y1, x2, y2, confidence, nearest_table]
                'Menu Card': np.zeros((max_lengths['Menu Card'], 6)),
                'Mobile': np.zeros((max_lengths['Mobile'], 6))
            }
            
            results = model(img)
            
            counters = {name: 0 for name in class_names}

            for result in results:
                boxes = result.boxes.xyxy.numpy() 
                classes = result.boxes.cls.numpy() 
                scores = result.boxes.conf.numpy() 

                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = box
                    class_name = model.names[int(cls)]
                    
                    if class_name in class_names and counters[class_name] < max_lengths[class_name]:
                        obj_center_x = (x1 + x2) / 2
                        obj_center_y = (y1 + y2) / 2
                        
                        nearest_table = min(tables, key=lambda t: distance.euclidean((obj_center_x, obj_center_y), (t['center_x'], t['center_y'])))
                        
                        frame_vectors[class_name][counters[class_name]] = [
                                x1, y1, x2, y2,
                                score,
                                int(nearest_table['id'])  
                            ]
                        counters[class_name] += 1
                        # print(f"Feature Vector: {frame_vectors}")  
                        
                
            dish_vectors.extend(frame_vectors['Dish'])
            menu_card_vectors.extend(frame_vectors['Menu Card'])
            ordering_device_vectors.extend(frame_vectors['Mobile'])
                    
            # for class_name, vectors in frame_vectors.items():
            #     for vector in vectors:
            #         if vector[4] > 0:  # If the confidence score is greater than 0, draw the bounding box
            #             x1, y1, x2, y2 = map(int, vector[:4])
            #             label = f'{class_name}: {vector[4]:.2f}'
            #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #             cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #             cv2.putText(img, f'Table-{int(vector[5])}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.imshow('detection', img)
        frame_index += 1
        # Batch Processing of frames
        if frame_count == batch_size:
            print('Batch processed',"frame_count",frame_index)
            # frame_filename = os.path.join("/home/manojinnovatics/work/projects/restaurant_cv/output_clips", f"frame_{frame_index}.jpg")
            # cv2.imwrite(frame_filename, img)
            frame_count=0
            dish_features = model_dict['Dish'].predict(np.expand_dims(dish_vectors, axis=0))
            menu_card_features = model_dict['Menu Card'].predict(np.expand_dims(menu_card_vectors, axis=0))
            mobile_features = model_dict['Mobile'].predict(np.expand_dims(ordering_device_vectors, axis=0))
            print("menu_card_features",menu_card_features)
            combined_features = np.concatenate([dish_features.flatten(), menu_card_features.flatten(), mobile_features.flatten()]) #5920 features
            # print("combined_features--",combined_features)
            print("len(combined_features)",len(combined_features))
            X_combined.append(combined_features)
            y_activity.append(1)
            y_table_activity.append([0, 0, 0, 1, 0, 0, 0])
            
            dish_vectors.clear()
            menu_card_vectors.clear()
            ordering_device_vectors.clear()
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()

print("len(X_combined)",len(X_combined))
# Save the dataset
np.savez("dataset.npz", X_combined=X_combined, y_activity=y_activity, y_table_activity=y_table_activity)

cv2.destroyAllWindows()
