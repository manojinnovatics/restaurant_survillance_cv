import numpy as np 
import cv2
from scipy.spatial import distance
import csv
import os
from ultralytics import YOLO

import tensorflow as tf
from keras.api.layers import Conv1D, MaxPooling1D, Flatten, Input, Dense,concatenate
from keras.api.models import Model

def save_vectors_to_csv(vectors, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for frame_vector in vectors:
            for vector in frame_vector:
                writer.writerow(vector)
                
def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    return Model(inputs=inputs, outputs=x)

Dish_model = create_cnn_model((15*10,6))
Dish_model.summary()
Menu_Card_model = create_cnn_model((15*10,6))
Menu_Card_model.summary()
Mobile_model = create_cnn_model((15*5,6))
Mobile_model.summary()

combined_input = concatenate([Dish_model.output, Menu_Card_model.output, Mobile_model.output])

x = Dense(64, activation='relu')(combined_input)
activity_output = Dense(1, activation='sigmoid', name='activity_output')(x)
table_output = Dense(7, activation='sigmoid', name='table_output')(x)

final_model = Model(inputs=[Dish_model.input, Menu_Card_model.input, Mobile_model.input], outputs=[activity_output, table_output])
final_model.compile(optimizer='adam', loss={'activity_output': 'binary_crossentropy', 'table_output': 'binary_crossentropy'})
final_model.summary()


model_dict = {
    'Dish': Dish_model,
    'Menu Card': Menu_Card_model,
    'Mobile': Mobile_model
}

def process_bbox_list_to_tensor(bbox_list, max_size):
    tensor = np.zeros((max_size, 6), dtype=np.float32)
    for i, bbox in enumerate(bbox_list[:max_size]):
        x1, y1, x2, y2,conf,tid = bbox
        tensor[i] = [x1, y1, x2, y2,conf,tid] 
    return tensor

# def tensor_model(flatten_model,tensor,file_name):
#     tensor = np.expand_dims(tensor, axis=-1)  # Add channel dimension
#     tensor = np.expand_dims(tensor, axis=0)   # Add batch dimension
#     flatten_output = flatten_model.predict(tensor)
#     print("Flatten layer output shape:", flatten_output.shape)
#     np.savetxt(f'flatten_output_{file_name}.txt', flatten_output)

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
    
dish_vectors = []
menu_card_vectors = []
ordering_device_vectors = []
class_names = ['Dish', 'Menu Card', 'Mobile']

fps = 5
batch_size = 15
frame_count = 0
frame_index = 0

vid_path = "/mnt/c/Users/pc/Downloads/Ordering_Table_2.mp4"
cap = cv2.VideoCapture(vid_path)
fps = round(cap.get(cv2.CAP_PROP_FPS))
interval = int(fps / 5) 

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
                
        for class_name, vectors in frame_vectors.items():
            for vector in vectors:
                if vector[4] > 0:  # If the confidence score is greater than 0, draw the bounding box
                    x1, y1, x2, y2 = map(int, vector[:4])
                    label = f'{class_name}: {vector[4]:.2f}'
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(img, f'Table-{int(vector[5])}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('detection', img)
    frame_index += 1
    # Batch Processing of frames
    if frame_count == batch_size:
        print('Batch processed',"frame_count",frame_index)
        frame_filename = os.path.join("/home/manojinnovatics/work/projects/restaurant_cv/output_clips", f"frame_{frame_index}.jpg")
        cv2.imwrite(frame_filename, img)
        frame_count=0
        
        # for class_name in class_names:
        #     tensor = process_bbox_list_to_tensor(frame_vectors[class_name], max_lengths[class_name])
            # model_name = class_name.replace(" ","_") + '_model'
            # tensor_model(model_dict[class_name],tensor,class_name+str(frame_index))
        # dish_vectors = process_bbox_list_to_tensor(dish_vectors, max_lengths['Dish'])
        # menu_card_vectors = process_bbox_list_to_tensor(menu_card_vectors, max_lengths['Menu Card'])
        # ordering_device_vectors = process_bbox_list_to_tensor(ordering_device_vectors, max_lengths['Mobile'])
        print("len(menu_card_vectors)",len(menu_card_vectors))
        activity_pred, table_activity_pred = final_model.predict([np.expand_dims(dish_vectors, axis=0), 
                                                                np.expand_dims(menu_card_vectors, axis=0), 
                                                                np.expand_dims(ordering_device_vectors, axis=0)])

        # Output results
        print(f"Activity Prediction: {activity_pred}")
        activity_happening = int(activity_pred > 0.5)
        table_activities = [int(x > 0.5) for x in table_activity_pred[0]]

        print(f"Activity happening: {activity_happening}")
        print(f"Table-wise activities: {table_activities}")
        
        # save_vectors_to_csv(dish_vectors, 'dish_vectors.csv')
        # save_vectors_to_csv(menu_card_vectors, 'menu_card_vectors.csv')
        # save_vectors_to_csv(ordering_device_vectors, 'ordering_device_vectors.csv')
        dish_vectors.clear()
        menu_card_vectors.clear()
        ordering_device_vectors.clear()
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()