import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO


def intersection_over_union(results_table,IoU_threshold):
    number_of_obj = results_table.shape[0]
    IoU_list = []
    box_to_drop = set()
    for i in range(number_of_obj-1):
        for j in range(i + 1, number_of_obj):
            if i == j:
                continue
            # Box i (first box)
            x1_1, y1_1 = results_table['x1'][i], results_table['y1'][i]
            x1_2, y1_2 = results_table['x2'][i], results_table['y2'][i]
            # Box j (second box)
            x2_1, y2_1 = results_table['x1'][j], results_table['y1'][j]
            x2_2, y2_2 = results_table['x2'][j], results_table['y2'][j]
            inter_width = max(0, min(x1_2, x2_2) - max(x1_1, x2_1))
            inter_height = max(0, min(y1_2, y2_2) - max(y1_1, y2_1))
            intersection = inter_width * inter_height

            area_box1 = (x1_2 - x1_1) * (y1_2 - y1_1)
            area_box2 = (x2_2 - x2_1) * (y2_2 - y2_1)
            union = area_box1 + area_box2 - intersection

            IoU = intersection / union if union != 0 else 0
            if IoU > IoU_threshold:
                print(f'Detected two overlapping bounding boxes at the threshold of {IoU_threshold}')
                if results_table['confidence'][i] < results_table['confidence'][j]:
                    box_to_drop.add(i)
                else:
                    box_to_drop.add(j)
            IoU_list.append({
                'Pair' : f'IoU_box{i}_box{j}',
                'IoU' : IoU,
                'conf_box1': results_table['confidence'][i],
                'conf_box2': results_table['confidence'][j]
                })


    df_IoU = pd.DataFrame(IoU_list)
    results_table = results_table.drop(index=results_table.index[list(box_to_drop)]).reset_index(drop=True)
    return results_table, df_IoU





def model_test(image_name,IoU_threshold, model):
    image = cv2.imread(image_name)

    results_list = []
    test_res = model(image_name, conf = 0.6)
    bb = test_res[0].boxes

    # Check if any bounding boxes are detected
    if bb is None or len(bb) == 0:
        print("No bounding boxes detected.")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("No bounding boxes detected")
        plt.axis('off')
        plt.show()
        return "No bounding boxes", None, None, pd.DataFrame(),pd.DataFrame()

    # Proceed if bounding boxes are present
    for box, cls, conf in zip(bb.xyxy, bb.cls, bb.conf):
        results_list.append({
            'filename': image_name,
            'x1': box[0].item(),
            'y1': box[1].item(),
            'x2': box[2].item(),
            'y2': box[3].item(),
            'confidence': conf.item(),
            'class': cls.item()
        })

    results_table = pd.DataFrame(results_list)
    results_table, df_IoU = intersection_over_union(results_table,IoU_threshold)
    
    for i in results_table.index:
        cv2.rectangle(image, (int(results_table['x1'][i]), int(results_table['y1'][i])),
                      (int(results_table['x2'][i]), int(results_table['y2'][i])), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Counting marks
    idx = results_table.index
    total_marks = np.sum(results_table['class'] == 0)
    num_marks_left = np.sum(results_table['x2'] < 1600)
    num_marks_right = np.sum(results_table['x1'] > 1900)
    return total_marks, num_marks_left, num_marks_right, results_table, df_IoU

