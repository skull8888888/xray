import numpy as np

def iou_score(a, b):
    
    w_1 = a[2]-a[0]
    h_1 = a[3]-a[1]

    w_2 = b[2]-b[0]
    h_2 = b[3]-b[1]
    
    a_1 = w_1*h_1
    a_2 = w_2*h_2
    
    xx1 = np.maximum(a[0], b[0])
    yy1 = np.maximum(a[1], b[1])
    xx2 = np.minimum(a[2], b[2])
    yy2 = np.minimum(a[3], b[3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / (a_1 + a_2 - inter)

    return ovr

def from_mm_format(results):
    
    boxes = []
    scores = []
    labels = []
    
    for i, pred in enumerate(results):
        if len(pred):
            for p in pred:
                
                score = p[4].astype(float)
                
                box = p[:4]
                boxes.append(box)
                scores.append(score)
                labels.append(i)

    return boxes, scores, labels

def to_mm_format(boxes, scores, labels):
    
    classes = [[] for _ in range(15)]
    
    for i in range(len(boxes)):
        
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        
        classes[label].append(np.concatenate([box, [score]]).tolist())
    
    classes = list(map(lambda x: np.array(x), classes))
    
    return classes

def cluster_boxes(results, iou_thr=0.9):
    
    boxes, scores, labels = from_mm_format(results)
    
    overlaps = np.zeros((len(boxes), len(boxes)))
    
    for i in range(len(boxes)):

        label = labels[i]
        bbox = boxes[i]

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        box_overlaps = []

        for j in range(len(boxes)):

            if i != j:
                iou = iou_score(bbox, boxes[j])

                if iou >= iou_thr:

                    overlaps[i,j] = 1

    overlaps = np.triu(overlaps) 
    checked = {}
    clean = {}
    
    for i in range(len(overlaps)):

        if i in checked:
            continue 

        row = overlaps[i].flatten()
        overlapping_ids = np.argwhere(row == 1).flatten().tolist()

        if len(overlapping_ids) > 0:

            overlapping_ids.append(i)
            clean[i] = overlapping_ids

            for box_idx in overlapping_ids:
                checked[box_idx] = 1
        else:
            clean[i] = [i]
            checked[i] = 1
    
    new_boxes = []
    new_labels = []
    new_scores = []
    
    
    for k, v in clean.items():
        
        if len(v) > 1:
        
            avg_box = []

            classes = {}

            for box_id in v:
                
                if labels[box_id] != 9:
                
                    if labels[box_id] in classes:
                        classes[labels[box_id]] += scores[box_id]
                    else:
                        classes[labels[box_id]] = scores[box_id]
                
#                 new_boxes.append(boxes[box_id].astype(int))
#                 new_labels.append(int(labels[box_id]))
#                 new_scores.append(scores[box_id])
                
                avg_box.append(boxes[box_id])

            avg_box = np.array(avg_box).mean(0).astype(int)
            
            for class_id, score in classes.items():
            
                new_boxes.append(avg_box.flatten())
                new_labels.append(int(class_id))
                new_scores.append(np.minimum(score, 1.0))
                
        elif len(v) == 1:

            new_boxes.append(boxes[k].astype(int))
            new_labels.append(int(labels[k]))
            new_scores.append(scores[k])

        
    res = to_mm_format(new_boxes, new_scores, new_labels)
    return res