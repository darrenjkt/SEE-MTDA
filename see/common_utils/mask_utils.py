import numpy as np
import cv2

def xyxy2xywh(bbox):
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.
    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.
    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]

def segm2json(result, score_thresh=0.5):
    """
    Filter out detections that are less than score_thresh, then
    convert to a more easily accessible dictionary format.
    
    :result: The output of the detector from mmdetection.
    :score_thresh: Set the minimum confidence level for detections to keep
    """
    if isinstance(result, tuple):
        det, seg = result
        if isinstance(seg, tuple):
            seg = seg[0]  # ms rcnn

    # Keep only the results with > 0.5 confidence
    class_ids = [0,1,2]
    for c_ids in class_ids:
        if det[c_ids].shape[0] != 0:
            scores = det[c_ids][:, -1]
            inds = scores > 0.5
            det[c_ids] = det[c_ids][inds, :]
            seg[c_ids] = np.array(seg[c_ids])[inds, :]
    
    segm_json_results = []
    for label_id in range(len(det)):
        # Get number of bboxes for the given class label_id
        bboxes = det[label_id]

        # Get confidence score of the predicted class
        if isinstance(seg, tuple):
            segms = seg[0][label_id]
            mask_score = seg[1][label_id]
        else:
            segms = seg[label_id]
            mask_score = [bbox[4] for bbox in bboxes]

        for i in range(bboxes.shape[0]):
            data = dict()
            if label_id <= 2:      # 0:person, 1:bicycle, 2:car       
                data['category_id'] = label_id           
            else:
                continue

            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(mask_score[i])
            data['segmentation'] = segms[i]
            segm_json_results.append(data)
            
    return segm_json_results

def mask2polygon(seg_mask):
    """
    Generates polygon representation of mask where each (x,y) pair are the 
    vertices of a polygon which can be connected to form the image boundary.
    If the mask is split in two because of occlusion, this will return two 
    polygons, and likewise if it is split into greater parts.
    
    # Segmentation polygon Format
    [
        [x1, y1, x2, y2, x3, y3,...],
        [x1, y1, x2, y2, x3, y3,...],
        ...
    ]
    :param seg_mask (bool array): Boolean array showing where the instance is
    :returns: Polygons representation
     
    """
    mask = seg_mask.astype(np.uint8)
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))    
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    polygons = [point.ravel().tolist() for point in polygons]

    # If there's less than 3 pixels to outline the polygon, it's a line so we filter it
    polygons = [polygon for polygon in polygons if (len(polygon) >= 6)]
    return polygons
