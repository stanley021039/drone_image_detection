import os
import cv2
import numpy
import torch
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_images_with_bboxes(image_folder, save_dir):
    for file in os.listdir(image_folder):
        if 'png' in file:
            image_path = os.path.join(image_folder, file)
            image = cv2.imread(image_path)
            bboxes = []
            with open(image_path.replace('png', 'txt')) as f:
                for line in f.readlines():
                    bbox = [int(x) for x in line.replace('\n', '').split(',')]
                    bboxes.append(bbox)
            image = plot_bboxes_on_images(image, bboxes)
            cv2.imwrite(os.path.join(save_dir, file.replace('.png', 'withbboxes.png')), image)


def plot_bboxes_on_images(image, bboxes: list, box_color=None):
    if box_color is None:
        box_color = {0: (220, 0, 0), 1: (0, 220, 0), 2: (0, 0, 220), 3: (0, 220, 220), 4: (220, 0, 0)}
    for bbox in bboxes:
        cls, x0, y0, w, h = bbox
        print(box_color[cls])
        cv2.rectangle(image, (x0, y0), (x0+w, y0+h), box_color[cls], 2)
    return image


def intersection_over_union(boxes_preds, boxes_labels):
    #boxes_preds shapes is (4) where N is number of bboxes
    #boxes_labels shape is (4)
    box1_x1 = boxes_preds[0]
    box1_y1 = boxes_preds[1]
    box1_x2 = boxes_preds[2]
    box1_y2 = boxes_preds[3]

    box2_x1 = boxes_labels[0]
    box2_y1 = boxes_labels[1]
    box2_x2 = boxes_labels[2]
    box2_y2 = boxes_labels[3]


    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2,box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    #.clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, num_classes=1
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        #detections存取在該類別別中的所有box
        for pred_box in pred_boxes:
            for detection in pred_box:
                if detection[1] == c:
                    detections.append(detection)
        #ground_truth存取該類別中所有box
        for i, true_boxs in enumerate(true_boxes):
            for true_box in true_boxs:
                if true_box[1] == c:
                    ground_truths.append(np.insert(true_box, 0, i))


        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        #針對某一個類別去看總共有幾個box
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            #取得在同一張照片中相同類別的座標值
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / (len(average_precisions)+1e-15)


def nms(idx, bboxes, scores, iou_threshold, threshold):
    #prediction = [[class,probality,x1,y1,x2,y2]]
    bboxes_after_thre = [np.append(score, bboxes[i]) for i, score in enumerate(scores) if score > threshold]
    '''bboxes_after_thre = np.zeros(5)
    for i, score in enumerate(scores):
        if score > threshold:
            bboxe = np.append(score, bboxes[i])
            bboxes_after_thre = np.vstack([bboxes_after_thre, bboxe])
    if len(bboxes_after_thre) > 5:
        bboxes_after_thre = np.delete(bboxes_after_thre, 0, axis=0)
    else:
        bboxes_after_thre = []'''
    bboxes_after_nms = []

    if bboxes_after_thre:
        chosen_box = bboxes_after_thre[0]
        bboxes_after_thre = np.delete(bboxes_after_thre, 0, axis=0)
    while len(bboxes_after_thre) != 0:
        bboxes_after_thre = [
            box
            for box in bboxes_after_thre
            if intersection_over_union(torch.tensor(chosen_box[1:]), torch.tensor(box[1:])) < iou_threshold
            ]
        bboxes_after_nm = np.append(1, chosen_box)
        bboxes_after_nm = np.append(idx, bboxes_after_nm)
        bboxes_after_nms.append(bboxes_after_nm)
        if bboxes_after_thre:
            chosen_box = bboxes_after_thre[0]
            bboxes_after_thre = np.delete(bboxes_after_thre, 0, axis=0)

    return bboxes_after_nms


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 11)
    bboxes1 = predictions[..., 2:6]
    bboxes2 = predictions[..., 7:11]
    scores = torch.cat(
        (predictions[..., 1].unsqueeze(0), predictions[..., 6].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :1].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 1], predictions[..., 6]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# generate_images_with_bboxes('dataset/train', 'dataset/train_with_bboxes')
