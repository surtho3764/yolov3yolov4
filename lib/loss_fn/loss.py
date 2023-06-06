

import math
import torch
import torch.nn as nn


from lib.loss_fn.utils import build_targets
from lib.utils.utils import to_cpu






# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py
def bbox_iou_loss(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    '''
    參數：box1, box2
    box1：單位要和box2相反，例如有個box1和box2要計算iou時，加入
    正常box1為tensor([[0.5285, 0.5828, 0.6858],
                    [0.5023, 0.7814, 0.5098],
                    [2.7856, 1.4156, 0.7774],
                    [0.4567, 0.8499, 1.5675]]) shape=(4,3)
       box2為tensor([[0.5310, 0.5310, 0.5310],
                    [0.3840, 0.3840, 0.3840],
                    [8.9380, 8.9380, 8.9380],
                    [7.7846, 7.7846, 7.7846]])shape=(4,3)
    才可以計算，但是程式碼中box2 = box2.T，
    所以要將box2還為tensor([[0.5310, 0.3840, 8.9380, 7.7846],
                       [0.5310, 0.3840, 8.9380, 7.7846],
                       [0.5310, 0.3840, 8.9380, 7.7846]]) shape(3,4)
    之後經過計box2 = box2.T變成shape=(4,3)，才可以和box1計算iou

    參數：x1y1x2y2：
    x1y1x2y2=True：表示box1的單位是x1y1x2y2，就不需要轉換
    x1y1x2y2=False：表示box1的單位是xywh，需要先轉換成x1y1x2y2
    '''
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    # print("bbox_iou_loss")
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


# compute_loss
def compute_loss(predictions, targets, model):
    # Check which device was used
    device = targets.device
    # print("predictions:",len(predictions))
    # print("targets:",targets)

    # Add placeholder varables for the different losses
    # 存放三種不同loss的placeholder varables
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Build yolo targets
    # 整理tagets來得到yolo模型格式需要的taget
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # Define different loss functions classification
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # Calculate losses for each yolo layer
    for layer_index, layer_predictions in enumerate(predictions):
        # 第i組特徵值當作預測值
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]

        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]
        # Check if there are targets for this batch
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            # 取得第i個特徵值中要真正要用來預測bbox的預測值，之後要和label值計算loss
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            # 因為模型預測出來的值為tx,ty,tw,th，所以需要額外計算，才會是預測邊框值
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            # 預測出來的bbox
            pbox = torch.cat((pxy, pwh), 1)

            # 計算bbox的loss
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            # pbox.T:預測的bbox, tbox:label的bbox
            iou = bbox_iou_loss(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)

            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            # 原本的loss要極大化iou，但是我們利用梯度下降，要改為極小化iou，所以就變成1-iou
            lbox += (1.0 - iou).mean()  # iou loss

            # 計算類別預測loss
            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(
                tobj.dtype)  # Use cells with iou > 0 as object targets

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                # label值中到class id 轉為(one-hot值)所以維度為共(3,num_classes)
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1

                # Use the tensor to calculate the BCE loss
                # 預測類別的值ps[:, 5:]和label值t來計算loss
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        # 計算置信度的loss
        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(layer_predictions[..., 4], tobj)  # obj loss

    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    # lbox:計算bbox的loss
    # lobj:計算置信度的loss
    # lcls:計算class的loss
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))




