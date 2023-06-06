import torch




# build_targets (整理label)函數
'''
整理label
# 處理targets，使之後可以看prediction計算loss，
回傳值
tcls:list大小為3- 每一個維度，對應一組特徵圖的label值 class for each target
tbox:list大小為3- 每一個維度，對應一組特徵圖的label值 邊框值target box from global grid coordinates to local offsets (tbox)
indices：list大小為3-指標順序 index list and limit index range to prevent out of bounds  indices
anch：list大小為3- 每一個維度，對應一組特徵圖的label值 correct anchor for each target 候選框
'''


def build_targets(p, targets, model):
    '''
    p:predictions,
    targets:targets,
    model:model
    '''
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    # na:number of anchors, targets
    # nt:TODO label的邊框數
    na, nt = 3, targets.shape[0]  # number of anchors, targets #TODO
    # 分別儲存經過處理過後的
    # tcls:class id
    # tbox:bbox
    # index:featureid
    # anchor:anchor box
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain

    # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    # print("build_targets:",targets)

    for i, yolo_layer in enumerate(model.yolo_layers):
        # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
        # print("yolo_layer:",yolo_layer)
        # print("yolo_layer.原始候選框anchors",yolo_layer.anchors)
        # print("下採樣倍數",yolo_layer.stride)
        # 得到預先設定的anchors box值
        anchors = yolo_layer.anchors / yolo_layer.stride
        # print("和特徵值同單位大小的anchors ",anchors)

        # 計算預測出來的值，在特徵值單位下的尺度大小
        # Add the number of yolo cells in this layer the gain tensor
        # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)
        # print("p[i].shape",p[i].shape)
        # print(torch.tensor(p[i].shape)[[3, 2, 3, 2]].shape)
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        # print("gain[2:6]",gain[2:6])
        # print("gain",gain.shape)

        # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
        # 計算label值，在特徵值單位下的尺度大小
        t = targets * gain

        # Check if we have targets
        if nt:
            # Calculate ration between anchor and target box for both width and height
            r = t[:, :, 4:6] / anchors[:, None]
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # The anchor id is still saved in the 7th value of each target
            t = t[j]
        else:
            t = targets[0]

        # Extract image id in batch and class id
        b, c = t[:, :2].long().T
        # We isolate the target cell associations.
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        # 得到label的xy
        gxy = t[:, 2:4]
        # 得到label的wh
        gwh = t[:, 4:6]  # grid wh
        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = gxy.long()
        # Isolate x and y index dimensions
        gi, gj = gij.T  # grid xy indices

        # Convert anchor indexes to int
        a = t[:, 6].long()
        # Add target tensors for this yolo layer to the output lists
        # Add to index list and limit index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
        # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
        # 需要將原本label值中的xy值，轉換成得到相對於網格單位下的xy值 ，所以需要計算gxy - gij。
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # Add correct anchor for each target to the list
        anch.append(anchors[a])
        # Add class for each target to the list
        tcls.append(c)
    # print("class for each target (tcls) ",tcls[0].shape)
    # print("class for each target (tcls) ",tcls[0])
    # print("target box from global grid coordinates to local offsets (tbox)",tbox[0].shape)
    # print("target box from global grid coordinates to local offsets (tbox)",tbox[0])
    # print("index list and limit index range to prevent out of bounds  indices",indices[0])
    # print("correct anchor for each target (anch)",anch[0])
    return tcls, tbox, indices, anch


