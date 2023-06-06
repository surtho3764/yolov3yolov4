# Yolov3, Yolov4 Implmentation
This repository is created for implmentation of yolov3 and yolov4 with support for training, inference and evaluation.it is modified from YOLOv3 (https://github.com/ultralytics/yolov3, https://github.com/ultralytics/yolov3, https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master).

## Requirements
Python 3.8 or later with all of the pip install -U -r requirements.txt packages including:
torch >= 1.3
opencv-python
Pillow....

```bash
git clone git@github.com:surtho3764/yolov3yolov4.git
cd src
```

### Download pretrained weights
```bash
cd weight 
./download_pretrained_weights.sh
```
### Download test coco data
```bash
cd data
./get_coco_dataset.sh
```




## Train your own data or coco, voc data as follows:

```bash
# run yolov3
python train.py --model ../config/yolov3.cfg  --pretrained_weights ../weights/darknet53.conv.74 
# run yolov4
python train.py --model ../config/yolov4/yolov4.cfg  --pretrained_weights ../weights/yolov4.weights
```


## Testing
```bash
# run yolov3
python test.py --model ../config/yolov3.cfg  --weights ../weights/yolov3.weights

# run yolov4
python test.py --model ../config/yolov4/yolov3.cfg  --weights ../weights/yolov4.weights
```

## Inference
```bash
# yolov3
python detect.py --model ../config/yolov3.cfg  --weights ../weights/yolov3.weights

# run yolov4
python detect.py --model ../config/yolov3.cfg  --weights ../weights/yolov4.weights
```
