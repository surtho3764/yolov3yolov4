import sys
import os
path = os.getcwd().split("/")[:-1]
path = "/".join(path)
sys.path.append(path)

import torchvision.transforms as transforms

from lib.utils.parse_config import parse_data_config
from lib.utils.utils import load_classes
from lib.utils.dataset import create_dataset
from lib.utils.dataloader import create_dataloader

from lib.utils.augmentations import AbsoluteLabels
from lib.utils.augmentations import PadSquare
from lib.utils.augmentations import RelativeLabels
from lib.utils.augmentations import ToTensor
from lib.evaluate.evaluate import evaluate_model_file

## 建立transforms，之後建立驗證資料集dataset時，需要做的檔案轉換設定DEFAULT_TRANSFORMS
# 驗證集的資料，不做資料擴增處理
DEFAULT_TRANSFORMS = transforms.Compose([
    # 原始資料的邊框值為相對於邊長1的相對邊框值，換算回絕對邊框值
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])


import argparse
parser = argparse.ArgumentParser(description="Evaluate validation data.")
parser.add_argument("-m", "--model", type=str, default="../config/yolov3.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-w", "--weights", type=str, default="../weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
parser.add_argument("-d", "--data", type=str, default="../config/coco.data", help="Path to data config file (.data)")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")

args = parser.parse_args()

print(f"Command line arguments: {args}")










def run_model_evaluate():
    # Load configuration from data file


    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names

    # - ########################################
    # Create Dataloader
    # - ########################################

    # create validation  dataset
    validation_dataset = create_dataset(
        img_path=valid_path,
        transforms=DEFAULT_TRANSFORMS,
        img_size = args.img_size)

    # Load validation dataloader
    validation_dataloader = create_dataloader(validation_dataset,
                                              batch_size= args.batch_size,
                                              num_workers= args.n_cpu,
                                              shuffle=False)

    # - ########################################
    # 計算precision,recall,AP,f1,ap_class
    # - ########################################

    # 計算precision,recall,AP,f1,ap_class
    precision, recall, AP, f1, ap_class = evaluate_model_file(
        dataloader = validation_dataloader,
        model_path = args.model,
        weights_path = args.weights,
        img_path = valid_path,
        class_names = class_names,
        batch_size = args.batch_size,
        img_size = args.img_size,
        n_cpu = args.n_cpu,
        iou_thres = args.iou_thres,
        conf_thres = args.conf_thres,
        nms_thres = args.nms_thres,
        verbose = True)


if __name__ == "__main__":
    run_model_evaluate()