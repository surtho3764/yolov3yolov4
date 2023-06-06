import sys
import os
path = os.getcwd().split("/")[:-1]
path = "/".join(path)
sys.path.append(path)

from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from lib.utils.utils import load_classes
from lib.utils.augmentations import AbsoluteLabels
from lib.utils.augmentations import PadSquare
from lib.utils.augmentations import RelativeLabels
from lib.utils.augmentations import ToTensor
from lib.utils.augmentations import Resize
from lib.utils.dataset import ImageFolder
from lib.evaluate.detection import detect_directory





import argparse
parser = argparse.ArgumentParser(description="Detect objects on images.")
parser.add_argument("-m", "--model", type=str, default="../config/yolov3.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-w", "--weights", type=str, default="../weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
parser.add_argument("-i", "--images", type=str, default="../data/samples", help="Path to directory with images to inference")
parser.add_argument("-c", "--classes", type=str, default="../data/coco.names", help="Path to classes label file (.names)")
parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")

args = parser.parse_args()

print(f"Command line arguments: {args}")


## 建立transforms，之後建立驗證資料集dataset時，需要做的檔案轉換設定DEFAULT_TRANSFORMS
# 驗證集的資料，不做資料擴增處理
DEFAULT_TRANSFORMS = transforms.Compose([
    # 原始資料的邊框值為相對於邊長1的相對邊框值，換算回絕對邊框值
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])


## 建立transform，之後建立detect dataset時，需要做的檔案轉換設定DETECT_TRANSFORMS
# detect集的資料，不做資料擴增處理，要resize為模型輸入的圖片大小args.img_size=416
# DEFAULT_TRANSFORMS為驗證集建立的transform
DETECT_TRANSFORMS = transforms.Compose([
    DEFAULT_TRANSFORMS,
    Resize(args.img_size)])


def run_detect():
    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    # - ########################################
    # load detect Dataloader
    # - ########################################
    # create detect ImageFolder dataset
    dataset = ImageFolder(
        folder_path=args.images,
        transform=DETECT_TRANSFORMS)

    # load detect dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        pin_memory=True)

    # - ###############################################################################
    # - detect
    # - ###############################################################################
    detect_directory(
        detect_dataloader=dataloader,
        model_path=args.model,
        weights_path=args.weights,
        img_path=args.images,
        classes=classes,
        output_path=args.output,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres)

if __name__ == '__main__':
    run_detect()