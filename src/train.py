import os
import sys
import tqdm
import torch
import torch.optim as optim
from torchsummary import summary


path = os.getcwd().split("/")[:-1]
path = "/".join(path)
sys.path.append(path)

from lib.utils.utils import provide_determinism
from lib.utils.logger import Logger
from lib.utils.parse_config import parse_data_config
from lib.utils.utils import load_classes
from lib.net.net import load_model
from lib.utils.dataset import create_dataset
import torchvision.transforms as transforms

from lib.utils.augmentations import AbsoluteLabels
from lib.utils.augmentations import DefaultAug
from lib.utils.augmentations import PadSquare
from lib.utils.augmentations import RelativeLabels
from lib.utils.augmentations import ToTensor
from lib.utils.dataloader import create_dataloader
from lib.loss_fn.loss import compute_loss
from lib.utils.utils import to_cpu
from lib.evaluate.evaluate import _evaluate

## 建立transforms，之後建立訓練資料集dataset時，需要做的檔案轉換設定AUGMENTATION_TRANSFORMS
AUGMENTATION_TRANSFORMS = transforms.Compose([
    # 原始資料的邊框值為相對於邊長1的相對邊框值，換算回絕對邊框值
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

## 建立transforms，之後建立驗證資料集dataset時，需要做的檔案轉換設定DEFAULT_TRANSFORMS
# 驗證集的資料，不做資料擴增處理
DEFAULT_TRANSFORMS = transforms.Compose([
    # 原始資料的邊框值為相對於邊長1的相對邊框值，換算回絕對邊框值
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

##########################################

import argparse
parser = argparse.ArgumentParser(description="Trains the YOLO model.")
parser.add_argument("-m", "--model", type=str, default="../config/yolov3.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-d", "--data", type=str, default="../config/coco.data", help="Path to data config file (.data)")
parser.add_argument("-e", "--epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
parser.add_argument("--n_cpu", type=int, default=0, help="Number of cpu threads to use during batch generation")
parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")

args = parser.parse_args()

print(f"Command line arguments: {args}")



def run_train():

    # set seed
    provide_determinism()

    # Tensorboard logger，設定logger
    logger = Logger(args.logdir)

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ######################################################
    # Get data configuration
    # ######################################################
    data_config = parse_data_config(args.data)

    # 載入tain和valid路徑
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    # 載入class_names

    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## ######################################################
    # Create model
    # ######################################################
    # args.model:模型的config檔案yolov3.cfg
    print(args.model)
    model = load_model(args.model, args.pretrained_weights)


    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #######################################################
    # Create Dataloader
    # #######################################################
    # create train dataset
    dataset = create_dataset(
        img_path= train_path,
        transforms=AUGMENTATION_TRANSFORMS,
        img_size = model.hyperparams['height'],
        multiscale_training=args.multiscale_training)

    # Load training dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size= mini_batch_size,
        num_workers= args.n_cpu)

    # create validation dataset

    validation_dataset = create_dataset(
        img_path=valid_path,
        transforms=DEFAULT_TRANSFORMS,
        img_size = model.hyperparams['height'],
    )

    # Load validation dataloader
    validation_dataloader = create_dataloader(
        validation_dataset,
        batch_size= mini_batch_size,
        num_workers= args.n_cpu,
        shuffle=False)


    # ######################################################
    # Create optimizer
    # #######################################################
    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")


    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20


    for epoch in range(1, args.epochs +1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode
        pbar = tqdm.tqdm(dataloader ,desc=f"Training Epoch {epoch}")

        for batch_i, (_, imgs, targets) in enumerate(pbar):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)
            # print("第一個預測特徵值 - 第176層大小",outputs[0].shape)
            # print("第二個預測特徵值 - 第200層大小",outputs[1].shape)
            # print("第三個預測特徵值 - 第224層大小",outputs[2].shape)

            loss, loss_components = compute_loss(outputs, targets, model)
            loss.backward()


            # ######################################################
            # Run optimizer
            # #######################################################

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value

                # Log the learning rate
                # write summary tensorboard
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ######################################################
            # Log progress
            # ######################################################
            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # ######################################################
        # Save progress
        # ######################################################
        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)


        # ######################################################
        # Evaluate
        # ######################################################
        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)


if __name__ == "__main__":
    run_train()