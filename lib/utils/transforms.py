
import torchvision.transforms as transforms




## 建立transforms，之後建立驗證資料集dataset時，需要做的檔案轉換設定DEFAULT_TRANSFORMS
# 驗證集的資料，不做資料擴增處理
DEFAULT_TRANSFORMS = transforms.Compose([
    # 原始資料的邊框值為相對於邊長1的相對邊框值，換算回絕對邊框值
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])


## 建立transforms，之後建立訓練資料集dataset時，需要做的檔案轉換設定AUGMENTATION_TRANSFORMS
AUGMENTATION_TRANSFORMS = transforms.Compose([
    # 原始資料的邊框值為相對於邊長1的相對邊框值，換算回絕對邊框值
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])