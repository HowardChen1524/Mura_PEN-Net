import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
# from vit_pytorch import ViT
import timm


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 128
        self.pool1 = nn.MaxPool2d(2, 2)  # 64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2), bias=False)  # 64
        self.pool2 = nn.MaxPool2d(2, 2)  # 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)  # 32
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 1)

        self.drop1 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.drop1(x)
        x = self.fc4(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 128
        self.pool1 = nn.MaxPool2d(2, 2)  # 64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2), bias=False)  # 64
        self.pool2 = nn.MaxPool2d(2, 2)  # 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)  # 32
        # self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc1_ext = nn.Linear(64 * 16 * 16 + 3, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 1)

        self.drop1 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x, centers: None):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x, centers), 1)  # cat center point
        if centers is None:
            centers = torch.tensor([[0, 0, 0] for _ in range(x.shape[0])]).to(x.device)
        x = F.relu(self.fc1_ext(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.drop1(x)
        x = self.fc4(x)
        return x


def get_model(model=None, size=256, pretrain=False):
    if model not in ["vgg16", "resnet50", "cnn", "cnn2", "xception", "mobilenet_v2", "vit", "seresnext101", "convit"]:
        raise ValueError("Get error model:{}".format(model))
    if size not in [128, 256, 224, 512]:
        raise ValueError("Get error image size:{}".format(size))

    if pretrain:
        print("Use pretrain model.")

    if model == "vgg16":
        mod = models.vgg16(pretrained=pretrain)
        mod.add_module("classifier",
                       nn.Sequential(
                           nn.Linear(in_features=25088, out_features=4096, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.2, inplace=False),
                           nn.Linear(in_features=4096, out_features=512, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.2, inplace=False),
                           nn.Linear(in_features=512, out_features=64, bias=True),
                           nn.ReLU(),
                           nn.Dropout(p=0.2, inplace=False),
                           nn.Linear(in_features=64, out_features=1, bias=True),
                       )
                       )

    elif model == "resnet50":
        mod = models.resnet50(pretrained=pretrain)
        mod.add_module("fc",
                       nn.Sequential(
                           nn.Linear(in_features=2048, out_features=1000, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=1000, out_features=512, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=512, out_features=64, bias=True),
                           nn.ReLU(),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=64, out_features=1, bias=True),
                           nn.Sigmoid()
                       )
                    )
    elif model == "cnn":
        mod = Net()
        if size == 128:
            mod.fc1 = nn.Linear(64 * 8 * 8, 512)
        if size == 512:
            mod.fc1 = nn.Linear(64 * 32 * 32, 512)
    elif model == "cnn2":
        mod = Net2()

    elif model == "vit":
        if size == 128:
            mod = ViT(image_size=128, patch_size=16, num_classes=1, channels=3, dim=64, depth=6, heads=8, mlp_dim=128)
        if size == 256:
            mod = ViT(image_size=256, patch_size=32, num_classes=1, channels=3, dim=64, depth=6, heads=8, mlp_dim=128)
        if size == 512:
            mod = ViT(image_size=512, patch_size=64, num_classes=1, channels=3, dim=64, depth=6, heads=8, mlp_dim=128)

    elif model == "xception":
        mod = pretrainedmodels.models.xception(pretrained=pretrain)
        mod.add_module("last_linear",
                       nn.Sequential(
                           nn.Linear(in_features=2048, out_features=1000, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=1000, out_features=512, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=512, out_features=64, bias=True),
                           nn.ReLU(),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=64, out_features=1, bias=True),
                           nn.Sigmoid()
                       ))

    elif model == "mobilenet_v2":
        mod = models.mobilenet_v2(pretrained=pretrain)
        mod.add_module("classifier",
                       nn.Sequential(
                           nn.Dropout(p=0.2, inplace=False),
                           nn.Linear(in_features=1280, out_features=1000, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=1000, out_features=512, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=512, out_features=64, bias=True),
                           nn.ReLU(),
                           nn.Dropout(p=0.3, inplace=False),
                           nn.Linear(in_features=64, out_features=1, bias=True),
                       ))

    elif model == "seresnext101":
        mod = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
        mod.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    elif model == "convit":
        model_name = 'convit_base'
        mod = timm.create_model(model_name, img_size=256, pretrained=False, num_classes=1)
        mod.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=1),
            nn.Sigmoid()
        )

    return mod
