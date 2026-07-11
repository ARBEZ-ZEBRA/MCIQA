import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class ColorQualityFusionNet(nn.Module):
    def __init__(self):
        super(ColorQualityFusionNet, self).__init__()

        res50 = models.resnet50(pretrained=True)
        self.naturalness_extractor = nn.Sequential(
            *list(res50.children())[:-1]
        )

        self.score_map_extractor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Linear(2048 + 128, 512),
            nn.ReLU(),

            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, img, score_maps):
        feat_img = self.naturalness_extractor(img)
        feat_img = feat_img.view(img.size(0), -1)  

        feat_score = self.score_map_extractor(score_maps)
        feat_score = feat_score.view(img.size(0), -1) 

        combined = torch.cat([feat_img, feat_score], dim=1)

        out = self.regressor(combined)

        return out


class FusionDataset(Dataset):
    def __init__(
        self,
        img_dir,
        CF_dir,
        SCC_dir,
        target_mos_json,
        transform=None
    ):
        self.img_dir = img_dir
        self.CF_dir = CF_dir
        self.SCC_dir = SCC_dir

        with open(target_mos_json, 'r') as f:
            self.targets = json.load(f)

        self.img_names = list(self.targets.keys())

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        name = self.img_names[idx]

        img_path = os.path.join(self.img_dir, name + ".jpg")

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        cf_path = os.path.join(self.CF_dir, name + ".npy")
        cf_map = np.load(cf_path).astype(np.float32)

        scc_path = os.path.join(self.SCC_dir, name + ".npy")
        scc_map = np.load(scc_path).astype(np.float32)

        score_maps = np.stack([cf_map, scc_map], axis=0)

        # tensor
        score_maps = torch.from_numpy(score_maps)

        target = torch.tensor(
            [self.targets[name]],
            dtype=torch.float32
        )

        return img, score_maps, target


def train_fusion_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    CONFIG = {
        "train_img_dir": "./MCIQA_2K_train/",
        "train_CS_dir": "/weight_QWEN_CS_result",
        "train_SCM_dir": "/weight_QWEN_SCM_result",
        "train_target_json": "MCIQA_2K_GN_MOS_train.json",

        "test_img_dir": "./MCIQA_2K_test/",
        "test_CS_dir": "/weight_QWEN_CS_result",
        "test_SCM_dir": "/weight_QWEN_SCM_result",
        "test_target_json": "/MCIQA_2K_GN_MOS_test.json",

        "batch_size": 16,
        "lr": 1e-4,
        "epochs": 50,

        "model_savepath": "best_gn_qwen.pth",
        "plot_savepath": "GN_training_curve.png"
    }

    train_dataset = FusionDataset(
        CONFIG["train_img_dir"],
        CONFIG["train_CS_dir"],
        CONFIG["train_SCM_dir"],
        CONFIG["train_target_json"]
    )

    test_dataset = FusionDataset(
        CONFIG["test_img_dir"],
        CONFIG["test_CS_dir"],
        CONFIG["test_SCM_dir"],
        CONFIG["test_target_json"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4
    )

    model = ColorQualityFusionNet().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"]
    )

    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    min_test_loss = 1

    for epoch in range(CONFIG["epochs"]):
        model.train()

        epoch_train_loss = 0

        for imgs, score_maps, targets in train_loader:

            imgs = imgs.to(device)
            score_maps = score_maps.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            preds = model(imgs, score_maps)

            loss = criterion(preds, targets)

            loss.backward()

            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        train_losses.append(avg_train_loss)

        model.eval()

        epoch_test_loss = 0

        with torch.no_grad():

            for imgs, score_maps, targets in test_loader:

                imgs = imgs.to(device)
                score_maps = score_maps.to(device)
                targets = targets.to(device)

                preds = model(imgs, score_maps)

                loss = criterion(preds, targets)

                epoch_test_loss += loss.item()

        avg_test_loss = epoch_test_loss / len(test_loader)

        test_losses.append(avg_test_loss)

        if (avg_test_loss < min_test_loss):
            min_test_loss = avg_test_loss
            torch.save(model.state_dict(), CONFIG["model_savepath"])

        print(
            f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Test Loss: {avg_test_loss:.6f}"
        )

        if avg_train_loss <= 0.001:
            break

    torch.save(
        model.state_dict(),
        "final_gn_qwen.pth"
    )


    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(CONFIG["plot_savepath"])


if __name__ == "__main__":
    train_fusion_model()