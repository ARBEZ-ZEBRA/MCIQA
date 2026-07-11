import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


class ColorQualityFusionNet(nn.Module):
    def __init__(self):
        super(ColorQualityFusionNet, self).__init__()

        res50 = models.resnet50(pretrained=False)

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


class FusionTestDataset(Dataset):
    def __init__(
        self,
        img_dir,
        CF_dir,
        SCC_dir,
        transform=None
    ):
        self.img_dir = img_dir
        self.CF_dir = CF_dir
        self.SCC_dir = SCC_dir

        self.img_names = []

        for file in os.listdir(img_dir):
            if file.endswith(".jpg"):
                self.img_names.append(
                    os.path.splitext(file)[0]
                )

        self.img_names.sort()

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

        img_path = os.path.join(
            self.img_dir,
            name + ".jpg"
        )

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        cf_path = os.path.join(
            self.CF_dir,
            name + ".npy"
        )

        cf_map = np.load(cf_path).astype(np.float32)

        scc_path = os.path.join(
            self.SCC_dir,
            name + ".npy"
        )

        scc_map = np.load(scc_path).astype(np.float32)

        score_maps = np.stack(
            [cf_map, scc_map],
            axis=0
        )

        score_maps = torch.from_numpy(score_maps)

        return img, score_maps, name


def test_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    CONFIG = {
        "img_dir": "./MCIQA_2K/",
        "CF_dir": "weight_QWEN_CS_result",
        "SCC_dir": "weight_QWEN_SCM_result",

        "model_path": "best_gn_qwen.pth",

        "batch_size": 16,

        "save_json": "MCIQA_2K_GN_result.json"
    }

    test_dataset = FusionTestDataset(
        CONFIG["img_dir"],
        CONFIG["CF_dir"],
        CONFIG["SCC_dir"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4
    )

    model = ColorQualityFusionNet().to(device)

    model.load_state_dict(
        torch.load(
            CONFIG["model_path"],
            map_location=device
        )
    )

    model.eval()

    print("Model loaded!")

    results = {}

    with torch.no_grad():

        for imgs, score_maps, names in test_loader:

            imgs = imgs.to(device)
            score_maps = score_maps.to(device)

            preds = model(imgs, score_maps)

            preds = preds.squeeze(1).cpu().numpy()

            for name, pred in zip(names, preds):

                results[name] = float(pred)

    with open(CONFIG["save_json"], "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )

    print(f"Prediction json saved to: {CONFIG['save_json']}")


if __name__ == "__main__":
    test_model()