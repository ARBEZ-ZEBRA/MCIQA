import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

class ICQADataset(Dataset):
    def __init__(self, image_folder, weight_folder, json_path, processor, result_path):
        self.image_folder = image_folder
        self.weight_folder = weight_folder
        with open(json_path, 'r', encoding='utf-8') as f:
            self.label_data = json.load(f)
        with open(result_path, 'r', encoding='utf-8') as f:
            self.result_data = json.load(f)
        self.image_names = list(self.label_data.keys())
        self.processor = processor
        
        self.valid_names = [n for n in self.image_names if os.path.exists(os.path.join(image_folder, n + ".jpg")) or os.path.exists(os.path.join(image_folder, n + ".png"))]

    def __len__(self):
        return len(self.valid_names)

    def __getitem__(self, idx):
        img_name = self.valid_names[idx]
        for ext in ['.jpg']:
            img_path = os.path.join(self.image_folder, img_name + ext)
            if os.path.exists(img_path):
                break

        weight_path = os.path.join(self.weight_folder, img_name + ".pt")
        weight_map = torch.load(weight_path, map_location='cpu')

        if not isinstance(weight_map, torch.Tensor):
            weight_map = torch.tensor(weight_map)

        weight_map = weight_map.sum(dim=-1)
        max_value = torch.max(weight_map)
        weight_map = torch.where(weight_map > 0, max_value - weight_map, weight_map)
        weight_map = (weight_map.view(-1)).float()
        weight_map = weight_map / (weight_map.sum() + 1e-8)
        weight_map = weight_map.view(1, 256, 256)
        
        image = Image.open(img_path).convert("RGB")
        label = float(self.label_data[img_name])
        result = float(self.result_data[img_name])
        
        prompt = "<|image_pad|>Analyze the color confidence of each object in this image." 
        inputs = self.processor(images=[image], text=[prompt], return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
            "result": torch.tensor(result, dtype=torch.float32),
            "weight_map": weight_map
        }

def qwen_collate_fn(batch):
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.stack([item["image_grid_thw"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    results = torch.stack([item["result"] for item in batch], dim=0)
    weight_maps = torch.stack([item["weight_map"] for item in batch], dim=0)
    
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,
        "results": results,
        "weight_maps": weight_maps
    }

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        )

    def forward(self, x):
        x = x * self.ca(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sam = torch.cat([max_out, avg_out], dim=1)
        return x * self.sa(sam)

class ImprovedDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(hidden_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        self.attention1 = CBAM(512)
        
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class QwenPixelCFModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto"
        )
        if hasattr(full_model, 'visual'):
            self.visual = full_model.visual
        elif hasattr(full_model, 'model') and hasattr(full_model.model, 'visual'):
            self.visual = full_model.model.visual
        hidden_size = self.visual.config.hidden_size

        self.decoder = ImprovedDecoder(hidden_size)

        # for param in self.visual.parameters():
        #     param.requires_grad = False
        # for param in self.decoder.parameters():
        #     param.requires_grad = False

    def forward(self, pixel_values, grid_thw, weight_maps, results):
        vis_outputs = self.visual(pixel_values, grid_thw)
        hidden_states = vis_outputs.last_hidden_state 

        batch_size = grid_thw.shape[0]
        feature_maps = []
        start_idx = 0
        
        for i in range(batch_size):
            h_patch = grid_thw[i][1].item()
            w_patch = grid_thw[i][2].item()
            num_tokens = h_patch * w_patch
            
            sample_feat = hidden_states[start_idx : start_idx + num_tokens]
            start_idx += num_tokens
            
            sample_feat = sample_feat.view(h_patch, w_patch, -1).permute(2, 0, 1)
            feature_maps.append(sample_feat)

        target_size = (32, 32)
        resized_maps = [
            torch.nn.functional.interpolate(f.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            for f in feature_maps
        ]

        feat_2d = torch.stack(resized_maps, dim=0) 
        
        pixel_score_map = self.decoder(feat_2d)
        
        score_map = pixel_score_map * weight_maps
        global_score = torch.sum(score_map, dim=[2, 3])
        
        final_score = (1 - 2 * global_score)

        return pixel_score_map, final_score

class CFLoss(nn.Module):
    def __init__(self, rank_weight=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.rank_weight = rank_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        
        if pred.size(0) > 1:
            diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)
            diff_target = target.unsqueeze(0) - target.unsqueeze(1)
            rank_loss = torch.mean(F.relu(-diff_pred * torch.sign(diff_target)))
        else:
            rank_loss = 0.0
            
        return mse_loss + self.rank_weight * rank_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/Qwen/Qwen3-VL-32B-Instruct" # local path of qwen3-vl-32B
    train_img_dir = 'MCIQA_2K_train'
    train_json = 'MCIQA_2K_SCM_MOS_train.json'
    test_img_dir = 'MCIQA_2K_test'
    test_json = 'MCIQA_2K_SCM_MOS_test.json'
    train_weight_dir = '/MCIQA_2K_SCM_MASK'
    test_weight_dir = '/MCIQA_2K_SCM_MASK'
    train_result_json = 'MCIQA_2K_SCM_MOS_train.json'
    test_result_json = 'MCIQA_2K_SCM_MOS_test.json'
    batch_size = 4  
    epochs = 1000

    processor = AutoProcessor.from_pretrained(model_path)
    
    train_dataset = ICQADataset(train_img_dir, train_weight_dir, train_json, processor, train_result_json)
    test_dataset = ICQADataset(test_img_dir, test_weight_dir, test_json, processor, test_result_json)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=qwen_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=qwen_collate_fn)

    criterion = CFLoss()
    
    model = QwenPixelCFModel(model_path).to(device, dtype=torch.bfloat16)
    # state_dict = torch.load("best_scm_qwen.pth", map_location='cpu')
    # model.load_state_dict(state_dict)
    optimizer = optim.AdamW([
        {'params': model.visual.parameters(), 'lr': 1e-5},
        {'params': model.decoder.parameters(), 'lr': 1e-5} 
    ], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.eval()
    train_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=f"Epoch {0}/{epochs} [Train]"):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            labels = batch["labels"].to(device, dtype=torch.bfloat16).unsqueeze(1)
            results = batch["results"].to(device, dtype=torch.bfloat16).unsqueeze(1)
            weight_maps = batch["weight_maps"].to(device, dtype=torch.bfloat16)

            _, pred_global = model(pixel_values, image_grid_thw, weight_maps=weight_maps, results=results)
            loss = criterion(pred_global, labels)
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            labels = batch["labels"].to(device, dtype=torch.bfloat16).unsqueeze(1)
            results = batch["results"].to(device, dtype=torch.bfloat16).unsqueeze(1)
            weight_maps = batch["weight_maps"].to(device, dtype=torch.bfloat16)
                
            _, pred_global = model(pixel_values, image_grid_thw, weight_maps=weight_maps, results=results)
            loss = nn.MSELoss()(pred_global, labels) 
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {0}: Train Loss = {avg_train_loss:.4f}, Test MSE = {avg_test_loss:.4f}")
    best_loss = 0.9*avg_train_loss + 0.1*avg_test_loss
    #best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            labels = batch["labels"].to(device, dtype=torch.bfloat16).unsqueeze(1)
            results = batch["results"].to(device, dtype=torch.bfloat16).unsqueeze(1)
            weight_maps = batch["weight_maps"].to(device, dtype=torch.bfloat16)

            optimizer.zero_grad()
            _, pred_global = model(pixel_values, image_grid_thw, weight_maps=weight_maps, results=results)
            loss = criterion(pred_global, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[Test]"):
                pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
                image_grid_thw = batch["image_grid_thw"].to(device)
                labels = batch["labels"].to(device, dtype=torch.bfloat16).unsqueeze(1)
                results = batch["results"].to(device, dtype=torch.bfloat16).unsqueeze(1)
                weight_maps = batch["weight_maps"].to(device, dtype=torch.bfloat16)
                
                _, pred_global = model(pixel_values, image_grid_thw, weight_maps=weight_maps, results=results)
                loss = nn.MSELoss()(pred_global, labels) 
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test MSE = {avg_test_loss:.4f}")

        if 0.9*avg_train_loss + 0.1*avg_test_loss < best_loss:
            best_loss = 0.9*avg_train_loss + 0.1*avg_test_loss
            torch.save(model.state_dict(), "best_scm_qwen.pth")
            print("Model saved!")

    torch.save(model.state_dict(), "final_scm_qwen.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()