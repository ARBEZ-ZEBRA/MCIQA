import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

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
        
        alpha = 1
        final_score = alpha * (1 - 2 * global_score) + (1 - alpha) * results

        return score_map, final_score

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

def save_score_map_visualization(image_path, score_map, save_path, global_score, label=None):
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape

    s_map = score_map.squeeze().cpu().float().numpy()
    np.save("weight_" + save_path + ".npy", s_map)
    
    s_max = s_map.max()
    s_min = s_map.min()

    s_map_norm = np.full(s_map.shape, 128, dtype=np.uint8)

    pos_mask = s_map > 0
    if pos_mask.any():
        pos_max = s_map[pos_mask].max()
        s_map_norm[pos_mask] = 128 + (s_map[pos_mask] / (pos_max + 1e-8) * 127)

    neg_mask = s_map < 0
    if neg_mask.any():
        neg_min = s_map[neg_mask].min()
        s_map_norm[neg_mask] = 128 - (s_map[neg_mask] / (neg_min - 1e-8) * 128)

    s_map_resized = cv2.resize(s_map_norm, (w, h), interpolation=cv2.INTER_LINEAR)

    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    for i in range(128):
        lut[i] = [int((127 - i) * 2), 0, 0]

    lut[128] = [0, 0, 0]

    for i in range(129, 256):
        lut[i] = [0, 0, int((i - 129) * 2)]

    heatmap = cv2.LUT(cv2.merge([s_map_resized]*3), lut)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    # h_max = heatmap_gray.max()
    # alpha = heatmap_gray.astype(np.float32) / heatmap_gray.max()
    # alpha = np.expand_dims(alpha, axis=2)
    # result = orig_img.astype(np.float32) * (1.0 - alpha) + heatmap.astype(np.float32) * alpha
    # heatmap = np.clip(result, 0, 255).astype(np.uint8)
    img = Image.fromarray(heatmap)
    img.save(save_path + ".png")

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.title(f"Label: {label if label is not None else 'N/A'}")
    # plt.imshow(orig_img)
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title(f"Heatmap")
    # plt.imshow(heatmap)
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title(f"Prediction: {global_score:.4f}")
    # plt.imshow(overlap)
    # plt.axis('off')

    # plt.tight_layout()
    # plt.savefig(save_path)
    # plt.close()

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen3-VL-32B-Instruct" # local path of qwen3-vl-32B
    checkpoint_path = "best_cs_qwen.pth"
    test_img_dir = 'MCIQA_2K'
    test_json = 'MCIQA_2K_CS_MOS.json'
    output_dir = 'QWEN_CS_result'
    weight_folder = 'MCIQA_2K_CS_MASK'
    os.makedirs(output_dir, exist_ok=True)
    test_result_json = 'MCIQA_2K_CS_MOS.json'
    with open(test_result_json, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    processor = AutoProcessor.from_pretrained(model_path)
    model = QwenPixelCFModel(model_path)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device, dtype=torch.bfloat16)
    model.eval()

    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    image_names = list(test_data.keys())

    print(f"Starting inference on {len(image_names)} images...")
    results = {}

    with torch.no_grad():
        for img_name in tqdm(image_names):
            img_path = os.path.join(test_img_dir, img_name + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(test_img_dir, img_name + ".png")
            
            if not os.path.exists(img_path):
                continue
            
            weight_path = os.path.join(weight_folder, img_name + ".pt")
            weight_map = torch.load(weight_path, map_location='cpu')

            if not isinstance(weight_map, torch.Tensor):
                weight_map = torch.tensor(weight_map)
            weight_map = weight_map.to(torch.float32)

            if weight_map.ndim == 2:
                weight_map = weight_map.unsqueeze(0).unsqueeze(0)
            elif weight_map.ndim == 3:
                weight_map = weight_map.unsqueeze(0)
            
            weight_map = torch.nn.functional.interpolate(weight_map, size=(256, 256), mode='bilinear', align_corners=False)
            blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 16.0
            weight_map = torch.nn.functional.conv2d(weight_map, blur_kernel, padding=1).squeeze(0).squeeze(0)
            weight_map = (weight_map.view(-1) > 48).float()
            weight_map = weight_map / (weight_map.sum() + 1e-8)
            weight_map = weight_map.view(1, 256, 256)

            image = Image.open(img_path).convert("RGB")
            prompt = "<|image_pad|>"
            inputs = processor(images=[image], text=[prompt], return_tensors="pt")
            
            pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
            grid_thw = inputs["image_grid_thw"].to(device)

            score_map, global_score = model(pixel_values, grid_thw, weight_maps=weight_map.to(device, dtype=torch.bfloat16), results=torch.tensor(float(result_data[img_name]), dtype=torch.float32))

            pred_val = global_score.item()
            gt_val = float(test_data[img_name])
            results[img_name] = pred_val

            save_path = (output_dir + f"/{img_name}")
            save_score_map_visualization(img_path, score_map, save_path, pred_val, gt_val)
    
    with open("MCIQA_2K_CS_result.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, sort_keys=True)
    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    test()