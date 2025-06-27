import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from Net import AutoEncoder
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

class SimpleDepthInference:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(feature_channels=256).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def infer(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        dummy_depth = torch.zeros_like(image[:, :1])
        image_with_depth = torch.cat([image, dummy_depth], dim=1)
        with torch.no_grad():
            output = self.model(image_with_depth, real=False)
        return output[0, 3, :, :].cpu().numpy()

    def visualize_and_save(self, rgb, depth_map):
        rgb = TF.resize(rgb, [128, 128])
        depth_map = np.clip(depth_map / np.percentile(depth_map, 95), 0, 1)
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        rgb_np = np.array(rgb)[:, :, ::-1]  # RGB to BGR for OpenCV
        vis = np.hstack([rgb_np, depth_color])
        return vis

    def run_on_folder(self, input_folder, output_video="output.mp4", max_frames=1000):
        frames = []
        image_files = sorted([
            f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])[:max_frames]

        for fname in tqdm(image_files, desc="🔍 Estimating depth"):
            path = os.path.join(input_folder, fname)
            image = Image.open(path).convert("RGB")
            depth_map = self.infer(image)
            vis = self.visualize_and_save(image, depth_map)
            frames.append(vis)

        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"✅ Saved visualization video to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder of RGB frames")
    parser.add_argument("--model_path", type=str, default="Depth.pth", help="Trained AutoEncoder model path")
    parser.add_argument("--output_video", type=str, default="output.mp4", help="Path to output video")
    parser.add_argument("--max_frames", type=int, default=1000, help="Max frames to process")
    args = parser.parse_args()

    inferencer = SimpleDepthInference(args.model_path)
    inferencer.run_on_folder(args.input_folder, args.output_video, args.max_frames)

# python Infer_Video.py --input_folder /path/to/frames --model_path /path/to/Depth.pth --output_video output.mp4 --max_frames 1000