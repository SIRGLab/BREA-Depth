import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from Net import AutoEncoder
import natsort
import numpy as np
import PIL.Image as pil
import tqdm

class AutoEncoderInference:
    def __init__(self, model_path, model, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.autoencoder = model
        if model_path:
            self.autoencoder.load_state_dict(torch.load(model_path, map_location=self.device))
            self.autoencoder.to(self.device)
            print(f"✅ Model loaded from {model_path} and ready for inference.")
        self.autoencoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def infer_image(self, image_path, real=True):
        image_tensor = self.load_image(image_path).unsqueeze(0).to(self.device)
        if image_tensor.shape[1] == 3:
            depth_tensor = torch.zeros_like(image_tensor[:, :1, :, :])
            image_tensor = torch.cat((image_tensor, depth_tensor), dim=1)

        output = self.autoencoder(image_tensor, real=real)
        return output

    def infer_folder(self, input_folder, output_folder, real=True, count_num=1000):
        os.makedirs(output_folder, exist_ok=True)
        image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
        image_files = natsort.natsorted(image_files)[:count_num]
        print(f"🔹 Found {len(image_files)} images in {input_folder}. Running inference...")

        for image_file in tqdm.tqdm(image_files, total=len(image_files)):
            image_path = os.path.join(input_folder, image_file)
            output = self.infer_image(image_path, real=real)
            output_path = os.path.join(output_folder, f"output_{image_file}")
            depth_path = os.path.join(output_folder, f"{image_file}")
            self.save_output(output, output_path, depth_path)
        
        print(f"✅ Inference completed. Results saved in {output_folder}.")

    def save_output(self, output, save_path, depth_path):
        depth = output[:, 3, :, :]
        depth_map = depth.squeeze().detach().cpu().numpy()
        np.save(depth_path.replace('.jpg', '.npy').replace('.png', '.npy'), depth_map)
        
        vmax = np.percentile(depth_map, 95)
        disp_resized_np = (255 * (depth_map / vmax)).clip(0, 255).astype(np.uint8)
        im = pil.fromarray(disp_resized_np, mode="L")
        im.save(depth_path, quality=95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder Inference Script")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder")
    parser.add_argument("--model_path", type=str, default="Depth.pth", help="Path to the model checkpoint")
    parser.add_argument("--real", action='store_true', default=False, help="Real or Fake")
    parser.add_argument("--count_num", type=int, default=100000, help="Max number of images to process")
    
    args = parser.parse_args()

    model = AutoEncoder(feature_channels=256)
    inference = AutoEncoderInference(args.model_path, model)
    
    os.makedirs(args.output_folder, exist_ok=True)
    inference.infer_folder(args.input_folder, args.output_folder, real=args.real, count_num=args.count_num)

    # Example Usage:
    # python Infer.py --input_folder Input --output_folder Output 