import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from patch_detector import PatchDetector
from tqdm import tqdm

def main():
    input_dir = 'test_60'
    output_dir = 'test_60_output'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize SAC processor
    print("Loading SAC Processor...")
    SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[125, 100, 75, 50, 25], n_patch=1)
    SAC_processor.unet.load_state_dict(torch.load("ckpts/coco_at.pth", map_location=device))
    SAC_processor = SAC_processor.to(device)
    SAC_processor.eval()

    transform = transforms.Compose([
        transforms.ToTensor(), # converts to [0, 1]
    ])

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in '{input_dir}'.")

    total_time = 0.0

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device) # Shape: 1x3xHxW

            # Start timer
            start_time = time.time()

            # Process image through SAC
            x_processed, mask, _ = SAC_processor(x, bpda=True, shape_completion=True)

            # End timer
            end_time = time.time()
            process_time = end_time - start_time
            total_time += process_time
            
            # Print time taken for this image
            tqdm.write(f"Image '{filename}' processed in {process_time:.4f} seconds.")

            # Save the processed image
            output_path = os.path.join(output_dir, filename)
            save_image(x_processed[0].cpu(), output_path)

    avg_time = total_time / len(image_files) if len(image_files) > 0 else 0
    print(f"\nFinished processing. Outputs saved to '{output_dir}'.")
    print(f"Total processing time for {len(image_files)} images: {total_time:.4f} seconds.")
    print(f"Average processing time per image: {avg_time:.4f} seconds.")

if __name__ == '__main__':
    main()
