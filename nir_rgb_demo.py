#!/usr/bin/env python3
import os
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from segfarm_segmentor import SegFarmSegmentation
import time
import argparse

# Add command line argument parsing for testing different weight files
parser = argparse.ArgumentParser(description='SegFarm NIR+RGB Farmland Segmentation Test')
parser.add_argument('--weights', type=str, default='simfeatup_dev/weights/xclip_jbu_one_agricrop.ckpt',
                   help='Path to upsampler weight file')
parser.add_argument('--input_dir', type=str, default='data/demo',
                   help='Input data directory')
parser.add_argument('--output_dir', type=str, default='data/demo/results',
                   help='Output results directory')
parser.add_argument('--model_type', type=str, default='jbu_one',
                   help='Upsampler model type')
parser.add_argument('--threshold', type=float, default=0.5,
                   help='Segmentation mask threshold')
parser.add_argument('--use_mlp_corrector', action='store_true',
                   help='Use MLP corrector instead of fixed cls_token_lambda')
parser.add_argument('--frozen_mlp', action='store_true',
                   help='Freeze MLP corrector weights (only effective when --use_mlp_corrector is True)')
args = parser.parse_args()

print(f"=== Farmland Segmentation Test ===")
print(f"Using weight file: {args.weights}")
print(f"Upsampler type: {args.model_type}")
print(f"Segmentation threshold: {args.threshold}")
print(f"Input directory: {args.input_dir}")
print(f"Output directory: {args.output_dir}")

# 1. Write class names
name_list = ['Background: A body of water alongside some land - based structures and vegetation.', 'Farmland: Composed of orderly, rectangular plots displaying a mix of earthy brown and green hues, indicative of different crops or growth stages, arranged in a systematic agricultural layout.']
os.makedirs('configs', exist_ok=True)
with open('configs/1_name.txt', 'w') as f:
    f.write('\n'.join(name_list))

# 2. Model initialization
print("\nInitializing model...")
print("Using Enhanced JBUOne model:")
print("  1. Improved feature fusion mechanism: by adding channel attention, spatial attention, and cross-modal attention")
print("  2. Adaptive loss weighting: dynamically adjusts weights based on contribution of each loss term during training")
print("  3. Farmland boundary awareness module: better preserves farmland boundaries")
print("  4. Self-supervised pretraining techniques")

if args.use_mlp_corrector:
    print("  5. Using MLP corrector: learnable global-local debiasing weights, replacing fixed global bias subtraction")
    if args.frozen_mlp:
        print("     MLP corrector weights frozen and will not be updated during training")
    cls_token_lambda = 0  # No longer need fixed cls_token_lambda when using MLP corrector
else:
    print("  5. Using fixed global bias subtraction, cls_token_lambda=-0.3")
    cls_token_lambda = -0.3  # Use original fixed coefficient method

model = SegFarmSegmentation(
    clip_type='CLIP',
    vit_type='ViT-B/16',
    model_type='SegFarm',
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
        model_name=args.model_type,  # Use upsampler specified by command-line argument
        model_path=args.weights  # Use weights file specified by command-line argument
    ),
    cls_token_lambda=cls_token_lambda,
    mlp_corrector=args.use_mlp_corrector,  # Whether to use MLP corrector
    frozen_mlp=args.frozen_mlp,           # Whether to freeze MLP corrector weights
    name_path='configs/1_name.txt',
    prob_thd=0.1,
    use_nir=True,  # Enable NIR channel support
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()
print(f"Model loaded to {device} device")

# 3. Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((256, 256))
])

# 4. Batch traverse and inference
rgb_dir = os.path.join(args.input_dir, 'rgb')
nir_dir = os.path.join(args.input_dir, 'nir')
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Get all RGB image files
rgb_files = glob.glob(os.path.join(rgb_dir, '*.png'))
if not rgb_files:
    print(f"Warning: No PNG image files found in {rgb_dir}")
    rgb_files = glob.glob(os.path.join(rgb_dir, '*.jpg'))
    if not rgb_files:
        print(f"Error: No image files found in {rgb_dir}")
        exit(1)

print(f"\nFound {len(rgb_files)} image files, starting processing...")
total_time = 0
processed_count = 0

for rgb_path in rgb_files:
    start_time = time.time()
    
    # 4.1 Get corresponding NIR image path
    base_name = os.path.basename(rgb_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    
    # Try different extensions
    nir_extensions = ['.png', '.jpg', '.tif']
    nir_path = None
    
    for ext in nir_extensions:
        test_path = os.path.join(nir_dir, base_name_no_ext + ext)
        if os.path.exists(test_path):
            nir_path = test_path
            break
    
    if not nir_path:
        print(f"Warning: Cannot find corresponding NIR image {os.path.join(nir_dir, base_name_no_ext)}.*, skipping processing")
        continue
    
    # 4.2 Read RGB and NIR images
    rgb_img = Image.open(rgb_path).convert('RGB')
    nir_img = Image.open(nir_path).convert('L')  # Read NIR image in grayscale mode
    
    # Ensure both images have the same dimensions
    if rgb_img.size != nir_img.size:
        print(f"Resizing NIR image to match RGB image: {rgb_img.size}")
        nir_img = nir_img.resize(rgb_img.size, Image.BILINEAR)
    
    # 4.3 Preprocess RGB image
    rgb_tensor = preprocess(rgb_img).unsqueeze(0)
    
    # 4.4 Preprocess NIR image (convert to single channel tensor and normalize)
    # Apply same normalization processing to NIR image
    nir_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466], [0.26862954]),  # Use first channel normalization parameters from RGB
        transforms.Resize((256, 256))
    ])(nir_img).unsqueeze(0)
    
    # 4.5 Combine RGB and NIR inputs into 4-channel input
    combined_tensor = torch.cat([rgb_tensor, nir_tensor], dim=1).to(device)
    
    with torch.no_grad():
        # Use model for prediction, passing in 4-channel input
        seg_pred = model.predict(combined_tensor, data_samples=None)
    
    # 4.6 Process prediction results
    mask = seg_pred.data.cpu().numpy().squeeze(0)  # H×W float array
    
    # 4.7 Binary threshold processing
    mask_bin = (mask > args.threshold).astype(np.uint8)
    mask_uint8 = mask_bin * 255  # 0→0, 1→255
    
    # 4.8 Save results
    base = os.path.splitext(base_name)[0]
    
    # Save binary mask
    mask_path = os.path.join(output_dir, f'{base}_mask.png')
    Image.fromarray(mask_uint8, mode='L').save(mask_path)
    
    # Create visualization result (RGB+NIR+mask combined image)
    # Display RGB, NIR, and mask side by side
    result_width = rgb_img.width * 3
    result_height = rgb_img.height
    result_img = Image.new('RGB', (result_width, result_height))
    
    # Place RGB image
    result_img.paste(rgb_img, (0, 0))
    
    # Convert NIR to RGB and place
    nir_rgb = Image.merge('RGB', (nir_img, nir_img, nir_img))
    result_img.paste(nir_rgb, (rgb_img.width, 0))
    
    # Convert mask to RGB and place
    mask_rgb = Image.merge('RGB', (Image.fromarray(mask_uint8), Image.fromarray(mask_uint8), Image.fromarray(mask_uint8)))
    result_img.paste(mask_rgb, (rgb_img.width * 2, 0))
    
    # Save combined image
    result_path = os.path.join(output_dir, f'{base}_segmentation.png')
    result_img.save(result_path)
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    processed_count += 1
    
    print(f"Processing completed: {base_name} (time elapsed: {elapsed_time:.2f}s)")

if processed_count > 0:
    avg_time = total_time / processed_count
    print(f"\nAll images processed! Results saved in {output_dir}")
    print(f"Average processing time: {avg_time:.2f}s/image")
    print(f"Total processing time: {total_time:.2f}s")
else:
    print("No images were successfully processed.")