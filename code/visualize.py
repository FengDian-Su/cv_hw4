import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PromptIR'))
from net.model import PromptIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       type=str, required=True)
    parser.add_argument('--data_root',  type=str, default='dataset/train')
    parser.add_argument('--output',     type=str, default='visualization.png')
    parser.add_argument('--gpu',        type=str, default='0')
    # which indices to visualize (one rain, one snow)
    parser.add_argument('--rain_idx',   type=int, default=1)
    parser.add_argument('--snow_idx',   type=int, default=1)
    return parser.parse_args()


def load_model(ckpt_path, device):
    model = PromptIR(decoder=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model']
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer(model, img_np, device):
    toTensor = ToTensor()
    tensor = toTensor(img_np).unsqueeze(0).to(device)
    with torch.no_grad():
        restored = model(tensor).clamp(0, 1)
    img_out = (restored.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return img_out


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.ckpt, device)
    print(f'Model loaded from {args.ckpt}')

    degraded_dir = os.path.join(args.data_root, 'degraded')
    clean_dir    = os.path.join(args.data_root, 'clean')

    samples = [
        (
            os.path.join(degraded_dir, f'rain-{args.rain_idx}.png'),
            os.path.join(clean_dir,    f'rain_clean-{args.rain_idx}.png'),
            'Rain'
        ),
        (
            os.path.join(degraded_dir, f'snow-{args.snow_idx}.png'),
            os.path.join(clean_dir,    f'snow_clean-{args.snow_idx}.png'),
            'Snow'
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    col_titles = ['Degraded Input', 'Restored Output', 'Clean Ground Truth']

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold')

    for row, (deg_path, cln_path, label) in enumerate(samples):
        deg_img = np.array(Image.open(deg_path).convert('RGB'))
        cln_img = np.array(Image.open(cln_path).convert('RGB'))
        rst_img = infer(model, deg_img, device)

        axes[row, 0].imshow(deg_img)
        axes[row, 1].imshow(rst_img)
        axes[row, 2].imshow(cln_img)

        for col in range(3):
            axes[row, col].axis('off')
            axes[row, col].set_aspect('auto')

        axes[row, 0].set_ylabel(label, fontsize=12, fontweight='bold', rotation=90, labelpad=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()