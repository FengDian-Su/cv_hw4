import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from dataset import RainSnowTestDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PromptIR'))
from net.model import PromptIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, default='dataset/test')
    parser.add_argument('--ckpt',      type=str, required=True)
    parser.add_argument('--output',    type=str, default='pred.npz')
    parser.add_argument('--gpu',       type=str, default='0')
    parser.add_argument('--tta',       action='store_true', help='Enable Test Time Augmentation')
    return parser.parse_args()


def tta_inference(model, img):
    """
    4 augmentations: original + hflip + vflip + hflip&vflip
    Rotation 捨棄，避免還原順序錯誤。
    """
    transforms = [
        (lambda x: x,                          lambda x: x),                          # 0: original
        (lambda x: x.flip(-1),                 lambda x: x.flip(-1)),                 # 1: hflip
        (lambda x: x.flip(-2),                 lambda x: x.flip(-2)),                 # 2: vflip
        (lambda x: x.flip(-1).flip(-2),        lambda x: x.flip(-2).flip(-1)),        # 3: hflip+vflip
    ]

    outputs = []
    for aug_fn, deaug_fn in transforms:
        aug_img = aug_fn(img)
        with torch.no_grad():
            out = model(aug_img)
        out = deaug_fn(out)
        outputs.append(out)

    return torch.stack(outputs, dim=0).mean(dim=0)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PromptIR(decoder=True).to(device)
    print(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)

    state_dict = ckpt['model']
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if args.tta:
        print('TTA enabled (4 augmentations: original, hflip, vflip, hflip+vflip)')

    dataset = RainSnowTestDataset(args.test_root)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    results = {}
    with torch.no_grad():
        for name_list, deg_tensor in tqdm(loader, desc='Inferring'):
            name = name_list[0]
            deg_tensor = deg_tensor.to(device)

            if args.tta:
                restored = tta_inference(model, deg_tensor)
            else:
                restored = model(deg_tensor)

            restored = restored.clamp(0, 1)
            img_np = (restored.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            results[name] = img_np

    np.savez(args.output, **results)
    print(f'Saved {len(results)} images to {args.output}')


if __name__ == '__main__':
    main()