import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF


class RainSnowTrainDataset(Dataset):
    """Training dataset for rain and snow image restoration."""

    def __init__(self, data_root, patch_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.toTensor = ToTensor()
        self.samples = []  # list of (degraded_path, clean_path)

        degraded_dir = os.path.join(data_root, 'degraded')
        clean_dir = os.path.join(data_root, 'clean')

        for deg_type in ['rain', 'snow']:
            for i in range(1, 1601):
                deg_path = os.path.join(degraded_dir, f'{deg_type}-{i}.png')
                cln_path = os.path.join(clean_dir, f'{deg_type}_clean-{i}.png')
                if os.path.exists(deg_path) and os.path.exists(cln_path):
                    self.samples.append((deg_path, cln_path))

        print(f'Total training samples: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        deg_path, cln_path = self.samples[idx]

        deg_img = np.array(Image.open(deg_path).convert('RGB'))
        cln_img = np.array(Image.open(cln_path).convert('RGB'))

        # Random crop
        H, W, _ = deg_img.shape
        ps = self.patch_size
        if H > ps and W > ps:
            i = random.randint(0, H - ps)
            j = random.randint(0, W - ps)
            deg_img = deg_img[i:i+ps, j:j+ps]
            cln_img = cln_img[i:i+ps, j:j+ps]

        # Random horizontal flip
        if random.random() > 0.5:
            deg_img = np.fliplr(deg_img).copy()
            cln_img = np.fliplr(cln_img).copy()

        # Random vertical flip
        if random.random() > 0.5:
            deg_img = np.flipud(deg_img).copy()
            cln_img = np.flipud(cln_img).copy()

        deg_tensor = self.toTensor(deg_img)
        cln_tensor = self.toTensor(cln_img)

        return deg_tensor, cln_tensor


class RainSnowTestDataset(Dataset):
    """Test dataset — loads degraded images only."""

    def __init__(self, test_root):
        super().__init__()
        self.toTensor = ToTensor()
        self.degraded_dir = os.path.join(test_root, 'degraded')

        self.names = sorted(
            [f for f in os.listdir(self.degraded_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('.')[0])
        )
        print(f'Total test images: {len(self.names)}')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = os.path.join(self.degraded_dir, name)
        img = np.array(Image.open(path).convert('RGB'))
        tensor = self.toTensor(img)
        return name, tensor