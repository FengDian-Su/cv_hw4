import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys

from dataset import RainSnowTrainDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PromptIR'))
from net.model import PromptIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',   type=str, default='dataset/train')
    parser.add_argument('--ckpt_dir',    type=str, default='checkpoints')
    parser.add_argument('--epochs',      type=int, default=150)
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--patch_size',  type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--val_split',   type=float, default=0.05)
    parser.add_argument('--resume',      type=str, default=None)
    parser.add_argument('--gpu',         type=str, default='0,1')
    return parser.parse_args()


def compute_psnr(pred, target):
    mse = torch.mean((pred.clamp(0, 1) - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


def warmup_cosine_lr(optimizer, epoch, warmup_epochs, max_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        import math
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f'  Saved checkpoint: {path}')


def init_log(log_path):
    """Create CSV log file with header (only if it doesn't exist)."""
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_psnr'])


def append_log(log_path, epoch, lr, train_loss, val_loss, val_psnr):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f'{lr:.6f}', f'{train_loss:.6f}',
                         f'{val_loss:.6f}', f'{val_psnr:.4f}'])


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_multi_gpu = torch.cuda.device_count() > 1
    print(f'Using device: {device}  |  GPUs: {torch.cuda.device_count()}')

    # Dataset
    full_dataset = RainSnowTrainDataset(args.data_root, patch_size=args.patch_size)
    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f'Train: {len(train_set)}  |  Val: {len(val_set)}')

    # Model
    model = PromptIR(decoder=True).to(device)
    if use_multi_gpu:
        model = nn.DataParallel(model)

    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    start_epoch = 0
    best_psnr   = 0.0

    # Log file
    log_path = os.path.join(args.ckpt_dir, 'train_log.csv')
    init_log(log_path)

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', 0.0)
        print(f'Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        lr = warmup_cosine_lr(optimizer, epoch, warmup_epochs=15,
                              max_epochs=args.epochs, base_lr=args.lr)

        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{args.epochs}] lr={lr:.2e}')
        for deg, cln in pbar:
            deg, cln = deg.to(device), cln.to(device)
            optimizer.zero_grad()
            with autocast():
                restored = model(deg)
                loss = loss_fn(restored, cln)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_psnr = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for deg, cln in val_loader:
                deg, cln = deg.to(device), cln.to(device)
                with autocast():
                    restored = model(deg)
                    loss = loss_fn(restored, cln)
                val_loss += loss.item()
                val_psnr += compute_psnr(restored, cln).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)

        print(f'Epoch {epoch+1:03d} | '
              f'lr={lr:.2e} | '
              f'train_loss: {avg_train_loss:.4f} | '
              f'val_loss: {avg_val_loss:.4f} | '
              f'val_psnr: {avg_val_psnr:.2f} dB')

        # Write to CSV
        append_log(log_path, epoch+1, lr, avg_train_loss, avg_val_loss, avg_val_psnr)

        # Save checkpoints
        state = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
        }
        save_checkpoint(state, os.path.join(args.ckpt_dir, 'last.pth'))

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            state['best_psnr'] = best_psnr
            save_checkpoint(state, os.path.join(args.ckpt_dir, 'best.pth'))
            print(f'  New best PSNR: {best_psnr:.2f} dB')

        if (epoch + 1) % 10 == 0:
            save_checkpoint(state, os.path.join(args.ckpt_dir, f'epoch_{epoch+1:03d}.pth'))

    print(f'\nTraining complete. Best val PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()