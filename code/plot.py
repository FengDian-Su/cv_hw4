import csv
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help='path to train_log.csv')
    parser.add_argument('--output', type=str, default='training_curves.png')
    return parser.parse_args()


def main():
    args = parse_args()

    epochs, train_loss, val_loss, val_psnr = [], [], [], []
    with open(args.log) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            val_psnr.append(float(row['val_psnr']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves', fontsize=14)

    # Loss
    ax1.plot(epochs, train_loss, label='Train Loss', color='steelblue')
    ax1.plot(epochs, val_loss,   label='Val Loss',   color='tomato')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('L1 Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PSNR
    best_epoch = epochs[val_psnr.index(max(val_psnr))]
    best_psnr  = max(val_psnr)
    ax2.plot(epochs, val_psnr, label='Val PSNR', color='seagreen')
    ax2.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {best_psnr:.2f} dB\n(epoch {best_epoch})',
                 xy=(best_epoch, best_psnr),
                 xytext=(best_epoch + len(epochs)*0.05, best_psnr - 1.0),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=9)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Validation PSNR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()