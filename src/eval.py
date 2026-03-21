import argparse, yaml, torch
import numpy as np
from sklearn.metrics import roc_auc_score

from data import make_loaders, LABELS, NUM_CLASSES
from architectures import build_model


def main(cfg, ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test_loader = make_loaders(cfg["data_dir"], cfg["img_size"],
                                     cfg["batch_size"], cfg["num_workers"])
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg["model_name"], cfg["pretrained"], 
                        num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()  # sigmoid for multi-label
            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs  = np.vstack(all_probs)   # (N, NUM_CLASSES)
    all_labels = np.vstack(all_labels)  # (N, NUM_CLASSES)

    # Per-label AUROC
    print(f"\n{'Label':<25} {'AUC':>6}  {'Pos':>6}  {'Total':>6}")
    print("-" * 50)

    aucs = []
    for i, label in enumerate(LABELS):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]
        n_pos  = int(y_true.sum())

        if n_pos == 0 or n_pos == len(y_true):
            # Can't compute AUC if only one class present
            print(f"{label:<25} {'N/A':>6}  {n_pos:>6}  {len(y_true):>6}")
            continue

        auc = roc_auc_score(y_true, y_prob)
        aucs.append(auc)
        print(f"{label:<25} {auc:>6.4f}  {n_pos:>6}  {len(y_true):>6}")

    print("-" * 50)
    print(f"{'Mean AUC':<25} {np.mean(aucs):>6.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    main(cfg, args.ckpt)