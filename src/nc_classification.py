import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('binned_tifs')
TEST_EMBRYOS = ['inner.i04channel_638_patterns', 'off_11channel_638patterns']
CNN_MODEL_PATH = Path('cnn_best.pt')
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f'Device: {DEVICE}')

class EmbryoDataset(Dataset):
    def __init__(self, root, img_size=(224, 224)):
        self.img_size = img_size
        self.samples = []
        for embryo_dir in sorted(Path(root).iterdir()):
            if not embryo_dir.is_dir():
                continue
            for cls_name, label in [('pre-nc10', 0), ('post-nc10', 1)]:
                cls_dir = embryo_dir / cls_name
                if not cls_dir.exists():
                    continue
                for f in sorted(cls_dir.glob('*.tif*')):
                    self.samples.append((f, label, embryo_dir.name))
        if not self.samples:
            raise RuntimeError(f'No TIFs found under {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        raw = tiff.imread(str(path))
        img = raw[0] if raw.ndim == 3 else raw
        img = img.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=self.img_size,
            mode='bilinear', align_corners=False
        ).squeeze(0)
        return tensor, label

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_embryo_ids(self):
        return [s[2] for s in self.samples]


class AugSubset(Dataset):
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment
        self.tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]) if augment else nn.Identity()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.tf(x), y


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    w = m.conv1.weight.mean(dim=1, keepdim=True)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight = nn.Parameter(w)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    all_probs, all_labels, total_loss = [], [], 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y_batch in loader:
            x = x.to(DEVICE)
            y_batch = y_batch.float().to(DEVICE)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y_batch)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())
            total_loss += loss.item() * len(y_batch)
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
    ba = balanced_accuracy_score(labels, (probs >= 0.5).astype(int))
    return total_loss / len(labels), auc, ba


# Build dataset and split
dataset = EmbryoDataset(DATA_DIR)
labels_arr = np.array(dataset.get_labels())
emb_arr = np.array(dataset.get_embryo_ids())

train_idx = [i for i, s in enumerate(dataset.samples) if s[2] not in TEST_EMBRYOS]
test_idx = [i for i, s in enumerate(dataset.samples) if s[2] in TEST_EMBRYOS]

print(f'Train frames: {len(train_idx)} ({len(set(emb_arr[train_idx]))} embryos)')
print(f'Test frames: {len(test_idx)} (embryos: {TEST_EMBRYOS})')
print(f'Train pre: {(labels_arr[train_idx]==0).sum()} post: {(labels_arr[train_idx]==1).sum()}')
print(f'Test pre: {(labels_arr[test_idx]==0).sum()} post: {(labels_arr[test_idx]==1).sum()}\n')

train_dl = DataLoader(AugSubset(Subset(dataset, train_idx), augment=True),
                      batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
test_dl = DataLoader(AugSubset(Subset(dataset, test_idx), augment=False),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# Training
model = build_model()
n_pos = int(labels_arr[train_idx].sum())
n_neg = len(train_idx) - n_pos
pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_auc, tr_ba = run_epoch(model, train_dl, criterion, optimizer)
    scheduler.step()
    if epoch % 5 == 0:
        print(f'Ep {epoch:2d}: loss {tr_loss:.3f}, auc {tr_auc:.3f}, bal_acc {tr_ba:.3f}')

# Testing
model.eval()
all_probs, all_labels = [], []

with torch.no_grad():
    for x, y_batch in test_dl:
        probs = torch.sigmoid(model(x.to(DEVICE)).squeeze(1))
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(y_batch.tolist())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
preds = (all_probs >= 0.5).astype(int)

print(f'\nTest results: {TEST_EMBRYOS}')
print(f'ROC-AUC: {roc_auc_score(all_labels, all_probs):.3f}')
print(f'Balanced accuracy: {balanced_accuracy_score(all_labels, preds):.3f}')

cm = confusion_matrix(all_labels, preds)
ConfusionMatrixDisplay(cm, display_labels=['pre-nc10', 'post-nc10']).plot(cmap='Blues')
plt.title(f'Test set confusion matrix')
plt.show()

torch.save(model.state_dict(), CNN_MODEL_PATH)
print(f'Model saved to {CNN_MODEL_PATH}')