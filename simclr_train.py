import os
import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


# ===============================
# FIX SSL (Mac)
# ===============================
ssl._create_default_https_context = ssl._create_unverified_context


# ===============================
# CONFIG
# ===============================
OUTDIR = os.path.expanduser("~/simclr_output_final")
os.makedirs(OUTDIR, exist_ok=True)

DATA_ROOT = os.path.expanduser("~/cifar_data")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

EPOCHS = 10
BATCH = 128
LR = 3e-4
TEMPERATURE = 0.5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"\n🔥 Using device: {DEVICE}")
print(f"📂 Output Folder: {OUTDIR}")
print(f"📂 Dataset Folder: {DATA_ROOT}\n")


# ===============================
# DATASET
# ===============================
class SimCLRTransform:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


base_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

print("\n📥 Loading CIFAR-10...")

train_ssl = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=SimCLRTransform())
train_eval = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=base_transform)
test_data = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=base_transform)

ssl_loader = DataLoader(train_ssl, batch_size=BATCH, shuffle=True)
train_loader = DataLoader(train_eval, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)

print(f"\nDataset Summary:")
print(f" - Training images: {len(train_eval)}")
print(f" - Test images: {len(test_data)}\n")


# ===============================
# MODEL (ResNet18 Encoder)
# ===============================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.normalize(x, dim=1)


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return F.normalize(self.p(x), dim=1)


encoder = Encoder().to(DEVICE)
projector = Projection().to(DEVICE)
optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)


# ===============================
# SAFE NT-XENT LOSS (FP32 Stable)
# ===============================
def nt_xent_loss(z_i, z_j):
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0).float()

    sim = torch.matmul(z, z.T) / TEMPERATURE
    mask = torch.eye(2 * N, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, -1e4)

    positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
    denom = torch.logsumexp(sim, dim=1)

    loss = -positives + denom
    return loss.mean()


# ===============================
# TRAINING
# ===============================
print("\n🚀 Training Started...\n")
loss_history = []

for epoch in range(EPOCHS):
    encoder.train()
    projector.train()
    epoch_loss = 0

    for (x_i, x_j), _ in tqdm(ssl_loader):
        x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)

        h_i = encoder(x_i)
        h_j = encoder(x_j)
        z_i = projector(h_i)
        z_j = projector(h_j)

        loss = nt_xent_loss(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(ssl_loader)
    loss_history.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss = {epoch_loss:.4f}")

np.save(f"{OUTDIR}/loss.npy", np.array(loss_history))
plt.plot(loss_history)
plt.title("SimCLR Loss")
plt.grid(True)
plt.savefig(f"{OUTDIR}/loss_curve.png")
plt.close()

print("\n📉 Loss curve saved\n")


# ===============================
# SAVE MODEL
# ===============================
torch.save(encoder.state_dict(), f"{OUTDIR}/encoder_resnet18_simclr.pth")
print("💾 Encoder Saved\n")


# ===============================
# EMBEDDINGS
# ===============================
def extract(loader):
    encoder.eval()
    E, L = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            E.append(encoder(x).cpu().numpy())
            L.append(y.numpy())
    return np.vstack(E), np.hstack(L)

print("📤 Extracting embeddings...")
train_emb, train_lbl = extract(train_loader)
test_emb, test_lbl = extract(test_loader)

train_emb /= np.linalg.norm(train_emb, axis=1, keepdims=True)
test_emb /= np.linalg.norm(test_emb, axis=1, keepdims=True)

np.save(f"{OUTDIR}/train_emb.npy", train_emb)
np.save(f"{OUTDIR}/test_emb.npy", test_emb)
np.save(f"{OUTDIR}/train_lbl.npy", train_lbl)
np.save(f"{OUTDIR}/test_lbl.npy", test_lbl)
print("✅ Embeddings Saved\n")


# ===============================
# LINEAR PROBE
# ===============================
print("🧪 Linear Evaluation...")

clf = LogisticRegression(max_iter=4000)
clf.fit(train_emb, train_lbl)

pred = clf.predict(test_emb)
acc = accuracy_score(test_lbl, pred)
print(f"\n🎯 Accuracy = {acc*100:.2f}%\n")

cm = confusion_matrix(test_lbl, pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(f"{OUTDIR}/confusion_matrix.png")
plt.close()

print("📊 Confusion Matrix Saved\n")


# ===============================
# PCA + T-SNE
# ===============================
print("🎨 Generating PCA + t-SNE...")

labels = ["airplane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
palette = ["#FF0000","#FF8C00","#FFD700","#32CD32","#00CED1","#1E90FF",
           "#0000CD","#8A2BE2","#FF1493","#696969"]
cmap = ListedColormap(palette)

# PCA
pca = PCA(2)
pca_out = pca.fit_transform(test_emb)

plt.figure(figsize=(10,8))
s = plt.scatter(pca_out[:,0], pca_out[:,1], c=test_lbl, cmap=cmap, s=8)
plt.title("PCA CIFAR-10 Embeddings")
handles = s.legend_elements()[0]
plt.legend(handles, labels, bbox_to_anchor=(1.05,1))
plt.savefig(f"{OUTDIR}/pca_embedding.png", dpi=400)
plt.close()

# TSNE
idx = np.random.choice(len(test_emb), 8000, replace=False)
tsne_out = TSNE(2, perplexity=40).fit_transform(test_emb[idx])

plt.figure(figsize=(10,8))
s = plt.scatter(tsne_out[:,0], tsne_out[:,1], c=test_lbl[idx], cmap=cmap, s=6)
plt.title("t-SNE CIFAR-10 Embeddings")
handles = s.legend_elements()[0]
plt.legend(handles, labels, bbox_to_anchor=(1.05,1))
plt.savefig(f"{OUTDIR}/tsne_embedding.png", dpi=400)
plt.close()

print("\n🌈 DONE! All results saved!")

