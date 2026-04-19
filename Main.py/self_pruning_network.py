"""
Self-Pruning Neural Network for CIFAR-10
=========================================
Tredence Analytics – AI Engineer Case Study

This script implements a neural network that learns to prune itself
*during* training using learnable gate parameters and L1 sparsity regularization.

Author: Candidate Submission
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# PART 1: The "Prunable" Linear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.

    Each weight has an associated scalar gate_score. During the forward
    pass, gate_scores are passed through a Sigmoid to produce gates in
    [0, 1]. Weights are element-wise multiplied by these gates before
    the standard linear operation, allowing the network to learn to
    zero-out (prune) unimportant connections entirely during training.

    Args:
        in_features  (int): Number of input features.
        out_features (int): Number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard learnable parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores – same shape as weight
        # Initialised near zero so initial gates ≈ 0.5 (neutral start)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialise weights with Kaiming uniform (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert gate_scores → gates ∈ (0, 1) via Sigmoid
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Element-wise multiply weights by gates to obtain pruned weights
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Standard linear operation with pruned weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below threshold (effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A feed-forward network for CIFAR-10 image classification built
    entirely from PrunableLinear layers.

    Architecture:
        Input (3072) → 1024 → 512 → 256 → 10
    Each hidden layer uses BatchNorm + ReLU for training stability.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 10)   # output layer

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # flatten: (B, 3072)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))

        x = self.fc4(x)                      # logits
        return x

    def prunable_layers(self):
        """Yield all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 sparsity penalty = sum of all gate values across all
        PrunableLinear layers.  Because gates are positive (sigmoid
        output), the L1 norm equals their sum, naturally encouraging
        them toward zero.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall fraction of weights pruned across the whole network."""
        pruned = total = 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            pruned += (gates < threshold).sum().item()
            total  += gates.numel()
        return pruned / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# PART 2 & 3: Training and Evaluation
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Download CIFAR-10 and return train / test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lambda_sparse):
    """Run one training epoch; returns (avg_total_loss, avg_clf_loss, avg_sparse_loss)."""
    model.train()
    total_loss_sum = clf_loss_sum = sparse_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Classification loss
        clf_loss    = F.cross_entropy(logits, labels)

        # Sparsity regularisation (L1 on sigmoid gates)
        sparse_loss = model.sparsity_loss()

        # Total loss
        loss = clf_loss + lambda_sparse * sparse_loss
        loss.backward()
        optimizer.step()

        n = images.size(0)
        total_loss_sum  += loss.item()        * n
        clf_loss_sum    += clf_loss.item()    * n
        sparse_loss_sum += sparse_loss.item() * n

    N = len(loader.dataset)
    return total_loss_sum / N, clf_loss_sum / N, sparse_loss_sum / N


@torch.no_grad()
def evaluate(model, loader, device):
    """Return top-1 accuracy on loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def run_experiment(lambda_sparse: float,
                   epochs: int,
                   train_loader,
                   test_loader,
                   device,
                   verbose: bool = True):
    """
    Train a fresh SelfPruningNet with the given lambda and return a
    results dict containing accuracy and sparsity.
    """
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'total_loss': [], 'clf_loss': [], 'sparse_loss': [],
               'train_acc': [], 'test_acc': [], 'sparsity': []}

    print(f"\n{'='*60}")
    print(f"  Training with λ = {lambda_sparse}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        tl, cl, sl = train_one_epoch(model, train_loader, optimizer, device, lambda_sparse)
        tr_acc  = evaluate(model, train_loader, device)
        te_acc  = evaluate(model, test_loader,  device)
        sparsity = model.global_sparsity()
        scheduler.step()

        history['total_loss'].append(tl)
        history['clf_loss'].append(cl)
        history['sparse_loss'].append(sl)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['sparsity'].append(sparsity)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss {tl:.4f} (clf {cl:.4f}, sparse {sl:.2f}) | "
                  f"Train {tr_acc*100:.2f}% | Test {te_acc*100:.2f}% | "
                  f"Sparsity {sparsity*100:.1f}%")

    final_acc      = te_acc
    final_sparsity = sparsity

    return {
        'lambda'   : lambda_sparse,
        'model'    : model,
        'history'  : history,
        'test_acc' : final_acc,
        'sparsity' : final_sparsity,
    }


def plot_gate_distribution(model, lambda_val: float, save_path: str):
    """Histogram of all gate values across the entire network."""
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.get_gates().cpu().numpy().flatten())
    all_gates = np.concatenate(all_gates)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_gates, bins=80, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(x=0.01, color='crimson', linestyle='--', label='Prune threshold (0.01)')
    ax.set_xlabel('Gate Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of Gate Values  (λ = {lambda_val})', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Gate distribution plot saved to: {save_path}")


def plot_training_curves(results_list, save_path: str):
    """Sparsity and test accuracy over epochs for all lambda values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for res in results_list:
        lam   = res['lambda']
        hist  = res['history']
        label = f"λ={lam}"
        epochs = range(1, len(hist['test_acc']) + 1)
        axes[0].plot(epochs, [s * 100 for s in hist['sparsity']],  label=label)
        axes[1].plot(epochs, [a * 100 for a in hist['test_acc']], label=label)

    axes[0].set_title('Sparsity Level Over Training', fontsize=13)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Sparsity (%)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title('Test Accuracy Over Training', fontsize=13)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Training curves saved to: {save_path}")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    EPOCHS     = 30          # increase to 50–100 for better accuracy
    BATCH_SIZE = 128
    LAMBDAS    = [1e-5, 1e-4, 1e-3]   # low, medium, high sparsity pressure

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results_list = []
    for lam in LAMBDAS:
        res = run_experiment(lam, EPOCHS, train_loader, test_loader, device)
        results_list.append(res)

    # ── Summary Table ────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>16} {'Sparsity':>14}")
    print("="*55)
    for res in results_list:
        print(f"  {res['lambda']:<12} "
              f"{res['test_acc']*100:>14.2f}%  "
              f"{res['sparsity']*100:>12.1f}%")
    print("="*55)

    # ── Plots ────────────────────────────────────────────────────
    # Gate distribution for best-accuracy model
    best = max(results_list, key=lambda r: r['test_acc'])
    plot_gate_distribution(best['model'], best['lambda'], 'gate_distribution.png')

    # Training curves for all lambdas
    plot_training_curves(results_list, 'training_curves.png')

    print("\nDone!  Check gate_distribution.png and training_curves.png.")


if __name__ == '__main__':
    main()
