"""
Logic Gates PID Validation Test

Trains tiny transformers on canonical PID logic gates and verifies that
PID computation recovers the expected information decomposition.

Logic gates tested:
- AND:  Y = X1 AND X2     → High synergy (need both inputs)
- XOR:  Y = X1 XOR X2     → Maximum synergy
- OR:   Y = X1 OR X2      → High redundancy (either input suffices)
- COPY: Y = X1 (X2 = X1)  → Maximum redundancy (identical inputs)
- UNIQ: Y = X1 (X2 random)→ Unique(X1) only

This validates that PID atoms respect information-theoretic bounds:
- All atoms sum to I(Y; X1, X2) <= H(Y)
- For binary Y with known distribution, H(Y) <= 1 bit
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

pytestmark = pytest.mark.slow

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

SEED = 42


@dataclass
class Config:
    vocab_size: int = 4
    embed_dim: int = 32
    num_heads: int = 2
    num_layers: int = 2
    max_seq_len: int = 8
    n_train: int = 10000
    n_test: int = 1000
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    SEP: int = 2
    PAD: int = 3


def generate_gate_data(gate_type: str, n_samples: int, config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate data for a specific logic gate."""
    x1 = np.random.randint(0, 2, n_samples)
    x2_base = np.random.randint(0, 2, n_samples)

    if gate_type == "AND":
        x2 = x2_base
        y = (x1 & x2).astype(int)
    elif gate_type == "OR":
        x2 = x2_base
        y = (x1 | x2).astype(int)
    elif gate_type == "XOR":
        x2 = x2_base
        y = (x1 ^ x2).astype(int)
    elif gate_type == "COPY":
        x2 = x1.copy()
        y = x1.copy()
    elif gate_type == "UNIQ":
        x2 = x2_base
        y = x1.copy()
    else:
        raise ValueError(f"Unknown gate: {gate_type}")

    sequences = np.stack([x1, np.full(n_samples, config.SEP),
                          x2, np.full(n_samples, config.SEP)], axis=1)

    return torch.tensor(sequences, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TinyTransformer(nn.Module):
    """Minimal causal transformer for sequence-to-token prediction."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_head = nn.Linear(config.embed_dim, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.transformer(h, mask=mask)
        h_last = h[:, -1, :]
        logits = self.output_head(h_last)
        return logits

    def get_log_probs(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        logits = self.forward(x, mask)
        return torch.log_softmax(logits, dim=-1)


def train_model(model: TinyTransformer, train_data: TensorDataset, config: Config) -> None:
    """Train the model on a logic gate task."""
    model.to(DEVICE)
    model.train()

    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()


def evaluate_accuracy(model: TinyTransformer, test_data: TensorDataset) -> float:
    """Evaluate model accuracy."""
    model.eval()
    loader = DataLoader(test_data, batch_size=256)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def create_attention_mask(seq_len: int, mask_positions: List[int], device) -> torch.Tensor:
    """Create attention mask where specified positions are blocked."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    for pos in mask_positions:
        mask[:, pos] = True
    return mask


def compute_pid_for_gate(model: TinyTransformer, test_data: TensorDataset, config: Config) -> Dict[str, float]:
    """
    Compute PID atoms using true marginal p(Y) as baseline.

    I(Y; X) = H(Y) - H(Y|X) = H(Y) + E[log p(Y|X)]
    """
    model.eval()
    loader = DataLoader(test_data, batch_size=256)

    X1_POS, X2_POS = 0, 2
    seq_len = 4

    # Compute H(Y) from marginal distribution
    all_y = []
    for _, y in loader:
        all_y.append(y)
    all_y = torch.cat(all_y)
    p_y1 = all_y.float().mean().item()
    p_y0 = 1 - p_y1
    h_y = -p_y0 * math.log2(p_y0) - p_y1 * math.log2(p_y1) if 0 < p_y1 < 1 else 0.0

    total_ll_full = 0.0
    total_ll_x1 = 0.0
    total_ll_x2 = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            batch_size = x.size(0)

            log_probs_full = model.get_log_probs(x, mask=None)
            total_ll_full += log_probs_full[torch.arange(batch_size), y].sum().item()

            mask_x2 = create_attention_mask(seq_len, [X2_POS], x.device)
            log_probs_x1 = model.get_log_probs(x, mask=mask_x2)
            total_ll_x1 += log_probs_x1[torch.arange(batch_size, device=x.device), y].sum().item()

            mask_x1 = create_attention_mask(seq_len, [X1_POS], x.device)
            log_probs_x2 = model.get_log_probs(x, mask=mask_x1)
            total_ll_x2 += log_probs_x2[torch.arange(batch_size, device=x.device), y].sum().item()

            total_samples += batch_size

    log2 = math.log(2)

    avg_ll_full = (total_ll_full / total_samples) / log2
    avg_ll_x1 = (total_ll_x1 / total_samples) / log2
    avg_ll_x2 = (total_ll_x2 / total_samples) / log2

    i_y_x1x2 = max(0, h_y + avg_ll_full)
    i_y_x1 = max(0, h_y + avg_ll_x1)
    i_y_x2 = max(0, h_y + avg_ll_x2)

    redundancy = min(i_y_x1, i_y_x2)
    unique_x1 = i_y_x1 - redundancy
    unique_x2 = i_y_x2 - redundancy
    synergy = i_y_x1x2 - unique_x1 - unique_x2 - redundancy

    return {
        "h_y": h_y,
        "i_y_x1x2": i_y_x1x2,
        "i_y_x1": i_y_x1,
        "i_y_x2": i_y_x2,
        "redundancy": redundancy,
        "unique_x1": unique_x1,
        "unique_x2": unique_x2,
        "synergy": synergy,
    }


# Theoretical expectations
GATES = {
    "XOR": {"synergy": "high", "redundancy": "zero", "unique": "zero"},
    "AND": {"synergy": "medium", "redundancy": "medium", "unique": "zero"},
    "OR": {"synergy": "medium", "redundancy": "medium", "unique": "zero"},
    "COPY": {"synergy": "zero", "redundancy": "high", "unique": "zero"},
    "UNIQ": {"synergy": "zero", "redundancy": "zero", "unique": "high"},
}


@pytest.fixture(scope="module")
def config():
    return Config()


@pytest.fixture(scope="module")
def trained_models(config):
    """Train models for all gates once per test session."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    models = {}
    test_datasets = {}

    for gate in GATES:
        train_x, train_y = generate_gate_data(gate, config.n_train, config)
        test_x, test_y = generate_gate_data(gate, config.n_test, config)

        train_data = TensorDataset(train_x, train_y)
        test_data = TensorDataset(test_x, test_y)

        model = TinyTransformer(config)
        train_model(model, train_data, config)

        models[gate] = model
        test_datasets[gate] = test_data

    return models, test_datasets


class TestLogicGatesPID:
    """Test suite for validating PID on logic gates."""

    def test_models_learn_gates(self, config, trained_models):
        """All models should achieve >95% accuracy."""
        models, test_datasets = trained_models

        print("\n" + "=" * 60)
        print("MODEL ACCURACY")
        print("=" * 60)

        for gate in GATES:
            accuracy = evaluate_accuracy(models[gate], test_datasets[gate])
            print(f"  {gate}: {accuracy:.4f}")
            assert accuracy > 0.95, f"{gate} model accuracy too low: {accuracy}"

    def test_pid_respects_entropy_bound(self, config, trained_models):
        """PID atoms should sum to <= H(Y)."""
        models, test_datasets = trained_models

        print("\n" + "=" * 60)
        print("PID ENTROPY BOUNDS")
        print("=" * 60)

        for gate in GATES:
            pid = compute_pid_for_gate(models[gate], test_datasets[gate], config)
            pid_sum = pid["redundancy"] + pid["unique_x1"] + pid["unique_x2"] + pid["synergy"]

            print(f"  {gate}: H(Y)={pid['h_y']:.3f}, Sum={pid_sum:.3f}")

            # PID atoms should sum to approximately I(Y; X1, X2) <= H(Y)
            assert pid_sum <= pid["h_y"] + 0.1, f"{gate}: PID sum {pid_sum} exceeds H(Y) {pid['h_y']}"
            assert pid["i_y_x1x2"] <= pid["h_y"] + 0.1, f"{gate}: I(Y;X1,X2) exceeds H(Y)"

    def test_xor_has_maximum_synergy(self, config, trained_models):
        """XOR should have synergy close to 1 bit and no redundancy."""
        models, test_datasets = trained_models
        pid = compute_pid_for_gate(models["XOR"], test_datasets["XOR"], config)

        print("\n" + "=" * 60)
        print("XOR PID (Expected: Syn~1, Red~0)")
        print("=" * 60)
        print(f"  Synergy:    {pid['synergy']:.4f}")
        print(f"  Redundancy: {pid['redundancy']:.4f}")
        print(f"  Unique(X1): {pid['unique_x1']:.4f}")
        print(f"  Unique(X2): {pid['unique_x2']:.4f}")

        assert pid["synergy"] > 0.8, f"XOR synergy too low: {pid['synergy']}"
        assert pid["redundancy"] < 0.2, f"XOR redundancy too high: {pid['redundancy']}"

    @pytest.mark.xfail(reason="Attention masking with identical inputs: model learns positional shortcut, attending only to X1 position")
    def test_copy_has_maximum_redundancy(self, config, trained_models):
        """COPY should have redundancy close to 1 bit and no synergy."""
        models, test_datasets = trained_models
        pid = compute_pid_for_gate(models["COPY"], test_datasets["COPY"], config)

        print("\n" + "=" * 60)
        print("COPY PID (Expected: Red~1, Syn~0)")
        print("=" * 60)
        print(f"  Redundancy: {pid['redundancy']:.4f}")
        print(f"  Synergy:    {pid['synergy']:.4f}")
        print(f"  Unique(X1): {pid['unique_x1']:.4f}")
        print(f"  Unique(X2): {pid['unique_x2']:.4f}")

        assert pid["redundancy"] > 0.8, f"COPY redundancy too low: {pid['redundancy']}"
        assert abs(pid["synergy"]) < 0.2, f"COPY synergy too high: {pid['synergy']}"

    def test_uniq_has_unique_information(self, config, trained_models):
        """UNIQ should have unique(X1) close to 1 bit."""
        models, test_datasets = trained_models
        pid = compute_pid_for_gate(models["UNIQ"], test_datasets["UNIQ"], config)

        print("\n" + "=" * 60)
        print("UNIQ PID (Expected: Uniq1~1, others~0)")
        print("=" * 60)
        print(f"  Unique(X1): {pid['unique_x1']:.4f}")
        print(f"  Unique(X2): {pid['unique_x2']:.4f}")
        print(f"  Redundancy: {pid['redundancy']:.4f}")
        print(f"  Synergy:    {pid['synergy']:.4f}")

        assert pid["unique_x1"] > 0.8, f"UNIQ unique_x1 too low: {pid['unique_x1']}"
        assert abs(pid["synergy"]) < 0.2, f"UNIQ synergy too high: {pid['synergy']}"

    def test_full_summary(self, config, trained_models):
        """Print full summary of all gates."""
        models, test_datasets = trained_models

        print("\n" + "=" * 80)
        print("FULL PID SUMMARY")
        print("=" * 80)
        print(f"{'Gate':<8} {'H(Y)':<7} {'I(Y;X)':<7} {'Red':<7} {'Unq1':<7} {'Unq2':<7} {'Syn':<7} {'Sum':<7}")
        print("-" * 80)

        for gate in GATES:
            pid = compute_pid_for_gate(models[gate], test_datasets[gate], config)
            pid_sum = pid["redundancy"] + pid["unique_x1"] + pid["unique_x2"] + pid["synergy"]

            print(f"{gate:<8} {pid['h_y']:<7.3f} {pid['i_y_x1x2']:<7.3f} "
                  f"{pid['redundancy']:<7.3f} {pid['unique_x1']:<7.3f} "
                  f"{pid['unique_x2']:<7.3f} {pid['synergy']:<7.3f} {pid_sum:<7.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
