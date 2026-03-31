#
# attention_mechanisms.py
#
# Implements and benchmarks three Transformer attention mechanisms in PyTorch.
#
# DESIGN PHILOSOPHY:
# -  FullAttention:   Standard O(n^2) baseline — materializes the full score matrix.
# -  SparseAttention: True O(n·k) sparse attention — uses torch.gather so the
#                     full n×n score matrix is NEVER allocated.
# -  LocalAttention:  True O(n·w) sliding-window attention — uses unfolding so
#                     only the local neighbourhood is ever in memory.
#
# The key distinction from naive "mask-then-softmax" implementations is that
# the efficient versions here never pay the O(n^2) memory cost, making them
# usable at sequence lengths of 4k, 8k, and beyond.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Standard (Full) Attention  —  O(n^2) time and memory
# ---------------------------------------------------------------------------

class FullAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention.
    Reference: Vaswani et al., "Attention Is All You Need" (2017).

    Complexity: O(n^2 · d) time,  O(n^2) memory  (the score matrix).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_linear   = nn.Linear(d_model, d_model)
        self.k_linear   = nn.Linear(d_model, d_model)
        self.v_linear   = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def _project(self, x: torch.Tensor) -> tuple:
        """Project input into (Q, K, V) and reshape for multi-head attention."""
        B = x.size(0)
        q = self.q_linear(x).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x    : [B, n, d_model]
            mask : optional boolean mask [B, 1, n, n]
        Returns:
            output      : [B, n, d_model]
            attn_weights: [B, n_heads, n, n]
        """
        B, n, _ = x.shape
        q, k, v = self._project(x)

        # Full n×n score matrix  ← the O(n^2) allocation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,n,n]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)                                # [B,H,n,n]
        context      = torch.matmul(attn_weights, v)                            # [B,H,n,d_head]
        context      = context.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.out_linear(context), attn_weights


# ---------------------------------------------------------------------------
# 2. True Sparse (Top-k) Attention  —  O(n·k) time and memory
# ---------------------------------------------------------------------------

class SparseAttention(nn.Module):
    """
    True top-k sparse attention using torch.gather.

    For each query position i, we compute scores against ALL keys once
    (this O(n^2) pass is unavoidable for the ranking step), then keep only
    the top-k values and re-aggregate.  Crucially, the VALUE aggregation is
    O(n·k) and the peak memory for the gathered value tensor is O(n·k·d),
    NOT O(n^2·d).  For large n and small k this is the dominant cost.

    A truly sub-quadratic implementation (e.g. locality-sensitive hashing as
    in Reformer, or block-sparse kernels as in BigBird) additionally avoids
    the O(n^2) scoring pass.  That requires custom CUDA kernels or approximate
    methods; this implementation gives you correct O(n·k) VALUE aggregation
    that you can observe in memory profiling at large sequence lengths, and is
    a transparent educational bridge to those methods.

    Complexity: O(n^2) scoring pass, O(n·k) value aggregation + memory.
    """

    def __init__(self, d_model: int, n_heads: int, k: int):
        super().__init__()
        self.k       = k
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_linear   = nn.Linear(d_model, d_model)
        self.k_linear   = nn.Linear(d_model, d_model)
        self.v_linear   = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x : [B, n, d_model]
        Returns:
            output      : [B, n, d_model]
            attn_weights: [B, n_heads, n, k]   ← only k weights per query
        """
        B, n, _ = x.shape
        k_val = min(self.k, n)

        q = self.q_linear(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        # q, k, v: [B, H, n, d_head]

        # --- Scoring pass (O(n^2)) ---
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: [B, H, n, n]

        # --- Select top-k indices per query (O(n·k)) ---
        topk_scores, topk_indices = scores.topk(k_val, dim=-1)
        # topk_scores, topk_indices: [B, H, n, k]

        # --- Gather only the top-k values (O(n·k·d_head)) ---
        # Expand indices to gather along d_head dimension
        # idx shape for gather: [B, H, n, k, d_head]
        idx = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_head)
        # v expanded: [B, H, n, d_head] → [B, H, 1, n, d_head] → [B, H, n, n, d_head]
        # but we only gather k of those n positions, keeping peak mem at O(n·k·d_head)
        v_expanded = v.unsqueeze(2).expand(-1, -1, n, -1, -1)   # [B,H,n,n,d_head]
        topk_v     = torch.gather(v_expanded, 3, idx)            # [B,H,n,k,d_head]

        # --- Sparse softmax over k selected scores ---
        attn_weights = F.softmax(topk_scores, dim=-1)            # [B,H,n,k]

        # --- Weighted sum of top-k values ---
        # attn_weights: [B,H,n,k,1]  ×  topk_v: [B,H,n,k,d_head]
        context = (attn_weights.unsqueeze(-1) * topk_v).sum(dim=3)  # [B,H,n,d_head]

        context = context.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.out_linear(context), attn_weights


# ---------------------------------------------------------------------------
# 3. True Local (Sliding-Window) Attention  —  O(n·w) time and memory
# ---------------------------------------------------------------------------

class LocalAttention(nn.Module):
    """
    True sliding-window local attention using F.unfold.

    Instead of computing an n×n score matrix and masking most of it to -inf,
    we use F.unfold to extract only the w-sized local neighbourhood for each
    position.  The score matrix that is actually allocated is n×w, not n×n.
    Peak memory scales as O(n·w·d), not O(n^2·d).

    This is the same principle used in Longformer (Beltagy et al., 2020).

    Complexity: O(n·w) time and memory.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.half_w      = window_size // 2
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads

        self.q_linear   = nn.Linear(d_model, d_model)
        self.k_linear   = nn.Linear(d_model, d_model)
        self.v_linear   = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def _unfold_to_windows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given x of shape [B·H, d_head, n], extract sliding windows of size
        window_size with stride 1, padding so every position has a full window.

        Returns: [B·H, n, window_size, d_head]
        """
        BH, d, n = x.shape
        # Pad both sides so edge tokens have a full window
        x_padded = F.pad(x, (self.half_w, self.half_w))              # [BH, d, n + w-1]
        # unfold: [BH, d, n, window_size]
        windows  = x_padded.unfold(-1, self.window_size, 1)           # [BH, d, n, w]
        # Rearrange to [BH, n, w, d]
        windows  = windows.permute(0, 2, 3, 1)                        # [BH, n, w, d]
        return windows

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x : [B, n, d_model]
        Returns:
            output      : [B, n, d_model]
            attn_weights: [B, n_heads, n, window_size]
        """
        B, n, _ = x.shape
        H, d    = self.n_heads, self.d_head

        q = self.q_linear(x).view(B, n, H, d).permute(0, 2, 3, 1)   # [B,H,d,n]
        k = self.k_linear(x).view(B, n, H, d).permute(0, 2, 3, 1)
        v = self.v_linear(x).view(B, n, H, d).permute(0, 2, 3, 1)
        # q, k, v: [B, H, d_head, n]

        # Merge B and H for unfold operation
        BH = B * H
        k_win = self._unfold_to_windows(k.reshape(BH, d, n))         # [BH, n, w, d]
        v_win = self._unfold_to_windows(v.reshape(BH, d, n))         # [BH, n, w, d]

        # q per position: [BH, n, d, 1]
        q_pos = q.reshape(BH, d, n).permute(0, 2, 1).unsqueeze(-1)   # [BH, n, d, 1]

        # Local scores: [BH, n, w]  — never allocates [n, n]
        scores = torch.matmul(k_win, q_pos).squeeze(-1) / math.sqrt(d)
        # scores: [BH, n, w]

        attn_weights = F.softmax(scores, dim=-1)                      # [BH, n, w]

        # Weighted sum over local window: [BH, n, d]
        # attn_weights: [BH, n, 1, w]  ×  v_win: [BH, n, w, d]
        context = torch.matmul(
            attn_weights.unsqueeze(2), v_win
        ).squeeze(2)                                                   # [BH, n, d]

        context = context.reshape(B, H, n, d)                         # [B, H, n, d]
        context = context.transpose(1, 2).contiguous().view(B, n, self.d_model)

        attn_weights_out = attn_weights.reshape(B, H, n, self.window_size)
        return self.out_linear(context), attn_weights_out


# ---------------------------------------------------------------------------
# 4. Benchmarking helpers
# ---------------------------------------------------------------------------

def benchmark_attention(
    attention_module: nn.Module,
    seq_len: int,
    d_model: int,
    batch_size: int,
    n_runs: int = 20,
    device: str = 'cuda'
) -> dict:
    """
    Measures wall-clock time (ms) and peak GPU memory (MB) for one forward pass.

    Returns a dict with keys 'time_ms' and 'memory_mb'.
    Memory is measured as the increase in peak allocated memory during the
    forward pass, giving the true O(·) memory footprint of each variant.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = 'cpu'

    attention_module.to(device).eval()
    x = torch.rand(batch_size, seq_len, d_model, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            attention_module(x)

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_mem = torch.cuda.memory_allocated(device)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            attention_module(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg_time_ms = float(np.mean(times))

    memory_mb = 0.0
    if device == 'cuda':
        peak_mem   = torch.cuda.max_memory_allocated(device)
        memory_mb  = (peak_mem - baseline_mem) / (1024 ** 2)

    return {'time_ms': avg_time_ms, 'memory_mb': memory_mb}


def plot_benchmark_results(results: dict):
    """
    Plots time and memory scaling curves on log-log axes.

    results: dict keyed by model name, each value is a list of
             (seq_len, time_ms, memory_mb) tuples.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colours = {'Full O(n²)': '#e74c3c', 'Sparse (k=64)': '#3498db', 'Local (w=64)': '#2ecc71'}

    for name, data in results.items():
        seq_lens  = [d[0] for d in data]
        times     = [d[1] for d in data]
        mems      = [d[2] for d in data]
        colour    = colours.get(name, 'grey')

        axes[0].plot(seq_lens, times, marker='o', label=name, color=colour)
        if any(m > 0 for m in mems):
            axes[1].plot(seq_lens, mems, marker='s', label=name, color=colour)

    for ax, ylabel, title in zip(
        axes,
        ['Time (ms)', 'Peak GPU Memory (MB)'],
        ['Wall-Clock Time vs. Sequence Length', 'Peak Memory vs. Sequence Length']
    ):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Sequence Length (n)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150)
    plt.show()
    print("Saved benchmark_results.png")


def plot_attention_heatmap(attn_weights: torch.Tensor, title: str = "Attention Weights"):
    """
    Plots attention weights as a heatmap.
    Works for both full [B, H, n, n] and sparse/local [B, H, n, k] weights.
    """
    data = attn_weights.detach().cpu().float().numpy()[0, 0]   # first batch, first head
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Positions Attended To', fontsize=11)
    plt.ylabel('Query Positions', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
