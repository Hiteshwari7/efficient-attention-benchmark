# Implementation and Benchmark of Efficient Attention Mechanisms

### [View the Full Benchmark Notebook](https://github.com/Hiteshwari7/efficient-attention-benchmark/blob/main/main_benchmark.ipynb)

This project implements and benchmarks three Transformer attention mechanisms in PyTorch, with a focus on **true** algorithmic efficiency — not just masking tricks.

## Summary

The standard attention mechanism (Vaswani et al., 2017) has $O(n^2)$ time and memory complexity in sequence length $n$. This quadratic scaling makes it infeasible for long sequences. This project implements and benchmarks two efficient alternatives that avoid materializing the full $n \times n$ score matrix:

| Mechanism | Time | Memory | Key idea |
|---|---|---|---|
| **Full O(n²)** | $O(n^2 \cdot d)$ | $O(n^2)$ | Baseline — full score matrix |
| **Sparse (Top-k)** | $O(n \cdot k \cdot d)$ | $O(n \cdot k \cdot d)$ | `torch.gather` — only k values aggregated |
| **Local (Sliding-Window)** | $O(n \cdot w \cdot d)$ | $O(n \cdot w \cdot d)$ | `F.unfold` — only local window in memory |

## Why this is different from naive masking

A common mistake is to implement "sparse" attention by computing the full $n \times n$ score matrix and then masking most of it to $-\infty$. That approach still allocates $O(n^2)$ memory — the masking happens *after* the damage is done.

This implementation **never materializes the full matrix for value aggregation**:

- **SparseAttention** uses `torch.gather` to extract and aggregate only the top-$k$ value vectors. Peak value-aggregation memory is $O(n \cdot k \cdot d)$.
- **LocalAttention** uses `F.unfold` to extract sliding windows of size $w$, making the score and value tensors $[n, w]$, not $[n, n]$.

The memory difference becomes measurable and dramatic at sequence lengths above 2k. Run the benchmark notebook with `n=4096` and `n=8192` to observe it directly.

## Methodology

1. **Implementation:** All three mechanisms in [`attention_mechanisms.py`](https://github.com/Hiteshwari7/efficient-attention-benchmark/blob/main/attention_mechanisms.py) as `nn.Module` classes with full multi-head support.
2. **Benchmark:** Wall-clock time (ms) **and peak GPU memory (MB)** measured over 10 runs for sequence lengths from 64 to 8192.
3. **Correctness:** Attention heatmaps verify each mechanism attends only where it should.

## Key Findings

### Time Scaling (Log-Log Plot)

On a log-log scale, slope reveals complexity class:
- Full attention slope ≈ 2 → quadratic (every doubling of $n$ quadruples time)
- Sparse and Local slope ≈ 1 → linear in $n$ (every doubling of $n$ doubles time)

### Memory Scaling

This is the most important result. Full attention hits OOM on a standard GPU at $n \approx 4096$–$8192$. Sparse and Local remain well within budget because the $n \times n$ value tensor is **never allocated**.

### Overhead at Short Sequences

At $n < 128$, Full attention is fastest. The efficient methods have fixed overhead (the gather/unfold operations) that outweighs the savings at small scale. This is expected — efficiency methods are designed for long sequences.

## How to Run

```bash
git clone https://github.com/Hiteshwari7/efficient-attention-benchmark.git
cd efficient-attention-benchmark
pip install -r requirements.txt
```

Open `main_benchmark.ipynb` in Jupyter or Google Colab and run all cells. A GPU is recommended to observe the memory measurements; the notebook gracefully handles OOM for full attention at long sequences.

## Implementation Notes

**SparseAttention:** The scoring pass (selecting top-k) still requires computing all $n^2$ scores once for ranking. True sub-quadratic scoring requires approximate methods (LSH as in Reformer, or block-sparse kernels as in BigBird). This implementation demonstrates correct O(n·k) value aggregation, which is the dominant memory cost at large $n$.

**LocalAttention:** Fully sub-quadratic — neither scoring nor aggregation ever sees more than $n \cdot w$ elements. This is the same principle used in Longformer (Beltagy et al., 2020).
