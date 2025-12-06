# GPT 模型结构与 GPU 负载分析

## 目标
了解 GPT.py 模型结构，分析其产生的 GPU 负载类型。

---

## 1. 模型架构概览

### 1.1 模型类型
**Decoder-only Transformer**（类似 GPT-2）

### 1.2 关键超参数

```python
emb_size = 128        # 嵌入维度 / 隐藏维度
head_size = 8         # 每个注意力头的维度
n_layer = 12          # Transformer 层数
sequence_len = 128    # 序列长度
batch_size = 250      # 批大小

# 计算得出
n_head = emb_size // head_size = 128 // 8 = 16  # 注意力头数
ff_hidden = 4 * emb_size = 512                   # FFN 中间维度
```

### 1.3 模型结构图

```
CharGPT
│
├── token_embedding: Embedding(vocab_size, 128)
├── position_embedding: Embedding(128, 128)
│
├── blocks: Sequential(12 x Block)
│   │
│   └── Block (重复 12 次)
│       ├── ln1: LayerNorm(128)
│       ├── mha: MaskedMultiHeadAttention
│       │   ├── heads: ModuleList(16 x MaskedAttention)
│       │   │   └── MaskedAttention
│       │   │       ├── query: Linear(128, 8)
│       │   │       ├── key: Linear(128, 8)
│       │   │       ├── value: Linear(128, 8)
│       │   │       └── dropout
│       │   ├── proj: Linear(128, 128)
│       │   └── dropout
│       │
│       ├── ln2: LayerNorm(128)
│       └── ff: FeedForward
│           ├── l1: Linear(128, 512)
│           ├── gelu
│           ├── l2: Linear(512, 128)
│           └── dropout
│
├── ln: LayerNorm(128)
└── lm_head: Linear(128, vocab_size)
```

---

## 2. Forward 中的 PyTorch 操作

### 2.1 主要操作分解

```python
def forward(self, x):  # x: (B, T) = (batch, seq_len)
    
    # 1. Embedding 查表
    tok_emb = self.token_embedding(x)       # (B, T, C)
    pos_emb = self.position_embedding(pos)  # (T, C)
    x = tok_emb + pos_emb                   # 广播加法
    
    # 2. 12 层 Transformer Block
    for block in self.blocks:
        # 2.1 LayerNorm
        x_norm = block.ln1(x)
        
        # 2.2 Multi-Head Attention (16 头)
        for head in heads:
            q = head.query(x_norm)   # Linear: (B,T,128) → (B,T,8)
            k = head.key(x_norm)     # Linear
            v = head.value(x_norm)   # Linear
            
            # Attention 计算
            scores = q @ k.T / sqrt(8)  # Matmul: (B,T,8) @ (B,8,T) → (B,T,T)
            scores = masked_fill(scores, mask, -inf)
            w = softmax(scores)         # Softmax
            w = dropout(w)
            out = w @ v                 # Matmul: (B,T,T) @ (B,T,8) → (B,T,8)
        
        # 2.3 Concat + Projection
        concat_out = cat([head_out for head in heads], dim=-1)  # (B,T,128)
        attn_out = block.mha.proj(concat_out)                   # Linear
        attn_out = dropout(attn_out)
        
        # 2.4 残差连接
        x = x + attn_out
        
        # 2.5 LayerNorm + FFN
        x_norm = block.ln2(x)
        ff_out = gelu(block.ff.l1(x_norm))  # Linear: 128→512, GELU
        ff_out = block.ff.l2(ff_out)         # Linear: 512→128
        ff_out = dropout(ff_out)
        
        # 2.6 残差连接
        x = x + ff_out
    
    # 3. 最后的 LayerNorm + LM Head
    x = self.ln(x)
    logits = self.lm_head(x)  # Linear: 128 → vocab_size
    
    return logits
```

---

## 3. GPU Kernel 映射

### 3.1 操作 → Kernel 对应表

| PyTorch 操作 | 底层 Kernel/库 | 调用次数/层 |
|-------------|---------------|-------------|
| `nn.Embedding` | `index_select` kernel | 2 (token + pos) |
| `nn.Linear` | cuBLAS GEMM | 3×16+2+2 = 52/层 |
| `@` (matmul) | cuBLAS GEMM | 2×16 = 32/层 |
| `nn.LayerNorm` | LayerNorm kernel | 3/层 |
| `F.softmax` | Softmax kernel | 16/层 |
| `F.gelu` | GELU kernel | 1/层 |
| `F.dropout` | Dropout kernel | 多次/层 |
| `masked_fill` | Element-wise kernel | 16/层 |
| `+` (add) | Element-wise add | 多次/层 |
| `torch.cat` | Concat kernel | 1/层 |

### 3.2 每层的主要 Kernel 统计

```
每个 Block (共 12 个):
├── Linear (GEMM): ~52 次
│   ├── Q/K/V projection: 16×3 = 48
│   ├── Output projection: 1
│   ├── FFN layer1: 1
│   └── FFN layer2: 1
│
├── Matmul (GEMM): 32 次
│   ├── Q @ K.T: 16
│   └── Attn @ V: 16
│
├── LayerNorm: 2 次
├── Softmax: 16 次
├── GELU: 1 次
└── Dropout/Add/Mask: 多次

总计 (12 层):
  GEMM: ~1000 次
  LayerNorm: ~24 次
  Softmax: ~192 次
  Element-wise: ~数百次
```

---

## 4. Compute-bound vs Memory-bound 分析

### 4.1 分类标准

```
Compute-bound: 计算密集，算术强度高
  - 算术强度 = FLOPs / Bytes (内存访问量)
  - 特点：GPU 计算单元忙碌，内存带宽空闲

Memory-bound: 内存密集，带宽受限
  - 特点：计算单元空闲，等待数据从显存加载
```

### 4.2 各操作分类

| 操作 | 类型 | 原因 |
|------|------|------|
| **Linear (GEMM)** | **Compute-bound** | 大矩阵乘法，O(n³) 计算 vs O(n²) 内存 |
| **Matmul (Attention)** | **Compute-bound** | 同上，但矩阵较小时可能偏 memory |
| LayerNorm | **Memory-bound** | 逐元素操作，计算简单 |
| Softmax | **Memory-bound** | 逐行操作，需要多次遍历 |
| GELU | **Memory-bound** | 逐元素激活函数 |
| Dropout | **Memory-bound** | 逐元素随机 mask |
| Add (残差) | **Memory-bound** | 逐元素加法 |
| Embedding | **Memory-bound** | 查表，纯内存访问 |
| Concat | **Memory-bound** | 内存拷贝 |

### 4.3 比例估算

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT Forward Pass                          │
│                                                              │
│  Compute-bound (GEMM):  ████████████████████████  ~60%      │
│  - Linear layers                                             │
│  - Attention matmul                                          │
│                                                              │
│  Memory-bound:          ████████████████  ~40%              │
│  - LayerNorm                                                 │
│  - Softmax                                                   │
│  - GELU, Dropout                                             │
│  - 残差连接, Concat                                          │
│  - Embedding                                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 负载规模计算

### 5.1 单次前向的计算量

```python
# 配置
B = 250        # batch_size
T = 128        # sequence_len
C = 128        # emb_size
H = 8          # head_size
L = 12         # n_layer
FF = 512       # 4 * emb_size

# 每层计算量 (FLOPs)
# Q/K/V projection: 3 × B×T×C×H × 16头 = 3 × B×T×C²
# Attention matmul: 2 × B×T×T×H × 16头 = 2 × B×T²×C
# Output projection: B×T×C×C
# FFN: B×T×C×FF + B×T×FF×C = 2 × B×T×C×FF

per_layer = 3*B*T*C*C + 2*B*T*T*C + B*T*C*C + 2*B*T*C*FF
          = B*T*C * (4*C + 2*T + 2*FF)
          = 250*128*128 * (512 + 256 + 1024)
          ≈ 7.3 GFLOPs/层

total = 12 * 7.3 ≈ 88 GFLOPs/forward
```

### 5.2 内存访问量

```python
# 参数量
params = L * (3*C*H*16 + C*C + 2*C*FF + 4*C) + 2*vocab*C
       ≈ 12 * (3*128*8*16 + 16384 + 2*128*512 + 512) + 2*256*128
       ≈ 2.5M 参数 = 10 MB (FP32)

# 激活值
activations_per_layer ≈ B*T*C * 10 ≈ 40 MB/层
total_activations ≈ 12 * 40 ≈ 480 MB
```

---

## 6. 调度器并发机会

### 6.1 Compute + Memory 并发场景

```
理想并发：
  HP (Compute): 执行大 GEMM（Linear/Attention）
  BE (Memory):  执行 LayerNorm/Softmax/GELU

时间线：
  HP: [====GEMM====][====GEMM====][====GEMM====]
  BE:   [LayerNorm]    [Softmax]     [GELU]
              ↑             ↑           ↑
          利用空闲内存带宽执行 BE 任务
```

### 6.2 实际考虑

| 因素 | 影响 |
|------|------|
| 操作粒度 | GPT 的单个操作较小，调度开销可能显著 |
| 依赖关系 | 层内操作有数据依赖，不能随意重排 |
| 同一模型 | 如果 HP/BE 跑同一个模型，操作类型相似 |
| 多请求并发 | 多个推理请求时，调度更有意义 |

---

## 7. 总结

### 7.1 模型特点

| 属性 | 值 |
|------|-----|
| 架构 | Decoder-only Transformer |
| 层数 | 12 |
| 隐藏维度 | 128 |
| 注意力头 | 16 × 8 |
| FFN 维度 | 512 |
| 序列长度 | 128 |
| 参数量 | ~2.5M |

### 7.2 GPU 负载特点

| 负载类型 | 占比 | 主要操作 |
|---------|------|---------|
| Compute-bound | ~60% | GEMM (Linear, Attention) |
| Memory-bound | ~40% | LayerNorm, Softmax, GELU, Dropout |

### 7.3 调度器应用

```
最适合调度的场景：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  HP Client: GPT 推理（大量 GEMM，compute-bound）            │
│  BE Client: 数据预处理/后处理（memcpy，memory-bound）       │
│                                                             │
│  → HP 执行 GEMM 时，BE 利用空闲内存带宽传输数据             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
