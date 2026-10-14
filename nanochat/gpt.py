"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

# =============================================================================
# BEGINNER'S PRIMER: what is this file?
# =============================================================================
# This file defines the neural network itself: a "GPT" (Generative Pre-trained
# Transformer). Conceptually a GPT is just a (very big) function:
#
#     input:  a list of token ids, e.g. [464, 3290, 3332]   ("The cat sat")
#     output: for EVERY position, a score for every word in the vocabulary
#             saying how likely that word is to come NEXT.
#
# Training = nudging millions of numbers ("parameters" / "weights") inside this
# function so that the score of the real next token goes up.
# Generating text = repeatedly asking "what comes next?", picking a token,
# appending it, and asking again.
#
# The data flow through the model (see GPT.forward at the bottom):
#
#   token ids --(embedding table)--> vectors (one list of n_embd numbers per token)
#            --> Block 1 --> Block 2 --> ... --> Block n_layer
#            --(lm_head)--> "logits" (one score per vocabulary word, per position)
#
# Each Block has two parts:
#   1. Attention: each token "looks at" the earlier tokens and gathers
#      information from the ones that are relevant to it.
#   2. MLP: each token then "thinks" on its own about what it gathered.
#
# Key vocabulary used throughout the code:
#   - tensor: a multi-dimensional array of numbers (like a nested Python list,
#     but fast and on the GPU). The shape (B, T, C) means:
#       B = batch size (how many independent text sequences we process at once)
#       T = time / sequence length (how many tokens in each sequence)
#       C = channels (= n_embd, the length of each token's vector)
#   - Linear layer: a matrix multiplication, y = x @ W. The matrix W is learned.
#     If you know y = m*x from high school, this is the same idea but with
#     vectors and matrices instead of single numbers.
#   - "learnable parameter": a number the optimizer is allowed to change
#     during training. Everything wrapped in nn.Parameter / nn.Linear /
#     nn.Embedding is learnable.
#
# Great background resources (highly recommended, in order):
#   - 3Blue1Brown "Neural Networks" series (visual intuition, high-school math):
#     https://www.3blue1brown.com/topics/neural-networks
#     (chapters 5-7 are specifically about GPT/Transformers and attention)
#   - Andrej Karpathy "Let's build GPT: from scratch, in code, spelled out":
#     https://www.youtube.com/watch?v=kCc8FmEb1nY
#   - The Illustrated Transformer (pictures of every step):
#     https://jalammar.github.io/illustrated-transformer/
#   - Original Transformer paper "Attention Is All You Need":
#     https://arxiv.org/abs/1706.03762
#   - PyTorch basics (tensors, nn.Module): https://pytorch.org/tutorials/beginner/basics/intro.html
#
# This model has many modern "tricks" on top of the original GPT-2 design
# (listed in the docstring above). On a first read you can skip the parts
# marked "smear", "backout", "value embeddings", "x0_lambdas", and the sliding
# window logic - the model would still work (slightly worse) without them.
# =============================================================================

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

# The "size knobs" of the model. A @dataclass is just a Python class that holds
# fields (like a struct). Bigger numbers = bigger, smarter, slower model.
@dataclass
class GPTConfig:
    sequence_len: int = 2048 # max number of tokens the model can look at at once ("context length")
    vocab_size: int = 32768 # number of distinct tokens the tokenizer knows (2**15)
    n_layer: int = 12 # number of Blocks stacked on top of each other ("depth")
    n_head: int = 6 # number of query heads (attention is split into this many parallel "heads")
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768 # length of the vector that represents each token ("width" / "model dimension")
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


# RMSNorm ("Root Mean Square Normalization"): rescale each token's vector so its
# numbers have an average size of ~1. Concretely: x / sqrt(mean(x**2)).
# Why? As vectors pass through many layers their numbers can drift to be huge
# or tiny, which makes training unstable. Normalizing keeps them in a sane range.
# Background: https://arxiv.org/abs/1910.07467 (RMSNorm paper)
def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings).

    Beginner note: a Linear layer computes y = x @ W.T, i.e. each output number
    is a weighted sum of the input numbers, and the weights W are learned.
    "dtype" is the number format: fp32 = 32-bit float (precise, slow),
    bf16 = 16-bit "bfloat16" (less precise, ~2x faster and half the memory).
    See https://en.wikipedia.org/wiki/Bfloat16_floating-point_format"""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

# Rotary Position Embeddings (RoPE).
# Problem: attention by itself has no idea about word ORDER ("dog bites man" vs
# "man bites dog" would look identical). We need to inject position info.
# RoPE's idea: treat pairs of numbers in the vector as (x, y) points on a 2D
# plane, and ROTATE each point by an angle proportional to the token's position.
# High-school trig: rotating (x1, x2) by angle theta gives
#     (x1*cos(theta) - x2*sin(theta),  x1*sin(theta) + x2*cos(theta))
# Because of how dot products work, after rotating both queries and keys, the
# attention score between two tokens only depends on their DISTANCE apart,
# which is exactly the kind of position info that language needs.
# Background: https://blog.eleuther.ai/rotary-embeddings/ and https://arxiv.org/abs/2104.09864
def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

# -----------------------------------------------------------------------------
# Self-Attention: how tokens share information with each other.
#
# Intuition: in "The cat sat on the mat because it was tired", to understand
# "it" the model needs to look back and find "cat". Attention does that.
#
# Each token produces three vectors (via three Linear layers):
#   - Query (q): "what am I looking for?"
#   - Key   (k): "what do I contain?" (an advertisement / label)
#   - Value (v): "what information do I hand over if someone picks me?"
# For each token, we compute a score against every earlier token:
#   score = dot_product(q_me, k_other)    (big when the two vectors point the same way)
# then turn the scores into percentages that sum to 1 using "softmax"
# (https://en.wikipedia.org/wiki/Softmax_function), and output the weighted
# average of the other tokens' values. In formula form:
#   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
#
# "Causal" means a token may only look at itself and tokens BEFORE it, never
# at the future. Otherwise during training it could simply peek at the answer.
#
# "Multi-head": we split the vector into n_head smaller chunks and run
# attention independently in each, so different heads can track different
# relationships (e.g. one tracks grammar, another tracks who "it" refers to).
#
# GQA (Grouped-Query Attention): use fewer key/value heads than query heads,
# with several query heads sharing one k/v head. It saves memory at inference.
# https://arxiv.org/abs/2305.13245
#
# Visual explanation: https://www.youtube.com/watch?v=eMlx5fFNoYc (3Blue1Brown, attention)
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head # e.g. 768 dims / 6 heads = 128 dims per head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False) # makes queries
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False) # makes keys
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False) # makes values
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False) # mixes the heads' outputs back together
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # .view() reshapes without copying: the last dim of size n_embd is split into (heads, head_dim).

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        # (advanced trick, safe to skip on first read: https://arxiv.org/abs/2410.17897)
        # sigmoid(z) = 1/(1+e^-z) squashes any number into (0, 1), a common way to make a "gate" / dimmer switch.
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm: keeps dot-product scores from blowing up (https://arxiv.org/abs/2302.05442)
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # This one call does the whole softmax(Q @ K^T / sqrt(d)) @ V computation described above,
        # but in a memory-efficient fused GPU kernel. See https://arxiv.org/abs/2205.14135
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            # (the KV cache stores keys/values of earlier tokens so we don't recompute them for
            # every new generated token - see KVCache in engine.py)
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# MLP ("Multi-Layer Perceptron", a.k.a. feed-forward network).
# After attention gathered information from other tokens, the MLP processes each
# token independently: expand the vector 4x wider, apply a non-linear function,
# then shrink it back. Much of the model's "knowledge" (facts) is thought to be
# stored in these weights. https://www.youtube.com/watch?v=9-Jl0dxWQs8 (3Blue1Brown)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False) # "fc" = fully connected, expands
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False) # projects back down

    def forward(self, x):
        x = self.c_fc(x)
        # Activation function: relu(x) = max(0, x), then squared. Without a non-linear step like
        # this, two Linear layers in a row would collapse into one single Linear layer (a product
        # of matrices is just another matrix), and the network could only learn straight-line
        # relationships. https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


# One Transformer layer = attention + MLP. The model stacks n_layer of these.
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # The "x = x + ..." pattern is a RESIDUAL CONNECTION: each sub-layer computes a small
        # update that gets ADDED to x, instead of replacing x. Think of x as a shared notepad
        # (the "residual stream") that every layer reads from and writes notes onto. This makes
        # very deep networks much easier to train. https://arxiv.org/abs/1512.03385 (ResNet)
        # norm() is applied before each sub-layer ("pre-norm").
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.

        Beginner note: the "meta device" is a PyTorch trick where tensors have a shape but no
        memory/values. It lets us build the model skeleton instantly, then move it to the GPU
        and fill in real numbers. https://pytorch.org/docs/stable/meta.html
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            # wte = "word token embedding": a lookup table with one learned row of n_embd numbers
            # per vocabulary token. Token id 464 -> row 464. Similar words end up with similar rows.
            # https://jalammar.github.io/illustrated-word2vec/
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            # h = "hidden layers": the stack of Transformer Blocks
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        # lm_head = "language modeling head": turns the final vector of each token into one score
        # ("logit") per vocabulary token. The higher the logit, the more likely that token comes next.
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros

        Beginner note: before training, every weight must start at SOME value. All zeros would
        make every neuron identical (they'd all learn the same thing), and values that are too big
        make the numbers explode as they flow through layers. So we use small random numbers whose
        spread ("std" = standard deviation) is carefully chosen, typically ~1/sqrt(input size) so
        that the output of each layer has roughly the same size as its input.
        Background: https://www.deeplearning.ai/ai-notes/initialization/
        Note: the exact stds used below have been tuned and differ a bit from the summary above.
        Projections into the residual stream start at zero, so every Block initially does nothing
        ("x = x + 0") and the network gradually learns to use each layer.
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # Each pair of channels rotates at its own speed (frequency). The first pairs rotate fast
        # (good for telling apart nearby positions), the last pairs rotate very slowly (good for
        # long distances) - like the second, minute and hour hands on a clock.
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        # torch.outer builds a table: angle[position, pair] = position * inv_freq[pair]
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)

        Beginner note: a "sliding window" layer only lets each token look back a limited number
        of tokens instead of the whole context. Cost of attention grows with how far you look,
        so short-window layers are cheaper; a few full-context layers keep long-range ability.
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        (FLOP = "floating point operation", i.e. one multiply or one add. Counting FLOPs tells us
        how much compute training costs, and lets us compare models of different sizes fairly.)
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        # The optimizer is the algorithm that updates the weights after each training step.
        # The simplest version ("gradient descent") is:  weight = weight - lr * gradient
        # where the gradient says which direction makes the loss go UP, so we step the other way,
        # and lr ("learning rate") is the step size. Different kinds of parameters behave
        # differently, so here we split them into groups, each with its own learning rate and
        # settings. Big 2D weight matrices use the Muon optimizer; everything else uses AdamW.
        # See nanochat/optim.py for details on both.
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # idx:     (B, T) tensor of token ids, the input text
        # targets: (B, T) tensor of the "correct next token" for every position (training only).
        #          Typically targets is idx shifted left by one: for input "The cat sat", the
        #          targets are "cat sat on". Positions with target -1 are ignored.
        # Returns the loss (a single number) if targets is given, else the logits.
        # (In PyTorch, calling model(idx) runs this forward() method.)
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token: (B, T) ids -> (B, T, n_embd) vectors
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        # (advanced trick, safe to skip; a "bigram" is a pair of adjacent tokens)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        # Core idea: x = block_n(...block_2(block_1(x))). The extra lambdas below are learned
        # scalars (single numbers) that let the model rescale the residual stream per layer.
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits
        # (tanh is an S-shaped curve from -1 to 1, so this keeps scores in [-15, 15] for stability)

        if targets is not None:
            # training: given the targets, compute and return the loss
            # Cross-entropy loss: softmax turns logits into probabilities, and the loss for each
            # position is -log(probability the model gave to the correct next token).
            # If the model was 100% sure of the right answer: -log(1) = 0 (perfect).
            # If it gave the right answer only 1%: -log(0.01) ~= 4.6 (bad).
            # We average this over all positions. Training = make this number go down.
            # https://en.wikipedia.org/wiki/Cross-entropy  and
            # https://www.youtube.com/watch?v=PaCmpygFfXo (Karpathy, makemore part 1, explains this loss)
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints

        Beginner note: "autoregressive" means the model generates one token at a time, feeding each
        new token back in as input for the next step. This is a Python generator (it uses `yield`),
        so the caller receives tokens one by one as they are produced, like a streaming chatbot.
        This version recomputes everything each step (slow); engine.py has the fast KV-cache version.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size) we only care about the prediction after the LAST token
            # top-k: throw away all but the k most likely tokens, to avoid picking rare nonsense
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                # temperature < 1 makes the distribution "peakier" (safer, more repetitive),
                # temperature > 1 flattens it (more random, more creative)
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1) # scores -> probabilities that sum to 1
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng) # roll a weighted die
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True) # temperature 0: always pick the top token ("greedy")
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
