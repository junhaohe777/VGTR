import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    """
    Semantic Token Memory for Text-based Person ReID
    - No hard part assumption
    - Token-level refinement
    - One-stage training friendly
    """

    def __init__(
        self,
        clip_model,
        feature_dim=512,
        memory_size=32,
        num_slots=4,          # semantic slots (NOT body parts)
        alpha=0.1,            # small residual weight
        momentum=0.9,
        warmup_epochs=5,
        learnable_alpha=False  # NEW: make alpha learnable
    ):
        super().__init__()

        self.device = next(clip_model.parameters()).device
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.num_slots = num_slots
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        
        # Learnable or fixed alpha
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.alpha = alpha

        # semantic memory: [num_slots, memory_size, dim]
        self.register_buffer(
            "memory",
            F.normalize(
                torch.randn(num_slots, memory_size, feature_dim),
                dim=-1
            )
        )

        # write pointer
        self.register_buffer(
            "ptr",
            torch.zeros(num_slots, dtype=torch.long)
        )

        # slot assignment for image tokens (soft)
        self.slot_assign = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, num_slots),
            nn.Softmax(dim=-1)
        )

        # text → memory attention
        self.text_to_mem_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )

        # runtime flags
        self.enable_read = False
        self.enable_write = False
        self.current_epoch = 0

    # ------------------------------------------------------------------
    # public control (called in training loop)
    # ------------------------------------------------------------------
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch >= self.warmup_epochs:
            self.enable_read = True
            self.enable_write = True

    # ------------------------------------------------------------------
    # write: image tokens → memory (NO gradient)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def write(self, image_tokens):
        """
        image_tokens: [B, N, D]
        """
        if not self.enable_write:
            return

        B, N, D = image_tokens.shape
        # Convert to float32 for computation
        image_tokens = F.normalize(image_tokens.float(), dim=-1)

        # soft slot assignment
        slot_weight = self.slot_assign(image_tokens)  # [B, N, S]

        for s in range(self.num_slots):
            # weighted aggregation
            w = slot_weight[..., s].unsqueeze(-1)      # [B, N, 1]
            feat = (w * image_tokens).sum(dim=1) / (w.sum(dim=1) + 1e-6)
            feat = F.normalize(feat, dim=-1)

            ptr = int(self.ptr[s])
            self.memory[s, ptr] = (
                self.momentum * self.memory[s, ptr]
                + (1 - self.momentum) * feat.mean(dim=0)
            )
            self.memory[s, ptr] = F.normalize(self.memory[s, ptr], dim=-1)
            self.ptr[s] = (ptr + 1) % self.memory_size

    # ------------------------------------------------------------------
    # read: memory → text tokens (residual, detached)
    # ------------------------------------------------------------------
    def read(self, text_tokens, token_mask=None):
        """
        text_tokens: [B, T, D]
        token_mask:  [B, T] (optional)
        """
        if not self.enable_read:
            return text_tokens

        B, T, D = text_tokens.shape
        orig_dtype = text_tokens.dtype

        # flatten memory slots and convert to float32
        mem = self.memory.view(-1, D).float()         # [S*M, D]
        mem = mem.unsqueeze(0).repeat(B, 1, 1)        # [B, S*M, D]

        # Convert inputs to float32 for computation
        text_tokens_float = text_tokens.float()

        # attention in float32
        attn_out, _ = self.text_to_mem_attn(
            query=text_tokens_float,
            key=mem,
            value=mem,
            key_padding_mask=None
        )

        # residual + detach (critical)
        refined = text_tokens_float + self.alpha * attn_out.detach()

        # Convert back to original dtype if needed
        refined = refined.to(orig_dtype)

        if token_mask is not None:
            refined = refined * token_mask.unsqueeze(-1)

        return refined

    # ------------------------------------------------------------------
    # forward wrapper
    # ------------------------------------------------------------------
    def forward(self, text_tokens, image_tokens=None, token_mask=None):
        """
        text_tokens:  [B, T, D]
        image_tokens: [B, N, D] (optional, for write)
        """
        if self.training and image_tokens is not None:
            self.write(image_tokens)

        text_tokens = self.read(text_tokens, token_mask)
        return text_tokens
