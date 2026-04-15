from typing import Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GELU, LayerNorm, Linear

from .utils import clip_outliers, flash_context, normalize_data


def _resolve_text_attention_layer_numbers(
    *,
    text_attn_layers: list[int] | None,
    text_enhanced: bool,
    nlayers: int,
) -> list[int]:
    if text_attn_layers is None:
        if not text_enhanced:
            return []
        num_text_enhanced_layers = min(2, nlayers)
        first_text_layer_number = nlayers - num_text_enhanced_layers + 1
        return list(range(first_text_layer_number, nlayers + 1))

    if not text_attn_layers:
        raise ValueError("text_attn_layers must be non-empty when provided.")

    validated_layers: list[int] = []
    seen_layers: set[int] = set()
    for layer_num in text_attn_layers:
        if isinstance(layer_num, bool) or not isinstance(layer_num, int):
            raise ValueError("text_attn_layers must contain only 1-based integer layer numbers.")
        if layer_num <= 0:
            raise ValueError("text_attn_layers must contain only positive 1-based layer numbers.")
        if layer_num > nlayers:
            raise ValueError(
                f"text_attn_layers contains out-of-range layer {layer_num}; model has {nlayers} layers."
            )
        if layer_num in seen_layers:
            raise ValueError("text_attn_layers must not contain duplicate layer numbers.")
        validated_layers.append(layer_num)
        seen_layers.add(layer_num)
    return validated_layers


def _share_text_attention_modules(transformer_encoder: nn.ModuleList) -> None:
    shared_owner = None
    for layer in transformer_encoder:
        if not getattr(layer, "text_enhanced", False):
            continue
        if shared_owner is None:
            shared_owner = layer
            continue
        layer.text_head_projs = shared_owner.text_head_projs
        layer.text_head_q_norms = shared_owner.text_head_q_norms
        layer.text_head_k_norms = shared_owner.text_head_k_norms


class TabDPTModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        use_flash: bool = True,
        clip_sigma: float = 4.
    ):
        super().__init__()
        self.n_out = n_out
        self.use_flash = use_flash
        self.ninp = ninp
        self.num_heads = nhead
        self.nhid = nhid
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoder = nn.Linear(1, ninp)
        self.head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out + 1))
        self.clip_sigma = clip_sigma

    @flash_context
    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: Literal["cls", "reg"],  # classification or regression
        text_train: torch.Tensor | None = None,
        text_test: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_src = x_src.transpose(0, 1)
        y_src = y_src.squeeze(-1).transpose(0, 1)
        eval_pos = y_src.shape[0]
        assert x_src.shape[1] == y_src.shape[1], "x_src and y_src must have the same batch size"
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        if task == "reg":
            y_src, mean_y, std_y = normalize_data(y_src, return_mean_std=True)
            y_src = clip_outliers(y_src)

        x_src = torch.nan_to_num(x_src, nan=0)
        x_src = self.encoder(x_src)
        mean = (x_src**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean)
        x_src = x_src / rms

        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:eval_pos] + y_src
        src = torch.cat([train_x, x_src[eval_pos:]], 0)
        for layer in self.transformer_encoder:
            src = layer(src, eval_pos, text_train=text_train, text_test=text_test)
        pred = self.head(src)
        if task == "reg":
            pred = pred[eval_pos:, ..., -1]
        elif task == "cls":
            pred = pred[eval_pos:, ..., :-1]
        else:
            raise ValueError(f"Invalid task: {task}")

        if task == "reg":
            pred = pred * std_y + mean_y
            return pred, std_y, mean_y

        return pred

    @classmethod
    def load(
        cls,
        model_state,
        config,
        use_flash,
        clip_sigma: float = 4.,
        text_enhanced: bool = False,
        text_attn_layers: list[int] | None = None,
    ):
        assert config.model.max_num_classes > 2
        model = TabDPTModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=config.model.emsize * config.model.nhid_factor,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
            use_flash=use_flash,
            clip_sigma=clip_sigma
        )

        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        model_state = {k.replace("model.", ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)

        text_attn_layer_numbers = _resolve_text_attention_layer_numbers(
            text_attn_layers=text_attn_layers,
            text_enhanced=text_enhanced,
            nlayers=len(model.transformer_encoder),
        )

        # Replace the configured transformer layers with text-enhanced versions.
        for layer_num in text_attn_layer_numbers:
            layer_idx = layer_num - 1
            if not model.transformer_encoder[layer_idx].text_enhanced:
                base_layer = model.transformer_encoder[layer_idx]
                model.transformer_encoder[layer_idx] = TransformerEncoderLayer(
                    embed_dim=base_layer.embed_dim,
                    num_heads=base_layer.num_heads,
                    ff_dim=base_layer.ff[0].out_features,
                    text_enhanced=True,
                )
                # Copy pretrained weights to the new layer while leaving text modules freshly initialized.
                model.transformer_encoder[layer_idx].load_state_dict(base_layer.state_dict(), strict=False)

        _share_text_attention_modules(model.transformer_encoder)
        model.to(config.env.device)
        model.eval()
        return model


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, text_enhanced=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_norm = LayerNorm(embed_dim)
        self.ff_norm = LayerNorm(embed_dim)
        self.ff = nn.Sequential(Linear(embed_dim, ff_dim), GELU(), Linear(ff_dim, embed_dim))
        self.q_norm = LayerNorm(self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)

        self.text_enhanced = text_enhanced
        if text_enhanced:
            # Use only the first D dims from each raw text embedding before scoring.
            self.TEXT_ENHANCED_EMBED_DIM = 1024
            # Keep total text capacity fixed, but split it across attention heads.
            self.TEXT_ENHANCED_TOTAL_DIM = 128
            if self.TEXT_ENHANCED_TOTAL_DIM % num_heads != 0:
                raise ValueError(
                    "Text semi-attention total width must be divisible by num_heads: "
                    f"total_width={self.TEXT_ENHANCED_TOTAL_DIM}, num_heads={num_heads}"
                )
            self.TEXT_ENHANCED_HEAD_DIM = self.TEXT_ENHANCED_TOTAL_DIM // num_heads

            self.alpha = nn.Parameter(torch.full((num_heads,), 0.0))
            self.text_head_projs = nn.ModuleList(
                [
                    nn.Linear(self.TEXT_ENHANCED_EMBED_DIM, self.TEXT_ENHANCED_HEAD_DIM, bias=False)
                    for _ in range(num_heads)
                ]
            )
            self.text_head_q_norms = nn.ModuleList(
                [LayerNorm(self.TEXT_ENHANCED_HEAD_DIM) for _ in range(num_heads)]
            )
            self.text_head_k_norms = nn.ModuleList(
                [LayerNorm(self.TEXT_ENHANCED_HEAD_DIM) for _ in range(num_heads)]
            )

    def _text_attention_logits(self, text_train: torch.Tensor, text_test: torch.Tensor) -> torch.Tensor:
        if text_train.shape[-1] < self.TEXT_ENHANCED_EMBED_DIM or text_test.shape[-1] < self.TEXT_ENHANCED_EMBED_DIM:
            raise ValueError(
                "Text embedding width is smaller than the configured truncation width: "
                f"train={text_train.shape}, test={text_test.shape}, required_last_dim={self.TEXT_ENHANCED_EMBED_DIM}"
            )
        text_train = text_train[..., :self.TEXT_ENHANCED_EMBED_DIM]
        text_test = text_test[..., :self.TEXT_ENHANCED_EMBED_DIM]
        scale = text_train.new_tensor(1.0 / math.sqrt(self.TEXT_ENHANCED_HEAD_DIM))

        head_logits: list[torch.Tensor] = []
        for proj, q_norm, k_norm in zip(
            self.text_head_projs,
            self.text_head_q_norms,
            self.text_head_k_norms,
        ):
            proj = proj.to(dtype=text_train.dtype)
            q_norm = q_norm.to(dtype=text_train.dtype)
            k_norm = k_norm.to(dtype=text_train.dtype)
            q = q_norm(proj(text_test))
            k = k_norm(proj(text_train))
            # (B, N, L, dk) @ (B, M, L, dk) summed over dk -> per-lag scores (B, L, N, M)
            logits_per_lag = torch.einsum("bnld,bmld->blnm", q, k) * scale
            head_logits.append(logits_per_lag.mean(dim=1))
        return torch.stack(head_logits, dim=1)

    def forward(
        self,
        x,
        eval_pos,
        text_train: torch.Tensor | None = None,
        text_test: torch.Tensor | None = None,
    ):
        x = x.transpose(0, 1)
        B, L, _ = x.size()
        h = self.attn_norm(x)
        q = self.q_proj(h)
        k, v = self.kv_proj(h[:, :eval_pos]).chunk(2, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if not self.text_enhanced or text_train is None or text_test is None:
            attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        else:
            _, attn_weight, _ = _scaled_dot_product_attention_with_attention_scores(q, k, v)
            text_logits = self._text_attention_logits(text_train, text_test)
            text_w = torch.softmax(text_logits, dim=-1)

            ts_gating_val = torch.sigmoid(self.alpha).view(1, self.num_heads, 1, 1)
            attn_weight_train = attn_weight[:, :, :eval_pos, :]
            attn_weight_test = attn_weight[:, :, eval_pos:, :]
            attn_weight_test = attn_weight_test * ts_gating_val + (1 - ts_gating_val) * text_w
            attn_weight = torch.cat([attn_weight_train, attn_weight_test], dim=2)
            attn = attn_weight @ v
            attn = attn.transpose(1, 2)
        attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))
        x = x + attn
        x = x + self.ff(self.ff_norm(x))
        # Keep the sequence-first output contiguous so every encoder block sees
        # the same input stride pattern under torch.compile.
        return x.transpose(0, 1).contiguous()


# from pytorch v2.8.0
# Efficient implementation equivalent to the following:

def _scaled_dot_product_attention_with_attention_scores(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_logit = query @ key.transpose(-2, -1) * scale_factor

    # this line does not change the attention weights for tabdpt
    attn_logit += attn_bias
    attn_weight = torch.softmax(attn_logit, dim=-1)

    # tabdpt use dropout_p=0.0, this line does not change the attention weights
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # return the attention logit, attention weight and the attention output
    return attn_logit, attn_weight, attn_weight @ value
