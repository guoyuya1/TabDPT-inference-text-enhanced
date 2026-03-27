from typing import Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GELU, LayerNorm, Linear

from .utils import clip_outliers, flash_context, normalize_data


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
        text_enhanced_attn_weight: torch.Tensor | None = None,
        manual_last_layer_attn_weight: torch.Tensor | None = None,
        return_last_layer_attn: bool = False,
        return_query_prehead_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
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
        final_attn_weight = None
        if text_enhanced_attn_weight is None:
            for i, layer in enumerate(self.transformer_encoder):
                capture = return_last_layer_attn and i == len(self.transformer_encoder) - 1
                manual_attn = (
                    manual_last_layer_attn_weight if i == len(self.transformer_encoder) - 1 else None
                )
                if capture:
                    src, final_attn_weight = layer(
                        src,
                        eval_pos,
                        return_attn_weights=True,
                        manual_attn_weight=manual_attn,
                    )
                else:
                    src = layer(src, eval_pos, manual_attn_weight=manual_attn)
        else:
            for layer in self.transformer_encoder[:-1]:
                src = layer(src, eval_pos)
            if return_last_layer_attn:
                src, final_attn_weight = self.transformer_encoder[-1](
                    src,
                    eval_pos,
                    text_enhanced_attn_weight,
                    return_attn_weights=True,
                    manual_attn_weight=manual_last_layer_attn_weight,
                )
            else:
                src = self.transformer_encoder[-1](
                    src,
                    eval_pos,
                    text_enhanced_attn_weight,
                    manual_attn_weight=manual_last_layer_attn_weight,
                )
        query_prehead_tokens = src[eval_pos:] if return_query_prehead_tokens else None

        pred = self.head(src)
        if task == "reg":
            pred = pred[eval_pos:, ..., -1]
        elif task == "cls":
            pred = pred[eval_pos:, ..., :-1]
        else:
            raise ValueError(f"Invalid task: {task}")

        if task == "reg":
            pred = pred * std_y + mean_y
            extras = []
            if return_last_layer_attn:
                extras.append(final_attn_weight)
            if return_query_prehead_tokens:
                extras.append(query_prehead_tokens)
            if extras:
                return (pred, std_y, mean_y, *extras)
            return pred, std_y, mean_y

        if return_query_prehead_tokens and return_last_layer_attn:
            return pred, final_attn_weight, query_prehead_tokens
        if return_last_layer_attn:
            return pred, final_attn_weight
        if return_query_prehead_tokens:
            return pred, query_prehead_tokens
        return pred

    @classmethod
    def load(cls, model_state, config, use_flash, clip_sigma: float = 4., text_enhanced: bool = False):
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
        
        # Replace last transformer layer with text-enhanced version
        if text_enhanced:
            last_idx = len(model.transformer_encoder) - 1
            last_layer = model.transformer_encoder[last_idx]
            model.transformer_encoder[last_idx] =TransformerEncoderLayer(
                embed_dim=last_layer.embed_dim,
                num_heads=last_layer.num_heads,
                ff_dim=last_layer.ff[0].out_features,
                text_enhanced=True,
            )
            # Copy pretrained weights to the new layer
            model.transformer_encoder[last_idx].load_state_dict(last_layer.state_dict(), strict=False)
        
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

        if text_enhanced:
            # one alpha for all heads
            # self.alpha = nn.Parameter(torch.zeros(1))
            # Per-head gating parameter: one alpha per attention head
            # initialize gating value =0.5
            self.alpha = nn.Parameter(torch.zeros(num_heads))

            # initialize gating value roughly to 1
            # Initialize to a large positive logit so sigmoid(alpha) starts ~1.0.
            self.alpha = nn.Parameter(torch.full((num_heads,), 10.0))
            # One per-head projection for text attention logits: Linear -> GELU.
            self.text_attn_linears = nn.ModuleList(
                [nn.Sequential(nn.Linear(1, 1), nn.GELU()) for _ in range(num_heads)]
            )
            
            with torch.no_grad():
                for proj in self.text_attn_linears:
                    proj[0].weight.fill_(1.0)
                    proj[0].bias.zero_()
            
            # self.register_buffer('ts_gating_val', torch.ones(1))

    def forward(
        self,
        x,
        eval_pos,
        text_enhanced_attn_weight: torch.Tensor | None = None,
        return_attn_weights: bool = False,
        manual_attn_weight: torch.Tensor | None = None,
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
        if manual_attn_weight is not None:
            if manual_attn_weight.dim() == 3:
                manual_attn_weight = manual_attn_weight.unsqueeze(0)
            attn_weight = manual_attn_weight.to(device=q.device, dtype=q.dtype)
            if attn_weight.shape != (B, self.num_heads, L, eval_pos):
                raise ValueError(
                    f"manual_attn_weight must have shape {(B, self.num_heads, L, eval_pos)}, "
                    f"got {tuple(attn_weight.shape)}"
                )
            attn = (attn_weight @ v).transpose(1, 2)
        elif text_enhanced_attn_weight is None:
            if return_attn_weights:
                _, attn_weight, _ = _scaled_dot_product_attention_with_attention_scores(q, k, v)
                attn = (attn_weight @ v).transpose(1, 2)
            else:
                attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        else:
            # N: number of training & test samples 
            # N_train=eval_pos: number of training samples
            # B: batch size
            # attn_weight: (B, num_heads, N, N_train) — softmaxed attention weights
            attn_logit, attn_weight, _ = _scaled_dot_product_attention_with_attention_scores(q, k, v)
            # text_enhanced_attn_weight: (B, N, N_train) -> (B, num_heads, N, N_train)
            # same text_enhanced_atten_weight for all heads
            text_enhanced_attn_weight = text_enhanced_attn_weight.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            # Compute per-head gating values from learnable parameter: (num_heads,)
            ts_gating_val = torch.sigmoid(self.alpha)  # Shape: (num_heads,)
            # Reshape for broadcasting: (1, num_heads, 1, 1) to match (B, num_heads, N, N_train)
            ts_gating_val = ts_gating_val.view(1, self.num_heads, 1, 1)
            # Blend attention weights: each head uses its own gating value
            # Only apply blending to test samples (last N_test rows in second-to-last dimension)
            # Split into train and test parts
            attn_weight_train = attn_weight[:, :, :eval_pos, :]  # (B, num_heads, N_train, N_train)
            attn_weight_test = attn_weight[:, :, eval_pos:, :]  # (B, num_heads, N_test, N_train)
            # Apply blending only to test part - slice text_enhanced_attn_weight to match test positions
            text_enhanced_attn_weight_test = text_enhanced_attn_weight[:, :, eval_pos:, :]  # (B, num_heads, N_test, N_train)
            # Apply one trainable linear transform per head to cosine logits, then softmax.
            head_logits = []
            for h in range(self.num_heads):
                head_logit = text_enhanced_attn_weight_test[:, h:h + 1, :, :]  # (B, 1, N_test, N_train)
                head_logit = self.text_attn_linears[h](head_logit.unsqueeze(-1)).squeeze(-1)
                head_logits.append(head_logit)
            text_enhanced_attn_logits_test = torch.cat(head_logits, dim=1)
            text_enhanced_attn_weight_test = torch.softmax(text_enhanced_attn_logits_test, dim=-1)
            attn_weight_test = attn_weight_test * ts_gating_val + (1 - ts_gating_val) * text_enhanced_attn_weight_test
            # Concatenate back
            attn_weight = torch.cat([attn_weight_train, attn_weight_test], dim=2)  # (B, num_heads, N, N_train)
            attn = attn_weight @ v
            attn = attn.transpose(1, 2)
        attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))
        x = x + attn
        x = x + self.ff(self.ff_norm(x))
        if return_attn_weights:
            return x.transpose(0, 1), attn_weight
        return x.transpose(0, 1)


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
