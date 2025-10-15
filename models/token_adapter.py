import math
import torch
from typing import Tuple
from .vision_transformer import VisionTransformer

from .ops.ta import (
    token_injector, token_ejector,
    compute_distance, group_tokens_by_imp_score
)

class TokenAdapterViT(VisionTransformer):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        rh: float = 0.3,
        rw: float = 0.3,
        rp_hr: float = 0.9,
        rp_wr: float = 0.95, 
        l_b: int = 11,
        l_m: int = 13,
        l_a: int =0,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.rh = rh  # rh and rw, ratios of row and cols tokens to drop
        self.rw = rw
        self.rp_hr = rp_hr # rp_hr and rp_wr, ratios of representative tokens in row and col
        self.rp_wr = rp_wr
        self.l_b = l_b
        self.l_m = l_m
        self.l_a = l_a
        self.threshold = threshold
        
    def __init__(
        self,
        rh: float = 0.3,
        rw: float = 0.3,
        rp_hr: float = 0.9,
        rp_wr: float = 0.95,
        l_b: int = 11,
        l_m: int = 13,
        l_a: int = 0,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rh = rh
        self.rw = rw
        self.rp_hr = rp_hr
        self.rp_wr = rp_wr
        self.l_b = l_b
        self.l_m = l_m
        self.l_a = l_a
        self.threshold = threshold

    @staticmethod
    def _split_cls(x: torch.Tensor):
        # x: (n, S, d) with CLS at index 0
        x_cls = x[:, 0, :]                    # (n, d)
        x_patches = x[:, 1:, :]               # (n, S-1, d)
        return x_cls, x_patches

    @staticmethod
    def _concat_cls(x_cls: torch.Tensor, x_seq: torch.Tensor):
        # x_cls: (n, d), x_seq: (n, S', d) -> (n, 1+S', d)
        return torch.cat([x_cls.unsqueeze(1), x_seq], dim=1)

    @staticmethod
    def _grid_shape(h: int, w: int, p: int) -> Tuple[int, int]:
        return h // p, w // p

    def _inject_tokens(
        self,
        x_patches: torch.Tensor,  # (n, N, d), N = n_h * n_w
        n: int,
        n_h: int,
        n_w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # group -> labels then inject
        x_labs = group_tokens_by_imp_score(
            x_patches.view(n, n_h, n_w, -1),
            rh=self.rh,
            rw=self.rw,
            rp_h=self.rp_hr,
            rp_w=self.rp_wr,
            device=device,
        )
        x_reduced = token_injector(x_patches, x_labs)  # (n, N', d)
        M_d = compute_distance(q=x_patches, kv=x_reduced, mode="cosine")  # (n, N, N')
        return x_reduced, M_d

    def _eject_tokens(
        self,
        x_patches_reduced: torch.Tensor,   # (n, N', d)
        M_d: torch.Tensor,                 # (n, N,  N')
    ) -> torch.Tensor:
        # compute_distance expects (n, N, d) for q/kv 
        x_full = token_ejector(x_patches_reduced, M_d, thres=self.threshold)  # (n, N, d)
        return x_full

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Expected (n, c, h, w)"
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == w == self.image_size

        n_h, n_w = self._grid_shape(h, w, p)

        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        x = x.permute(2, 0, 1)  # (S-1, n, d) where S-1=N

        if self.classifier == "token":
            batch_class_token = self.class_token.expand(-1, n, -1)  # (1, n, d)
            x = torch.cat([batch_class_token, x], dim=0)      

        x = x + self.encoder.pos_embedding
        x = x.permute(1, 0, 2)  # (n, S, d)
        
        reduced = False
        Md_cached = None
        print(f'Original input sequence: {x.shape}')
        for i, layer in enumerate(self.encoder.layers):
            
            # Inject (reduce) exactly at l_b: operate on patch tokens only.
            if (i == self.l_b) and (not reduced):
                x_cls, x_p = self._split_cls(x)                # exclude CLS
                
                x_p, M_d = self._inject_tokens(
                    x_patches=x_p,
                    n=n,
                    n_h=n_h,
                    n_w=n_w,
                    device=x.device,
                )    

                print(f'Injected token at layer {i}: {x_p.shape}')
                x = self._concat_cls(x_cls, x_p)    # reattach CLS for blocks
                Md_cached = M_d
                reduced = True

            # Run the transformer block
            x = layer(self.encoder.dropout(x))

            # Eject (reconstruct) exactly at l_m: use original patch tokens for q.
            if (i == self.l_m) and reduced:
                # reduced tokens X`r already processed by subsequent layers after after the injection
                x_cls_now, x_patches_red = self._split_cls(x)  # (n, N', d)
                if Md_cached:
                    # eject tokens back to full length N
                    x_patches_full = self._eject_tokens(
                        x_patches_reduced=x_patches_red,
                        M_d=Md_cached,
                    )                                              
                    print(f'Ejected token at layer {i}: {x_patches_full.shape}')
                    # restore sequence length and continue
                    x = self._concat_cls(x_cls_now, x_patches_full)
                    reduced = False
                else:
                    print("Warning: M_d not cached. No Reconstruction possible.")

        x = self.encoder.ln(x)

        if self.classifier == "token":
            x = x[:, 0, :]   # (n, d)
        else:
            x = x.mean(dim=1)

        x = self.trunk_output(x)
        if self.head is not None:
            x = self.head(x)
        return x

class TokenAdapterViTL16(TokenAdapterViT):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )


class TokenAdapterViTH14(TokenAdapterViT):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            **kwargs,
        )

