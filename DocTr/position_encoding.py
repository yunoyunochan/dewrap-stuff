# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from typing import List
from typing import Optional
from torch import Tensor


class NestedTensor(object):
    """
    container สำหรับ “feature + padding mask” ที่เดินทางคู่กันตลอด pipeline ของ DETR 
    เพื่อให้ทุกที่ (รวมถึง position encoding) รู้ว่าพิกเซลไหนคือข้อมูลจริง
    """
    def __init__(self, tensors, mask: Optional[Tensor]):
        """
        tensors : feature map จาก backbone, เช่น [B, C, H, W]
        mask : ใช้ตอนสร้าง positional encoding (PositionEmbeddingSine) ให้ “เดิน index” 
        เฉพาะจุดที่เป็นข้อมูลจริงบอกว่าตำแหน่งไหนเป็น padding (ไม่มีข้อมูลจริง) เพื่อ ใช้ตอนทำ attention mask
        """
        self.tensors = tensors # feature map จาก backbone, เช่น [B, C, H, W]
        
        self.mask = mask # บอกว่าตำแหน่งไหนเป็น padding (ไม่มีข้อมูลจริง) เพื่อ ใช้ตอนทำ attention mask

    def to(self, device):
        ### type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        # สร้าง NestedTensor ตัวใหม่ที่อยู่บน device ใหม่
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        """
        เทคนิคของ DETR สำหรับสร้าง “ตำแหน่ง” จาก mask ก่อน แล้วค่อยเอาไปเข้าฟังก์ชัน sinusoidal
        mask : [B,H,W]
        """

        assert mask is not None
        """
        e.g. [B,H,W] mask[0] (H x W):
            [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]

        y_embed = mask.cumsum(1)
        [[1, 1, 1, 1],   # row 0
        [2, 2, 2, 2],   # row 1
        [3, 3, 3, 3]]   # row 2

        x_embed:

        [[1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]]

        ก็จะได้ x_embed[b, y, x] ≈ index ของแกน x
        """
        # [B,H,W] -> [B,H,W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        # [B,H,W] -> [B,H,W]
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize: # ใช้ค่าสุดท้ายของ H,W เพราะมันนับมาแล้วว่า H,W มีค่าได้เท่าไหร่
            eps = 1e-6;
            # scale * ([B,H,W] / [B,1,W] + eps) -> [B,H,W]
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            #  scale * ([B,H,W] / [B,H,1] + eps) -> [B,H,W]
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # [|pos_feats|]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32).cuda();
        # [|pos_feats|] -> [|pos_feats|]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # [B,H,W,1] / [|pos_feats|] -> [B,H,W,|pos_feats|]
        pos_x = x_embed[:, :, :, None] / dim_t
        # # [B,H,W,1] / [|pos_feats|] -> [B,H,W,|pos_feats|]
        pos_y = y_embed[:, :, :, None] / dim_t
        # stack([B,H,W,|pos_feats|//2], [B,H,W,|pos_feats|//2], dim= 4) -> [B,H,W,|pos_feats|//2,2] -> [B,H,W,|pos_feats|]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # stack([B,H,W,|pos_feats|//2], [B,H,W,|pos_feats|//2], dim= 4) -> [B,H,W,|pos_feats|//2,2] -> [B,H,W,|pos_feats|]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # cat([B,H,W,|pos_feats|], [B,H,W,|pos_feats|], dim= 3) -> [B,H,W,|pos_feats|x2] -> [B,|pos_feats|x2,H,W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # |pos_feats|x2 -> (ครึ่งหนึ่ง encode y, อีกครึ่ง encode x)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.

    row_embed ต้องรองรับตำแหน่งแนวตั้ง j ในช่วง [0, H-1]
    col_embed ต้องรองรับตำแหน่งแนวนอน i ในช่วง [0, W-1]
    (assume ว่า feature map หลัง backbone มี H <= 50, W <= 50)
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        #  self.row_embed, self.col_embed : [50,num_pos_feats]
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) # [w]
        j = torch.arange(h, device=x.device) # [h]
        x_emb = self.col_embed(i)  # [w,num_pos_feats]
        y_emb = self.row_embed(j)  # [h,num_pos_feats]
        # ([1,w,num_pos_feats] -> [h,w,num_pos_feats])cat([h,1,num_pos_feats] -> [h,w,num_pos_feats])
        # -> [h,w,num_pos_featsx2] -> [num_pos_featsx2,h,w] -> [1,num_pos_featsx2,h,w] 
        # -> [B,num_pos_featsx2,h,w]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # [B,num_pos_featsx2,h,w]
        return pos

def build_position_encoding(hidden_dim=512, position_embedding='sine'):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # [B,num_pos_feats,H,W]
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


