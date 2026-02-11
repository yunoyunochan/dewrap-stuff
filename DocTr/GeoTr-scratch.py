import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional;
from position_encoding import build_position_encoding;


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def coords_grid(batch, ht, wd):
    """
        ฟังก์ชัน coords_grid นี้สร้าง coordinate grid สำหรับแต่ละ pixel ในภาพ 
        โดยจะสร้างตาราง (x, y) coordinates ที่ระบุตำแหน่งของทุก pixel.

        ปล Image Tensor - เก็บ ค่าสี/intensity ของแต่ละ pixel แต่ Coordinate Grid - เก็บ ตำแหน่ง (x, y) ของแต่ละ pixel.
    """
    # [ht,wd] * 2 (ไม่ใช่ wt)
    # e.g. [[0,0,0],[1,1,1],[2,2,2]] and [[0,1,2],[0,1,2],[0,1,2]] 
    # (rows, columns)
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    
    # cols,rows → กลับเป็น (x,y) : [2,ht,wd]
    coords = torch.stack(coords[::-1], dim=0).float()
    
    # [2,ht,wd] -> [1,2,ht,wd] -> [B,2,ht,wd]
    return coords[None].repeat(batch, 1, 1, 1)

def upflow8(flow, mode='bilinear'):
    # flow : [B,C,H,W]
    new_size = (8 * flow.shape[2], 8 * flow.shape[3]);
    # F.interpolate มันแค่ทำ spatial upsampling, Bilinear interpolation จะ "เติมค่าระหว่าง" อย่างเดียว
    """
    e.g.สมมติมี flow 1D: [2.0, 6.0] (2 pixels) 
        ถ้า interpolate 2x (ขึ้นเป็น 4 pixels):
        มันเติมค่าให้ smooth ระหว่าง 2.0 → 6.0 แต่มันไม่รู้ว่า:
            ค่า 2.0 หมายถึง "pixel เคลื่อนที่ 2 pixels ใน resolution เดิม"
            ใน resolution ใหม่ควรเป็น 4.0 (เพราะ 2 pixels เดิม = 4 pixels ใหม่)
    """
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True);

class ResidualBlock(nn.Module):
    """
        conv x2 + 1 (if stride != 1), norm x2 + 1 if (cov x 3), relu x3
        + skip connection

        down sample ถ้า stride != 1 ใน conv1
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8;
        if norm_fn == 'group':

            """
                Group Normalization (GN) แบ่ง channels เป็นกลุ่มๆ แล้วทำ normalization แยกภายในแต่ละกลุ่ม
                ไม่ขึ้นกับ batch size เหมาะกับงานที่ memory จำกัด. **แบ่ง C channels เป็น G groups (แต่ละกลุ่มมี C/G channels)
                Batch 0:
                ├─ Group 0: channels [0,1]   → คำนวณ mean₀, var₀ จาก 2×4×4 = 32 values
                ├─ Group 1: channels [2,3]   → คำนวณ mean₁, var₁ จาก 32 values  
                ├─ Group 2: channels [4,5]   → คำนวณ mean₂, var₂ จาก 32 values
                └─ Group 3: channels [6,7]   → คำนวณ mean₃, var₃ จาก 32 values

            """

            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes);
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes);
            if not stride == 1: # แปลว่ามีการ down sample จึงต้อง normalize ใหม่
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes);
            self.norm2 = nn.BatchNorm2d(planes);
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes);

        elif norm_fn == 'instance':
            """
                normalize แต่ละ sample และ channel แยกอิสระกัน โดยคำนวณ mean/variance จาก spatial dimensions (H, W) เท่านั้น.
                ตัวอย่าง [2, 3, 4, 4]:
                รวม 2×3 = 6 การ normalize แยกกัน
                ช่วย preserve content แต่เปลี่ยน style, 
                Preserve instance characteristics - แต่ละภาพถูก normalize แยก ไม่ถูกผลกระทบจากภาพอื่นใน batch
                e.g. 
                    input = torch.randn(20, 100, 35, 45)  # [20, 100, 35, 45]
                    output = instance_norm(input)  # [20, 100, 35, 45]
                    
                Batch 0, Channel 0: คำนวณ mean/var จาก 4×4 = 16 values → normalize
                Batch 0, Channel 1: คำนวณ mean/var จาก 16 values → normalize
                Batch 0, Channel 2: คำนวณ mean/var จาก 16 values → normalize
                Batch 1, Channel 0: คำนวณ mean/var จาก 16 values → normalize
            """
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), 
                self.norm3
            );

    def forward(self, x : torch.Tensor):
        # x : [B,C,H,W]
        y = x;
        # [B,C,H,W] -> [B,C2,H//s,W//s];
        y = self.relu(self.norm1(self.conv1(y)));
        # [B,C2,H//s,W//s] -> [B,C2,H//s,W//s];
        y = self.relu(self.norm2(self.conv2(y)));

        # [B,C,H,W] -> [B,C2,H//s,W//s]
        if self.downsample is not None:
            x = self.downsample(x)
        # [B,C2,H//s,W//s] + [B,C2,H//s,W//s] -> [B,C2,H//s,W//s]
        return self.relu(x+y)


class FlowHead(nn.Module):
    """
    
    upsample flow จาก resolution ต่ำ (เช่น 36×36) ไป resolution สูง (เช่น 288×288) โดยมองว่า “ทุก pixel ความละเอียดสูงเป็น convex combination ของ flow ความละเอียดต่ำรอบๆ” 
    แทนที่จะใช้ bilinear interpolation ธรรมดา
    ไอเดียคือ:
        - แต่ละพิกเซล high-res F_hr(p) จะคำนวณจาก neighbor ของ F_lr รอบ ๆ จุดนั้น
        (เช่น 3×3) โดยใช้ weight ที่เรียนรู้ได้ α_i(p)
        - เงื่อนไข "convex":
        - weights ทุกตัวเป็นบวก: α_i(p) ≥ 0
        - weights รวมกันเป็น 1: Σ_i α_i(p) = 1
        - ดังนั้น F_hr(p) = Σ_{i ∈ N(p)} α_i(p) * F_lr(i)
        โดย N(p) คือเซตของ low-res neighbors รอบจุด p (เช่น 3×3)

    Decoded Features (36 × 36 × 512)  ← จาก Transformer Decoder
          ↓
   [UpdateBlock]  ← FlowHead อยู่ตรงนี้! 🎯
        ├─ FlowHead: predict displacement
        └─ Mask: สำหรับ convex upsampling
          ↓
    Displacement Field (288 × 288 × 2)

    """
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x : [B,C,H,W] -> [B,hidden_dim,H,W] -> [B,2,H,W]
        return self.conv2(self.relu(self.conv1(x)))

# ============================================================
# LEVEL 2: Encoders (ใช้ Level 1)
# ============================================================


class BasicEncoder(nn.Module):
    """
        //8 to the spatial dim.
        conv1 // 2 | conv2 map to output dim. 
        res-block x 3 (block2 //2, block3 //2)
    """
    def __init__(self, output_dim=128, norm_fn='batch'):
        super(BasicEncoder, self).__init__();
        self.norm_fn = norm_fn;

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential();

        # H_out = (H_in + 2P - K)//s + 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3); # //2
        self.relu1 = nn.ReLU(inplace=True);

        """
            planes มาจาก ResNet paper หมายถึง "feature planes" หรือ จำนวน channels ของ feature map
            in_planes = input channels, out_planes = output channels
            "ระนาบ" (plane) ของ features ที่แทน patterns/characteristics ต่างๆ ของภาพ
            "Planes" เน้น geometric interpretation - เป็น 2D plane/layer ที่ stack กัน
            "Channels" เน้น information flow - เป็นช่องทางส่งข้อมูล
        """
        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1);  # //1
        self.layer2 = self._make_layer(128, stride=2);  # //2
        self.layer3 = self._make_layer(192, stride=2);  # //2

        # output convolution
        self.conv2 = nn.Conv2d(192, output_dim, kernel_size=1);
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride);
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1);
        layers = (layer1,layer2);
        # อัพเดท state เพื่อเตรียมสำหรับ layer ถัดไป
        self.in_planes = dim;
        return nn.Sequential(*layers);

    def forward(self,x : torch.Tensor):
        # [B,3,H,W] -> [B,64,H//2,W//2]
        x = self.conv1(x);
        # [B,64,H//2,W//2] -> [B,64,H//2,W//2] -> [B,64,H//2,W//2]
        x = self.norm1(x);
        x = self.relu1(x);

        #  [B,64,H//2,W//2] ->  [B,64,H//2,W//2] ->  [B,128,H//4,W//4] ->  [B,192,H//8,W//8]
        x = self.layer1(x);
        x = self.layer2(x);
        x = self.layer3(x);
        
        # [B,192,H//8,W//8] -> [B,output,H//8,W//8]
        x = self.conv2(x);

        # [B,output,H//8,W//8]
        return x;

# ============================================================
# LEVEL 3: Attention Layer (ใช้ Level 0)
# ============================================================
class attnLayer(nn.Module):
    """
    Docstring for attnLayer
    
    memory_list เป็นลิสต์ของ memory หลายชุด (เช่น feature จากภาพ/ระดับต่างกัน 2 ตัว)
    อยากให้ แต่ละ memory source มี cross‑attention layer แยกกัน ไม่แชร์ weights กัน
    - ตัวแรกใน multihead_attn_list เรียนรู้การ attend จาก tgt ไปหา memory_list[0]
    - ตัวที่สองเรียนรู้การ attend จาก tgt ไปหา memory_list[1]
    ใช้ ModuleList เพราะต้องการ cross‑attention หลายก้อนที่เป็น คนละ module จริงๆ (ไม่แชร์ weights) 
    เพื่อให้แต่ละก้อนเรียนรู้วิธีอ่านข้อมูลจาก memory แต่ละแหล่งได้เฉพาะทาง

    เพราะ attnLayer ใน Pytorch ถูกออกแบบให้รองรับทั้ง “pre‑norm” และ “post‑norm” pattern 
    แบบเดียวกับ Transformer ของ PyTorch/DETR เลยมีเมทอดแยกชื่อ forward_post 
    (และในโค้ดเต็มจะมี forward_pre) แล้ว forward จริงจะเลือกเรียกอันใดอันหนึ่งตาม 
    flag normalize_before ครับ
    """
    def __init__(
        self, 
        d_model, 
        nhead=8, 
        dim_feedforward=2048, 
        dropout=0.1,
        activation="relu", 
        normalize_before=False
    ):
        super().__init__();
        # Multi-Head Attention แบบ Transformer ที่มีการทำ Query, Key, Value (QKV) เหมือนใน paper "Attention Is All You Need" 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout);
        self.multihead_attn_list = nn.ModuleList([
            copy.deepcopy(nn.MultiheadAttention(d_model, nhead, dropout=dropout)) for _ in range(2)
        ]);
        self.linear1 = nn.Linear(d_model, dim_feedforward);
        self.dropout = nn.Dropout(dropout);
        self.linear2 = nn.Linear(dim_feedforward, d_model);

        self.norm1 = nn.LayerNorm(d_model);
        self.norm2_list = nn.ModuleList([copy.deepcopy(nn.LayerNorm(d_model)) for i in range(2)]);
        self.norm3 = nn.LayerNorm(d_model);

        self.dropout1 = nn.Dropout(dropout);
        self.dropout2_list = nn.ModuleList([copy.deepcopy(nn.Dropout(dropout)) for i in range(2)]);
        self.dropout3 = nn.Dropout(dropout);

        self.activation = _get_activation_fn(activation);
        self.normalize_before = normalize_before;

    def with_pos_embed(self, tensor : torch.Tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos;

    def forward_post(
            self, 
            tgt, 
            memory_list, 
            tgt_mask=None, 
            memory_mask=None,
            tgt_key_padding_mask=None, 
            memory_key_padding_mask=None,
            pos=None, 
            memory_pos=None
        ):
        """
            tgt : [|Query_tokens|,B,D], |Query_tokens| = H*W
            memory_list[0] = imgf : [H*W=|Query_tokens|,B,D];
            pos : positional encoding ของ tgt : [|Query_tokens|,B,D]
        """
        # [|QT|, B, D] + [|QT|, B, D] -> [|QT|, B, D]
        # ไม่ต้องใส่ positional encoding กับ value เพราะ value คือข้อมูลจริงที่ต้องการ attend ไปหา
        q = k = self.with_pos_embed(tgt, pos);
        """
        multihead attention with batch_first = False:
            return output, attn_output_weights
             - output : (N,B,D), N = target sequence length, B = batch size, D = embedding dim
             - attn_output_weights : (B, N, S), S = source sequence length
        """
        # [|QT|, B, D] attn [|QT|, B, D] -> [|QT|, B, D]
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0];

        """
            ใส่ dropout ก่อนบวกเข้ากับ residual เพื่อให้มันทำงานกับ 
            "ส่วนที่เพิ่งคำนวณใหม่" (tgt2) เท่านั้น ไม่กระทบ input เดิม (tgt)
        """
        # [|QT|, B, D] + [|QT|, B, D] -> [|QT|, B, D]
        tgt = tgt + self.dropout1(tgt2);
        tgt = self.norm1(tgt);
        
        for memory, multihead_attn, norm2, dropout2, m_pos in zip(
                memory_list, 
                self.multihead_attn_list, 
                self.norm2_list, 
                self.dropout2_list, 
                memory_pos
            ):
            # [|QT|, B, D] attn [|QT|, B, D] -> [|QT|, B, D]
            tgt2 = multihead_attn(
                query=self.with_pos_embed(tgt, pos),
                key=self.with_pos_embed(memory, m_pos),
                value=memory, 
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + dropout2(tgt2);
            tgt = norm2(tgt);
        
        # [|QT|, B, D] -> [|QT|, B, dim_feedforward] -> [|QT|, B, D]
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))));
        # [|QT|, B, D] + [|QT|, B, D] -> [|QT|, B, D]
        tgt = tgt + self.dropout3(tgt2);
        # [|QT|, B, D] -> [|QT|, B, D]
        tgt = self.norm3(tgt);
        return tgt;


# ============================================================
# LEVEL 4: Transformer Modules (ใช้ Level 0 + Level 3)
# ============================================================
class TransEncoder(nn.Module):
    """
        เข้ารหัส global context ของ feature map ให้กลายเป็น representation ที่ rich ขึ้นสำหรับใช้ทำนาย 
    displacement field ต่อ

    รับ fmap จาก BasicEncoder ขนาด [B,C,H,W] แล้วแปลงเป็น sequence [L=H*W,B,D] เพื่อป้อนเข้า 
    transformer encoder.

    - ใส่ 2D positional encoding ให้แต่ละ patch (pixel หลัง downsample) รู้ตำแหน่ง x,y ของตัวเอง.
    - ใช้หลายชั้นของ self‑attention (ผ่าน attnLayer) ให้ทุกตำแหน่งมองเห็นทุกตำแหน่ง 
    → capture global deformation structure ของทั้งหน้าเอกสาร ไม่ใช่คิด flow แบบ local CNN อย่างเดียว.

    """
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransEncoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)
    
    def forward(self,imgf):
        # imgf : [B,C,H,W]

        # [B,2xC,H,W] -> [B,C,H,W]
        pos = self.position_embedding(
            torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3]).bool().cuda()
        );
        bs, c, h, w = imgf.shape;
        # [B,C,H,W] -> [B,C,H*W] -> [H*W,B,C]
        imgf = imgf.flatten(2).permute(2, 0, 1);
        # D = hidden_dim
        
        # pytorch multihead attn ต้องเป็น (L,B,D) เมื่อ batchfirst = False 
        # [B,C,H,W] -> [B,C,H*W] -> [H*W,B,C]
        pos = pos.flatten(2).permute(2, 0, 1);

        for layer in self.layers:
            """
                memory_list = [imgf] (คือใช้ feature ตัวเองเป็น memory)
                pos เป็น positional encoding ของ imgf
            """
            # [H*W,B,C] -> [H*W,B,C]
            imgf = layer(imgf, [imgf], pos=pos, memory_pos=[pos, pos])
        
        # [H*W,B,C] -> [B,C,H*W] -> [B,C,H,W]
        imgf = imgf.permute(1, 2, 0).reshape(bs, c, h, w)

        # [B,C,H,W]
        return imgf


class TransDecoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransDecoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        # [B,2xC,H,W] -> [B,C,H,W]
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf, query_embed):

        # [B,H,W] -> [B,2xC,H,W] -> [B,C,H,W]
        pos = self.position_embedding(
            torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3]).bool().cuda()
        );
        
        bs, c, h, w = imgf.shape
        # [B,C,H,W] -> [B,C,H*W] -> [H*W,B,C]
        imgf = imgf.flatten(2).permute(2, 0, 1)
        # [num_query,D] -> [num_query,1,D] -> [num_query,B,D]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            query_embed = layer(query_embed, [imgf], pos=pos, memory_pos=[pos, pos])
        
        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h, w)
        return query_embed

# ============================================================
# LEVEL 5: Update Block (ใช้ Level 1)
# ============================================================

class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = .25 * self.mask(imgf)  # scale mask to balance gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow
        return mask, coords1

# ============================================================
# LEVEL 6: Main Model (ประกอบจากทุก module)
# ============================================================

class GeoTr(nn.Module):
    def __init__(self, num_attn_layers):
        super(GeoTr, self).__init__()
        self.num_attn_layers = num_attn_layers

        self.hidden_dim = hdim = 256

        self.fnet = BasicEncoder(output_dim=hdim, norm_fn='instance')

        self.TransEncoder = TransEncoder(self.num_attn_layers, hidden_dim=hdim)
        self.TransDecoder = TransDecoder(self.num_attn_layers, hidden_dim=hdim)
        self.query_embed = nn.Embedding(1296, self.hidden_dim)
        
        self.update_block = UpdateBlock(self.hidden_dim)
                                    
    def initialize_flow(self, img):
    
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2) 

        up_flow = F.unfold(8 * flow, [3, 3], padding=1) 
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W) 

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1: torch.Tensor):
        """
            image1 : [B,3,H,W]
        """
        # [B,3,H,W] → [B,hdim,H//8,W//8]
        fmap = self.fnet(image1)
        fmap = torch.relu(fmap)
        
        fmap = self.TransEncoder(fmap)
        fmap = self.TransDecoder(fmap, self.query_embed.weight)  

        # convex upsample baesd on fmap
        coodslar, coords0, coords1 = self.initialize_flow(image1)
        coords1 = coords1.detach()
        mask, coords1 = self.update_block(fmap, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up

        return bm_up 

if __name__ == "__main__":
   m = torch.meshgrid(torch.tensor([1,2,1]),torch.tensor([3,2,1]));
   print(m)