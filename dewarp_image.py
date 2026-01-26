#!/usr/bin/env python3
"""
Standalone DocTr-Plus inference script for document dewarping.
Usage: python3 dewarp_image.py --input data/1152226.jpg --model DocTr-Plus/model_pretrained/DocTrP.pth --output data/1152226_geo.png --cpu
python3 dewarp_image.py --cpu

"""
import math
import copy
import argparse
import warnings
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional

warnings.filterwarnings('ignore')


# ========== Position Encoding ==========
class PositionEmbeddingSine(nn.Module):
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
        assert mask is not None
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        device = mask.device
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(hidden_dim=512):
    N_steps = hidden_dim // 2
    return PositionEmbeddingSine(N_steps, normalize=True)


# ========== Feature Extractor ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
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
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch'):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(192, stride=2)
        self.conv2 = nn.Conv2d(192, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        return x


# ========== Transformer Components ==========
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class attnLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_list = nn.ModuleList([copy.deepcopy(nn.MultiheadAttention(d_model, nhead, dropout=dropout)) for i in range(2)])
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_list = nn.ModuleList([copy.deepcopy(nn.LayerNorm(d_model)) for i in range(2)])
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2_list = nn.ModuleList([copy.deepcopy(nn.Dropout(dropout)) for i in range(2)])
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        for memory, multihead_attn, norm2, dropout2, m_pos in zip(memory_list, self.multihead_attn_list, self.norm2_list, self.dropout2_list, memory_pos):
            tgt2 = multihead_attn(query=self.with_pos_embed(tgt, pos),
                                  key=self.with_pos_embed(memory, m_pos),
                                  value=memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + dropout2(tgt2)
            tgt = norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, memory_pos=None):
        return self.forward_post(tgt, memory_list, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, memory_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransEncoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransEncoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf, device):
        pos = self.position_embedding(torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3], device=device).bool())
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        for layer in self.layers:
            imgf = layer(imgf, [imgf], pos=pos, memory_pos=[pos, pos])
        imgf = imgf.permute(1, 2, 0).reshape(bs, c, h, w)
        return imgf


class TransDecoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransDecoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf, query_embed, device):
        pos = self.position_embedding(torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3], device=device).bool())
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        for layer in self.layers:
            query_embed = layer(query_embed, [imgf], pos=pos, memory_pos=[pos, pos])
        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h, w)
        return query_embed


# ========== Flow Prediction Components ==========
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = .25 * self.mask(imgf)
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow
        return mask, coords1


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# ========== Main Model ==========
class GeoTr(nn.Module):
    def __init__(self):
        super(GeoTr, self).__init__()
        self.hidden_dim = hdim = 256
        self.fnet = BasicEncoder(output_dim=hdim, norm_fn='instance')

        self.encoder_block = ['encoder_block' + str(i) for i in range(3)]
        for i in self.encoder_block:
            self.__setattr__(i, TransEncoder(2, hidden_dim=hdim))
        self.down_layer = ['down_layer' + str(i) for i in range(2)]
        for i in self.down_layer:
            self.__setattr__(i, nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1))

        self.decoder_block = ['decoder_block' + str(i) for i in range(3)]
        for i in self.decoder_block:
            self.__setattr__(i, TransDecoder(2, hidden_dim=hdim))
        self.up_layer = ['up_layer' + str(i) for i in range(2)]
        for i in self.up_layer:
            self.__setattr__(i, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.query_embed = nn.Embedding(81, self.hidden_dim)
        self.update_block = UpdateBlock(self.hidden_dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        device = img.device
        coodslar = coords_grid(N, H, W, device)
        coords0 = coords_grid(N, H // 8, W // 8, device)
        coords1 = coords_grid(N, H // 8, W // 8, device)
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

    def forward(self, image1):
        device = image1.device
        fmap = self.fnet(image1)
        fmap = torch.relu(fmap)

        fmap1 = self.__getattr__(self.encoder_block[0])(fmap, device)
        fmap1d = self.__getattr__(self.down_layer[0])(fmap1)
        fmap2 = self.__getattr__(self.encoder_block[1])(fmap1d, device)
        fmap2d = self.__getattr__(self.down_layer[1])(fmap2)
        fmap3 = self.__getattr__(self.encoder_block[2])(fmap2d, device)

        query_embed0 = self.query_embed.weight.unsqueeze(1).repeat(1, fmap3.size(0), 1)
        fmap3d_ = self.__getattr__(self.decoder_block[0])(fmap3, query_embed0, device)
        fmap3du_ = self.__getattr__(self.up_layer[0])(fmap3d_).flatten(2).permute(2, 0, 1)
        fmap2d_ = self.__getattr__(self.decoder_block[1])(fmap2, fmap3du_, device)
        fmap2du_ = self.__getattr__(self.up_layer[1])(fmap2d_).flatten(2).permute(2, 0, 1)
        fmap_out = self.__getattr__(self.decoder_block[2])(fmap1, fmap2du_, device)

        coodslar, coords0, coords1 = self.initialize_flow(image1)
        coords1 = coords1.detach()
        mask, coords1 = self.update_block(fmap_out, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up
        return bm_up


class GeoTrP(nn.Module):
    def __init__(self):
        super(GeoTrP, self).__init__()
        self.GeoTr = GeoTr()

    def forward(self, x):
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        return bm


# ========== Utilities ==========
def load_model(model_path, device):
    """Load pretrained model weights."""
    model = GeoTrP()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # Remove 'module.' prefix from state dict keys
        model_dict = model.state_dict()
        pretrained_dict = {k[7:]: v for k, v in checkpoint.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✓ Loaded model weights from {model_path}")
    else:
        print(f"⚠ Warning: Model file not found at {model_path}")
        print("  Using randomly initialized weights")
    
    model = model.to(device)
    model.eval()
    return model


def dewarp_image(input_path, model, device, output_path=None):
    """Dewarp a single image using the model."""
    # Load and preprocess image
    im_ori = np.array(Image.open(input_path))[:, :, :3] / 255.
    h, w, _ = im_ori.shape
    
    # Resize to 288x288 for model input
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        bm = model(im)
        bm = bm.cpu()
        
        # Resize flow back to original size
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
        
        # Apply dewarping
        out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(),
                           lbl, align_corners=True)
        img_geo = ((out[0] * 255).permute(1, 2, 0).numpy())[:, :, ::-1].astype(np.uint8)
    
    # Save output
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_geo.png"
    
    cv2.imwrite(output_path, img_geo)
    print(f"✓ Dewarped image saved to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='DocTr-Plus Document Dewarping')
    parser.add_argument('--input', type=str, default='data/1152226.jpg',
                       help='Input image path')
    parser.add_argument('--model', type=str, default='DocTr-Plus/model_pretrained/DocTrP.pth',
                       help='Path to pretrained model weights')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: input_geo.png)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode even if GPU is available')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
        print("Using CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU available)")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    
    # Process image
    print(f"Processing image: {args.input}")
    if not os.path.exists(args.input):
        print(f"✗ Error: Input image not found at {args.input}")
        return
    
    dewarp_image(args.input, model, device, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
