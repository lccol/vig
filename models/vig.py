import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from torch import Tensor
from torch.nn import functional as F
from typing import Union, List, Optional, Tuple

from .utils import graph_to_image
from .patch_embedding import PatchEmbedding, PatchEmbeddingV2
from .grapher import GrapherFC
from .decoder import ViGDecoder

class FFN(nn.Module):
    def __init__(self, in_features: int, out_features: int, act: str, alpha: Optional[float]=None) -> None:
        super(FFN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        assert act in {'relu', 'gelu', 'leakyrelu'}

        self.lin_bef = nn.Linear(in_features, out_features)
        self.lin_aft = nn.Linear(out_features, in_features)
        if act == 'relu':
            self.act_l = nn.ReLU()
        elif act == 'gelu':
            self.act_l = nn.GELU()
        elif act == 'leakyrelu':
            self.act_l = nn.LeakyReLU(alpha)
        else:
            raise ValueError(f'{act} not implemented yet')

        self.reset()
        return
    
    def reset(self) -> None:
        self.lin_bef.reset_parameters()
        self.lin_aft.reset_parameters()
        return

    def forward(self, x: Tensor) -> Tensor:
        tmp = self.lin_bef(x)
        y = self.act_l(tmp)
        y = self.lin_aft(y) + x
        return y

class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(Downsample, self).__init__()
        self.in_channels = in_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels)
        )

        return
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        return x

class ViG(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: List[int],
                heads: int,
                n_classes: int,
                input_resolution: Tuple[int, int],
                reduce_factor: int=4,
                act: str='relu',
                k: int=9,
                overlapped_patch_emb: bool=True,
                **kwargs) -> None:
        super(ViG, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.n_classes = n_classes
        self.reduce_factor = reduce_factor
        self.act = act
        self.k = k
        self.overlapped_patch_emb = overlapped_patch_emb
        self.input_res = input_resolution

        self.patch_embedding = PatchEmbeddingV2(reduce_factor,
                                                in_channels,
                                                out_channels[0],
                                                overlapped=overlapped_patch_emb,
                                                act=act,
                                                reshape=False)

        self.pos_encoding = nn.Parameter(
            torch.zeros((1,
                        out_channels[0],
                        self.input_res[0] // reduce_factor,
                        self.input_res[1] // reduce_factor)
                )
            )
        
        encoder_blocks = []
        in_ch = out_channels[0]
        for idx, out_ch in enumerate(out_channels):
            if idx > 0:
                encoder_blocks.append(
                    Downsample(in_ch)
                )
            encoder_blocks.append(nn.ModuleList([
                    GrapherFC(in_ch, heads, out_ch, reconstruct_image=False, act=act, **kwargs),
                    FFN(in_ch, out_ch * 2, act)
                ])
            )
        self.encoder = nn.ModuleList(encoder_blocks)

        self.decoder = ViGDecoder(in_ch, #out_channels[-1],
                                    1024,
                                    n_classes,
                                    act,
                                    dropout_p=0.5)
        
        return

    def forward(self, x: Tensor) -> Tensor:
        # input x is of shape: B, C, H, W
        x = self.patch_embedding(x) + self.pos_encoding
        # after patch embedding, x.shape == B, C, H, W
        B, _, H, W = x.shape
        encoder_features = []
        for idx in range(len(self.encoder)):
            # prepare the data depending on the layer type and do the forward pass
            if isinstance(self.encoder[idx], Downsample):
                H, W = H // 2, W // 2
                x = self.encoder[idx](x)
            else:
                x = self.encoder[idx][0](x)
                x = self.encoder[idx][1](x)
                # x.shape == N, C
                # reshape to image
                x = graph_to_image(x, B, H, W)
                encoder_features.append(x)
                
        x = self.decoder(x)
        return x

class PyramidViG(ViG):
    def __init__(self,
                in_channels: int,
                out_channels: List[int],
                heads: int,
                n_classes: int,
                input_resolution: Tuple[int, int],
                reduce_factor: int=4,
                pyramid_reduction: int=2,
                act: str='relu',
                k: int=9,
                overlapped_patch_emb: bool=True,
                **kwargs) -> None:
        super(PyramidViG, self).__init__(in_channels,
                                        out_channels,
                                        heads,
                                        n_classes,
                                        input_resolution,
                                        reduce_factor,
                                        act,
                                        k,
                                        overlapped_patch_emb,
                                        **kwargs)
        assert pyramid_reduction > 1
        self.pyramid_reduction = pyramid_reduction
        return

    def forward(self, x: Tensor) -> Tensor:
        # input x is of shape: B, C, H, W
        x = self.patch_embedding(x) + self.pos_encoding
        # after patch embedding, x.shape == B, C, H, W
        B, _, H, W = x.shape
        encoder_features = []
        for idx in range(len(self.encoder)):
            # prepare the data depending on the layer type and do the forward pass
            if isinstance(self.encoder[idx], Downsample):
                H, W = H // 2, W // 2
                x = self.encoder[idx](x)
            else:
                y = F.avg_pool2d(x, self.pyramid_reduction, self.pyramid_reduction)
                x = self.encoder[idx][0](x, y) # GrapherFC
                x = self.encoder[idx][1](x) # FFN
                # x.shape == N, C
                # reshape to image
                x = graph_to_image(x, B, H, W)
                encoder_features.append(x)
                
        x = self.decoder(x)
        return x