import torch
import torch.nn as nn
import math
from timm.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

class Downsample3D(nn.Module):
    def __init__(self,dim,keep_dim=False,):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv3d(dim, dim_out, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
            nn.BatchNorm3d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

class PConvBlock3D(nn.Module):
    def __init__(self, dim,n_div):
        super().__init__()
        self.pconv = APConv3D(dim, n_div=n_div)
        #self.pconv = PConv3D(dim, n_div=n_div)
        self.dwconv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
    def forward(self, x):
        x = self.pconv(x)
        x = self.dwconv(x)
        return x

class PConv3D(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv3d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
    
class APConv3D(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim = dim
        self.dim_conv3 = dim // n_div
        self.partial_conv3 = nn.Conv3d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv1d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def select_channels(self, x):
        B, C, D, H, W = x.shape
        x1 = x.clone()
        x1 = self.avg_pool(x1).view(B, C, -1)
        channel_weights = self.channel_attention(x1)
        x_flat = x.view(B, C, D, -1)
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        x_norm = x_norm.permute(0, 2, 1, 3)  
        sim = torch.matmul(x_norm, x_norm.transpose(-2, -1)) 
        sim = sim.mean(dim=-1)  
        sim = sim.permute(0, 2, 1) 
        channel_importance = self.alpha * (1 - sim) + (1 - self.alpha) * channel_weights
        _, selected_indices = torch.topk(channel_importance, self.dim_conv3, dim=1) 
        
        return selected_indices

    def forward(self, x):
        selected_indices = self.select_channels(x)  
        B, C, D, H, W = x.shape
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, self.dim_conv3, D)
        frame_idx = torch.arange(D, device=x.device).view(1, 1, D).expand(B, self.dim_conv3, -1)
        selected_channels = x[batch_idx, selected_indices, frame_idx]  
        selected_channels_conv = self.partial_conv3(selected_channels)
        x_out = x.clone()
        x_out[batch_idx, selected_indices, frame_idx] = selected_channels_conv

        return x_out
    
class ConvBlock3D(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3,
                 apconv=False,):
        super().__init__()
        if apconv:
            print("Using APConv3D")
            self.conv1 = PConvBlock3D(dim, n_div=4)
            self.conv2 = PConvBlock3D(dim, n_div=4)
        else:
            print("Using standard Conv3D")
            self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x

class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class LHMambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
        num_heads=8,
        qkv_bias=False, 
        qk_norm=False, 
        proj_drop=0., 
        attn_drop=0.,
        norm_layer=nn.LayerNorm, 
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand 
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)  
        self.x_proj = nn.Linear(
            self.d_inner//3, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//3, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//3, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//3,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//3, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//3,
            out_channels=self.d_inner//3,
            bias=conv_bias//3,
            kernel_size=d_conv,
            groups=self.d_inner//3,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//3,
            out_channels=self.d_inner//3,
            bias=conv_bias//3,
            kernel_size=d_conv,
            groups=self.d_inner//3,
            **factory_kwargs,
        )
        self.conv1d_y = nn.Conv1d(
            in_channels=self.d_inner//3,
            out_channels=self.d_inner//3,
            bias=conv_bias//3,
            kernel_size=d_conv,
            groups=self.d_inner//3,
            **factory_kwargs,
        )
        self.att = Attention(
            d_model//3,
            num_heads=num_heads,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            norm_layer = norm_layer,
            )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xzy = self.in_proj(hidden_states)
        xzy = rearrange(xzy, "b l d -> b d l")
        x, z, y = xzy.chunk(3, dim=1) 
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//3))
        A = -torch.exp(self.A_log.float()) 
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen) 
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() 
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() 

        x = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)

        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//3))
        y = F.silu(F.conv1d(input=y, weight=self.conv1d_y.weight, bias=self.conv1d_y.bias, padding='same', groups=self.d_inner//3))
        y = self.att(rearrange(y, "b d l -> b l d")) 
        y =rearrange(y, "b l d -> b d l")
        output = torch.cat([x, z, y], dim=1)
        output = rearrange(output, "b d l -> b l d")
        output = self.out_proj(output)
        return output
 
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MambaBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            print(f"Using Attention for block {counter}")
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            print(f"Using MambaVisionMixer for block {counter}")
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1,
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    

class LHMambaBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        print("Using MobileMambaMixer")
        self.mixer = LHMambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1,
                                          num_heads=num_heads,
                                          qkv_bias=qkv_bias,
                                          qk_norm=qk_scale,
                                          attn_drop=attn_drop,
                                          proj_drop=drop,
                                          norm_layer=norm_layer,
                                          )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class MambaVisionLayer(nn.Module):
    def __init__(self,
                 dim, 
                 depth, 
                 num_heads, 
                 window_size, 
                 mlp_ratio=4.,
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0., 
                 layer_scale=None, 
                 layer_scale_conv=None, 
                 transformer_blocks = [], 
                 lhmamba=False, 
                 num_frames=30, 
            ):
        super().__init__()

        if lhmamba:
            self.blocks = nn.ModuleList([LHMambaBlock(dim=dim,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop=drop,
                                            attn_drop=attn_drop,
                                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                            layer_scale=layer_scale,
                                            )
                                            for i in range(depth//2)])
        else:
            self.blocks = nn.ModuleList([MambaBlock(dim=dim,
                                            counter=i, 
                                            transformer_blocks=transformer_blocks,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop=drop,
                                            attn_drop=attn_drop,
                                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                            layer_scale=layer_scale,
                                            )
                                            for i in range(depth)])

        self.do_gt = False
        self.window_size = window_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1,  dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14 + 1,  dim)) 
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // 1,  dim)) 
        self.pos_drop = nn.Dropout(p=0)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.view(B * T, C, H, W)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0: 
            x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
            _, _, Hp, Wp = x.shape 
        else:
            Hp, Wp = H, W
        x = window_partition(x, self.window_size)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
    
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        output = x[:, 0, :] 
        return output

class LHMambaVideo(nn.Module):
    def __init__(self,
                 dim=60,
                 in_dim=32,
                 depths= [1, 3, 12], 
                 window_size = [8, 8, 14], 
                 mlp_ratio = 4, 
                 num_heads=[2, 4, 8], 
                 drop_path_rate=0.2, 
                 in_chans=3,
                 num_classes = 4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 apconv=False, 
                 lhmamba=False, 
                 num_frames=32, 
                 ):
        super().__init__()
        print(f"dim: {dim}, in_dim: {in_dim}, depths: {depths}, window_size: {window_size}, num_heads: {num_heads}, drop_path_rate: {drop_path_rate}")
        num_features = int(dim * 2 ** (len(depths) - 1))
        print(f"num_features: {num_features}")
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed3D(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        drop_path0 = dpr[sum(depths[:0]):sum(depths[:0 + 1])]
        self.layer0 = nn.ModuleList([ConvBlock3D(dim=dim,
                                                drop_path=drop_path0[i] if isinstance(drop_path0, list) else drop_path0,
                                                layer_scale=layer_scale_conv,
                                                apconv=apconv,)
                                                for i in range(depths[0])])
        self.downsample0 = Downsample3D(dim=int(dim),keep_dim=False)

        drop_path1 = dpr[sum(depths[:1]):sum(depths[:1 + 1])]
        self.layer1 = nn.ModuleList([ConvBlock3D(dim=dim * 2 ** 1 ,
                                                drop_path=drop_path1[i] if isinstance(drop_path1, list) else drop_path1,
                                                layer_scale=layer_scale_conv,
                                                apconv=apconv,)
                                                for i in range(depths[1])])
        self.downsample1 = Downsample3D(dim=int(dim * 2 ** 1),keep_dim=False)

        self.layer2 = MambaVisionLayer(dim=int(dim * 2 ** 2),
                                     depth=depths[2], 
                                     num_heads=num_heads[2], 
                                     window_size=window_size[2], 
                                     qkv_bias=qkv_bias, 
                                     qk_scale=qk_scale, 
                                     mlp_ratio = mlp_ratio, 
                                     drop=drop_rate, 
                                     attn_drop=attn_drop_rate, 
                                     drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])], 
                                     layer_scale=layer_scale, 
                                     layer_scale_conv=layer_scale_conv, 
                                     transformer_blocks=list(range(depths[2]//2+1, depths[2])) if depths[2]%2!=0 else list(range(depths[2]//2, depths[2])), # 指定哪些 block 用 Transformer（或 Attention），其余用 Mamba（或其它混合结构）
                                     lhmamba=lhmamba,
                                     num_frames=num_frames,
                                     )

        self.head = nn.Sequential(nn.LayerNorm(num_features),nn.Linear(num_features, 64), nn.GELU(),
                                                  nn.Dropout(0.5),nn.Linear(64, num_classes))

        self.apply(self._init_weights)    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        for _, blk in enumerate(self.layer0):
            x = blk(x)
        x = self.downsample0(x)
 
        for _, blk in enumerate(self.layer1):
            x = blk(x)
        x = self.downsample1(x)

        x = self.layer2(x)

        x = self.head(x)
        return x

if __name__ == "__main__":
    device = "cuda"
    dim = 96  #(12,24,48,60,96)
    depths = [1, 3, 8]
    APConv = True  
    LHMamba = True  
    num_frames = 32 
    model = LHMambaVideo(dim = dim, depths=depths, num_classes=4, apconv=APConv, lhmamba=LHMamba, num_frames=num_frames)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    input_tensor = torch.randn(1, 3, 32, 224, 224)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    print(output.shape)
    