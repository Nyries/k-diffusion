import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Recreating ADM architecture like https://github.com/openai/guided-diffusion and using schematic from Karras et al. paper: Analyzing and Improving the Training Dynamics of Diffusion Models.

Low layer moduels like Linear and Conv2d have been recreated because Karras paper is doing improvements directly inside these layers.
As it is stated in the name this is the baseline for further improvements. """

activation_array = [] # Array used to record activation magnitudes and imported into the training file

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# Basic Layers

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels)))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        global activation_array
        x = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        activation_array.append(float(torch.mean(torch.abs(x))))
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features)))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        global activation_array
        x = F.linear(x, self.weight, self.bias)
        activation_array.append(float(torch.mean(torch.abs(x))))
        return x


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = (torch.rand(x.shape, device='cuda') > self.p).float()
            return x * mask / (1 - self.p)
        else:
            return x
        
    
class UpSample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
    

class DownSample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=1.0/self.scale_factor, mode=self.mode)

# Norm Layer

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        G = self.num_groups
        assert C % G == 0, "num_channels must be divisible by num_groups"

        x = x.view(N, G, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var= x.var(dim=-1, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)

        if self.affine:
            x = x * self.weight + self.bias

        return x
    
# Attention block

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=64, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.group_norm = GroupNorm(num_groups=32, num_channels=in_channels)
        self.qkv_conv = Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=bias)
        self.out_conv = zero_module(Conv2d(in_channels, in_channels, kernel_size=1, bias=bias))

    def forward(self, x:torch.Tensor) -> torch.Tensor :
        batch_size, channels, height, width = x.shape
        skip = x
        # GroupNorm
        x = self.group_norm(x)

        # Convolution
        qkv = self.qkv_conv(x) # (batch_size, 3 * in_channels, height, width)

        # Reshape & Split
        qkv = qkv.view(batch_size, self.num_heads, 3*self.head_dim, height * width)
        query, key, value = torch.split(qkv, self.head_dim, dim=2) # Each (batch_size, num_heads, head_dim, height * width)
        key = key.permute(0, 1, 3, 2) # (batch_size, num_heads, height * width, head_dim)
        attention_scores = torch.matmul(query, key) / (self.head_dim ** 0.5) # (batch_size, num_heads, height * width, height * width)
        # Softmax
        attention_scores = F.softmax(attention_scores, dim=1) # (batch_size, num_heads, height * width, height * width)

        # Matmul
        out = torch.matmul(attention_scores, value) # (batch_size, num_heads, height * width, head_dim)
        out = out.view(batch_size, self.in_channels, height, width)
        out = self.out_conv(out)
        return out + skip

# Embedding Layers

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    

class Embedding(nn.Module):
    def __init__(self, noise_dim, hidden_dim=768, bias=True):
        super().__init__()

        self.pos_emb = PositionalEmbedding(noise_dim)
        self.linear1 = Linear(noise_dim, hidden_dim, bias=bias)
        self.linear2 = Linear(hidden_dim, hidden_dim, bias=bias)
        self.silu= SiLU()
    
    def forward(self, noise_level):
        noise_embedding = self.pos_emb(noise_level)
        x = self.silu(self.linear1(noise_embedding))
        x = self.silu(self.linear2(x))
        return x

# Blocks part of the Unet
 
class Input(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor): 
        return self.conv(x)
    
    def __repr__(self):
        return f"Input with in_channels: {self.in_channels}, out_channels: {self.out_channels}"
    

class Output(nn.Module):
    def __init__(self, in_channels, out_channels, num_group=32, bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.group_norm = GroupNorm(num_group, in_channels)
        self.silu = SiLU()
        self.conv = zero_module(Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.group_norm(x)
        x = self.silu(x)
        x = self.conv(x)
        return x

    def __repr__(self):
        return f"Output with in_channels: {self.in_channels}, out_channels: {self.out_channels}"
    

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=768, prob_dropout=0.5, num_groups=32, attention=True, downsample=True, bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.attention = attention
        self.downsample = downsample

        assert not (attention and downsample), "Can't do both attention and downsample"
        
        #Direct Line
        if self.downsample:
            self.down2 = DownSample(scale_factor=2)
        self.conv_skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=bias)
        if self.attention:
            self.attn = SelfAttention(out_channels, bias=bias)
        
        #Embedding
        self.linear_emb = Linear(in_features=hidden_dim, out_features=out_channels*2, bias=bias)

        self.group_norm1 = GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.silu = SiLU()
        if downsample:
            self.down1 = DownSample(scale_factor=2)
        self.conv1 = Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        self.group_norm2 = GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = Dropout(p=prob_dropout)
        self.conv3 = zero_module(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias))
    
    def forward(self, x:torch.Tensor, emb) -> torch.Tensor:
        h = self.group_norm1(x)
        h = self.silu(h)
        if self.downsample:
            h = self.down1(h)
            x = self.down2(x)
        x = self.conv_skip(x)
        h = self.conv1(h)
        h = self.group_norm2(h)
        emb = self.linear_emb(emb)
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        h = h * (1 + scale) + shift
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv3(h)
        x = h + x   
        if self.attention:
            x = self.attn(x)
        return x
    
    def __repr__(self):
        return f"Encoder with in_channels: {self.in_channels}, out_channels: {self.out_channels}, downsample: {self.downsample}, attention: {self.attention}"


class Decoder(nn.Module):
    def __init__(self, in_channels:int, skip_channels:int, out_channels, hidden_dim=768, prob_dropout=0.5, num_groups=32, attention=True, upsample=True, bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = 0 if skip_channels is None else skip_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.attention = attention
        self.upsample = upsample
        assert not (attention and upsample), "Can't do both attention and downsample"
        
        #Direct Line
        if self.upsample:
            self.down2 = UpSample(scale_factor=2)
        self.conv2 = Conv2d(in_channels=in_channels+self.skip_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        if self.attention:
            self.attn = SelfAttention(out_channels, bias=bias)
        
        #Embedding
        self.linear_emb = Linear(in_features=hidden_dim, out_features=out_channels*2, bias=bias)

        self.group_norm1 = GroupNorm(num_groups=num_groups, num_channels=in_channels+self.skip_channels)
        self.silu = SiLU()
        if upsample:
            self.down1 = UpSample(scale_factor=2)
        self.conv1 = Conv2d(in_channels=in_channels+self.skip_channels,out_channels=out_channels,kernel_size=3, padding=1 ,bias=bias)
        self.group_norm2 = GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = Dropout(p=prob_dropout)
        self.conv3 = zero_module(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1 , bias=bias))
    
    def forward(self, x:torch.Tensor, emb, skip=None) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        h = self.group_norm1(x)
        h = self.silu(h)
        if self.upsample:
            h = self.down1(h)
            x = self.down2(x)
        x = self.conv2(x)
        h = self.conv1(h)
        h = self.group_norm2(h)
        emb = self.linear_emb(emb)
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        h = h * (1 + scale) + shift
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv3(h)
        x = h + x
        if self.attention:
            x = self.attn(x)
        return x
    
    def __repr__(self):
        return f"Decoder with in_channels: {self.in_channels}, out_channels: {self.out_channels}, skip_channels: {self.skip_channels}, upsample: {self.upsample}, attention: {self.attention}"

# Unet 

class ConfigADenoiser(nn.Module):
    def __init__(self, channels:tuple, resolutions:tuple, attn_res:tuple, depths: int, prob_dropout, num_group=32, bias=True):
        super().__init__()
        assert len(channels) == len(resolutions), "Tuple size don't match"
        self.attn_res = attn_res
        self.channels = channels
        self.resolutions = resolutions
        self.depths = depths
        self.skip_channels = []
        self.skips = []
        # Embedding
        self.embedding = Embedding(noise_dim=192, hidden_dim=768, bias=True)
        # Input / Output
        self.input = Input(3, channels[0], bias=bias)
        self.skip_channels.append(channels[0])
        self.output = Output(channels[0], 3, num_group=num_group, bias=bias)

        # Encoders
        self.encoders =  nn.Sequential()
        for (j,channel), resolution in zip(enumerate(self.channels),self.resolutions):
            for i in range(self.depths):
                if i == self.depths - 1:
                    if channel==self.channels[-1]:
                        self.skip_channels.pop()
                        pass
                    else:
                        self.encoders.append(Encoder(in_channels=channel, out_channels=channel, prob_dropout=prob_dropout, num_groups=num_group, attention=False, downsample=True, bias=bias))
                elif resolution not in self.attn_res:
                    self.encoders.append(Encoder(in_channels=channel, out_channels=channel, prob_dropout=prob_dropout, num_groups=num_group, attention=False, downsample=False, bias=bias))
                elif resolution in self.attn_res: 
                    if self.encoders[-1].downsample:
                        self.encoders.append(Encoder(in_channels=self.channels[j-1], out_channels=channel, prob_dropout=prob_dropout, num_groups=num_group, attention=True, downsample=False, bias=bias))
                    else:
                        self.encoders.append(Encoder(in_channels=channel, out_channels=channel, prob_dropout=prob_dropout, num_groups=num_group, attention=True, downsample=False, bias=bias))
                self.skip_channels.append(channel)
                
        for encoder in self.encoders:
            print(repr(encoder))

        # Up
        self.decoders = nn.Sequential(Decoder(in_channels=self.channels[-1], skip_channels=None, out_channels=self.channels[-1], upsample=False, attention=True, bias=bias, num_groups=num_group, prob_dropout=prob_dropout),
                                      Decoder(in_channels=channel, skip_channels=None, out_channels=channel, upsample=False, attention=False, bias=bias, num_groups=num_group, prob_dropout=prob_dropout))
        for (j,channel), res in zip(enumerate(reversed(self.channels)),reversed(self.resolutions)):
            for i in range(self.depths+1):
                if i == self.depths:
                    if channel == self.channels[0]:
                        pass
                    else:
                        self.decoders.append(Decoder(in_channels=channel, skip_channels=None, out_channels=channel, upsample=True, attention=False, bias=bias, num_groups=num_group, prob_dropout=prob_dropout))
                elif res in self.attn_res:
                    if self.decoders[-1].upsample:
                        self.decoders.append(Decoder(in_channels=self.channels[-j], skip_channels=self.skip_channels.pop(), out_channels=channel, upsample=False, attention=True, bias=bias, num_groups=num_group, prob_dropout=prob_dropout))
                    else:
                        self.decoders.append(Decoder(in_channels=channel, skip_channels=self.skip_channels.pop(), out_channels=channel, upsample=False, attention=True, bias=bias, num_groups=num_group, prob_dropout=prob_dropout))
                elif res not in self.attn_res:
                    if self.decoders[-1].upsample:
                        self.decoders.append(Decoder(in_channels=self.channels[-j], skip_channels=self.skip_channels.pop(), out_channels=channel, upsample=False, attention=False, bias=bias, num_groups=num_group, prob_dropout=prob_dropout))
                    else:
                        self.decoders.append(Decoder(in_channels=channel, skip_channels=self.skip_channels.pop(), out_channels=channel, upsample=False, attention=False, bias=bias, num_groups=num_group, prob_dropout=prob_dropout))
        
        for decoder in self.decoders:
            print(repr(decoder))

        
    def param_groups(self, base_lr=2e-4):
        wd_names = []
        for name, _ in self.named_parameters():
            if name.endswith(".weight"):
                wd_names.append(name)
        wd, no_wd = [], []
        for name, param in self.named_parameters():
            if name in wd_names:
                wd.append(param)
            else:
                no_wd.append(param)
        groups = [
            {"params": wd, "lr": base_lr},
            {"params": no_wd, "lr": base_lr, "weight_decay": 0.0},
        ]
        return groups
    
    def forward(self, x:torch.Tensor, sigma, aug_cond=None, class_cond=None, mapping_cond=None) -> torch.Tensor:
        global activation_array
        c_noise = torch.log(sigma) / 4
        emb = self.embedding(c_noise)
        x= self.input(x)
        self.skips.append(x)
        for encoder in self.encoders:
            # print(f'channels entrée et sortie {encode.in_channels}, {encode.out_channels}')
            x = encoder(x, emb)
            activation_array.append(float(torch.mean(torch.abs(x))))
            self.skips.append(x)
        for decoder in self.decoders:
            x = decoder(x, emb, skip=self.skips.pop()) if decoder.skip_channels != 0 else decoder(x, emb)    
            activation_array.append(float(torch.mean(torch.abs(x)))) 
        x = self.output(x)   
        return x
    
# model = ConfigADenoiser([192,384,576,768],[64, 32, 16, 8],[32,16,8],4, prob_dropout=0.5)