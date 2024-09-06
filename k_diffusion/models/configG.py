import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""File containing the architecture of configuration D of Karra et al. paper: Analyzing and Improving the Training Dynamics of Diffusion Models.
This configuration's is called magnitude-preserving fixed-function layers.
This configuration has the objective to standardize the magnitude of activations (see paper for equations (page 29)). Every new/modified functions have MP as prefix."""

activation_array = [] # Array used to record activation magnitudes and imported into the training file

# Base Layers

def normalize(x:torch.Tensor, eps=1e-4) -> torch.Tensor:
    dim = list(range(1, x.ndim ))
    n = torch.linalg.vector_norm(x, dim=dim , keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, eps=1e-4, activation_normalization=True, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.activation_normalization = activation_normalization
        self.eps = eps

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels)))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        global activation_array
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = normalize(self.weight) / np.sqrt(fan_in)
        x = F.conv2d(x, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        activation_array.append(float(torch.mean(torch.abs(x))))
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-4, activation_normalization=True, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_normalization = activation_normalization
        self.eps= eps

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features)))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        global activation_array
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = normalize(self.weight) / np.sqrt(fan_in)
        x = F.linear(x, weight=weight, bias=self.bias)
        activation_array.append(float(torch.mean(torch.abs(x))))
        return x

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = (torch.rand(x.shape, device=x.device) > self.p).float()
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
    
# Layers added/modified from configG

class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.gain * x


class MPSiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.silu(x) / 0.596


class MPSum(nn.Module):
    def __init__(self, t=0.5) -> None:
        super().__init__()
        self.t = t

    def forward(self, x, y):
        return ((1 - self.t) * x + self.t * y) / math.sqrt((1 - self.t) ** 2 + self.t ** 2)
    

class MPCat(nn.Module):
    def __init__(self, t=0.5) -> None:
        super().__init__()
        self.t =t

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        Nx = x.numel()
        Ny = y.numel()
        scalex = (1 - self.t) / math.sqrt(Nx)
        scaley = self.t / math.sqrt(Ny)
        scale = math.sqrt((Nx + Ny) / ((1 - self.t) ** 2 + self.t ** 2 ))
        cat = torch.cat([scalex * x, scaley * y], dim=1)
        return scale * cat

# Norm layer

class PixelNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-4):
        super().__init__()
        self.num_channels =num_channels
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        N, C, *_= x.shape

        x = torch.sqrt(torch.tensor(C)) * x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps)

        return x

# Attention layer

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=64, t=0.3, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.qkv_conv = Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=bias)
        self.pixel_norm = PixelNorm(num_channels=in_channels*3)
        self.out_conv = Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)
        self.add = MPSum(t)

    def forward(self, x:torch.Tensor) -> torch.Tensor :
        batch_size, channels, height, width = x.shape
        skip = x

        # Convolution
        qkv = self.qkv_conv(x) # (batch_size, 3 * in_channels, height, width)

        # Reshape 
        qkv = qkv.view(batch_size, 3*self.head_dim, self.num_heads, height * width)
        # Pixel Norm
        qkv = self.pixel_norm(qkv)
        qkv = qkv.permute(0, 2, 1, 3) # (batch_size, num_heads, 3 * head_dim, height * width)
        # Split
        query, key, value = torch.split(qkv, self.head_dim, dim=2) # Each (batch_size, num_heads, head_dim, height * width)
        key = key.permute(0, 1, 3, 2) # (batch_size, num_heads, height * width, head_dim)
        attention_scores = torch.matmul(query, key) / (self.head_dim ** 0.5) # (batch_size, num_heads, height * width, height * width)
        # Softmax
        attention_scores = F.softmax(attention_scores, dim=1) # (batch_size, num_heads, height * width, height * width)

        # Matmul
        out = torch.matmul(attention_scores, value) # (batch_size, num_heads, height * width, head_dim)
        out = out.view(batch_size, self.in_channels, height, width)
        out = self.out_conv(out)
        return self.add(out, skip)

# Embedding Layers

class MPFourierFeatures(nn.Module):
    def __init__(self, out_channels, bias=True) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.bias = bias
        self.frequency = torch.randn([1,out_channels], device='cuda')
        if self.bias:
            self.phi = torch.rand([1, out_channels], device='cuda')

    def forward(self, x: torch.Tensor):
        """x shape must be [batch_size]"""
        x = x[..., None]
        nu = 2 * math.pi * (x @ self.frequency  + self.phi) if self.bias else 2 * math.pi * self.frequency * x
        return math.sqrt(2) * torch.cos(nu)


class Embedding(nn.Module):
    def __init__(self, noise_dim, hidden_dim=768, bias=False):
        super().__init__()

        self.fourier_emb = MPFourierFeatures(out_channels=noise_dim)
        self.linear1 = Linear(noise_dim, hidden_dim, bias=bias)
        self.silu= MPSiLU()
    
    def forward(self, noise_level:torch.Tensor) -> torch.Tensor:
        noise_embedding = self.fourier_emb(noise_level)
        x = self.silu(self.linear1(noise_embedding))
        return x
    
# BLock part of the Unet

class Input(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = Conv2d(in_channels+1, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        batch_size, _, height, width = x.shape
        constant_channel = torch.ones(batch_size, 1, height, width).to(x.device)
        x = torch.cat([x, constant_channel], dim=1)
        return self.conv(x)
    
    def __repr__(self):
        return f"Input with in_channels: {self.in_channels}, out_channels: {self.out_channels}"


class Output(nn.Module):
    def __init__(self, in_channels, out_channels, num_group=32, bias=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.gain = Gain()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.gain(self.conv(x))

    def __repr__(self):
        return f"Output with in_channels: {self.in_channels}, out_channels: {self.out_channels}"


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=768, prob_dropout=0.5, num_groups=32, attention=True, downsample=True, t=0.3,  bias=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.attention = attention
        self.downsample = downsample

        assert not (attention and downsample), "Can't do both attention and downsample"
        
        #Direct Line
        if self.downsample:
            self.down = DownSample(scale_factor=2)
        if self.in_channels != self.out_channels:
            self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=bias)
        if self.attention:
            self.attn = SelfAttention(out_channels, bias=bias)
        self.pixnorm = PixelNorm(num_channels=out_channels)
        
        #Embedding
        self.linear_emb = Linear(in_features=hidden_dim, out_features=out_channels, bias=bias)
        self.gain = Gain()

        self.silu1 = MPSiLU()
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        self.silu2 = MPSiLU()
        self.dropout = Dropout(p=prob_dropout)
        self.conv3 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        self.add = MPSum(t=t)
    
    def forward(self, x:torch.Tensor, emb) -> torch.Tensor:
        if self.downsample:
            x = self.down(x)
        if self.in_channels != self.out_channels:
            x = self.conv1(x)
        x = self.pixnorm(x)
        h = self.silu1(x)
        h = self.conv2(h)
        emb = self.gain(self.linear_emb(emb))
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
        h = h * (1 + emb)
        h = self.silu2(h)
        h = self.dropout(h)
        h = self.conv3(h)
        x = self.add(x, h)
        if self.attention:
            x = self.attn(x)
        return x
    
    def __repr__(self):
        return f"Encoder with in_channels: {self.in_channels}, out_channels: {self.out_channels}, downsample: {self.downsample}, attention: {self.attention}"


class Decoder(nn.Module):
    def __init__(self, in_channels:int, skip_channels:int, out_channels, hidden_dim=768, prob_dropout=0.5, num_groups=32, attention=True, upsample=True, t=0.3, bias=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = 0 if skip_channels is None else skip_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.attention = attention
        self.upsample = upsample
        assert not (attention and upsample), "Can't do both attention and downsample"
        
        #Direct Line
        if skip_channels != 0:
            self.cat = MPCat(t=t)
        if self.upsample:
            self.down = UpSample(scale_factor=2)
        if self.in_channels+self.skip_channels != self.out_channels:
            self.conv_skip = Conv2d(in_channels=in_channels+self.skip_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        if self.attention:
            self.attn = SelfAttention(out_channels, bias=bias)
        
        #Embedding
        self.linear_emb = Linear(in_features=hidden_dim, out_features=out_channels, bias=bias)
        self.gain = Gain()

        self.silu1 = MPSiLU()
        self.conv1 = Conv2d(in_channels=in_channels+self.skip_channels, out_channels=out_channels, kernel_size=3, padding=1 ,bias=bias)
        self.silu2 = MPSiLU()
        self.dropout = Dropout(p=prob_dropout)
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1 , bias=bias)
        self.add = MPSum(t=t)
    
    def forward(self, x:torch.Tensor, emb, skip:torch.Tensor=None) -> torch.Tensor:
        if skip is not None:
            x = self.cat(x, skip)
        if self.upsample:
            x = self.down(x)
        h = self.silu1(x)
        h = self.conv1(h)
        emb = self.gain(self.linear_emb(emb))

        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
        h = h * (1 + emb)
        h = self.silu2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels+self.skip_channels != self.out_channels:
            x = self.conv_skip(x)
        x = self.add(x, h)
        if self.attention:
            x = self.attn(x)
        return x
    
    def __repr__(self):
        return f"Decoder with in_channels: {self.in_channels}, out_channels: {self.out_channels}, skip_channels: {self.skip_channels}, upsample: {self.upsample}, attention: {self.attention}"

# Unet

class ConfigGDenoiser(nn.Module):
    def __init__(self, channels:tuple, resolutions:tuple, attn_res:tuple, depths: int, prob_dropout:float, num_group=32, bias=False):
        super().__init__()
        assert len(channels) == len(resolutions), "Tuple size don't match"
        self.attn_res = attn_res
        self.channels = channels
        self.resolutions = resolutions
        self.depths = depths
        self.skip_channels = []
        self.skips = []
        # Embedding
        self.embedding = Embedding(noise_dim=192, hidden_dim=768, bias=False)
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
                    if len(self.encoders) !=0 and self.encoders[-1].downsample:
                        self.encoders.append(Encoder(in_channels=self.channels[j-1], out_channels=channel, prob_dropout=prob_dropout, num_groups=num_group, attention=False, downsample=False, bias=bias))
                    else:
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
    
    def forward(self, x:torch.Tensor, sigma, **kwargs) -> torch.Tensor:
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
    