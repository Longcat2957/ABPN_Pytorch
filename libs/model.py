import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, act:bool=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            stride=1, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if act else nn.Identity()
    
    def fuse_modules(self):
        try:
            torch.quantization.fuse_modules(self, [["conv", "bn", "act"]], inplace=True)
        except:
            torch.quantization.fuse_modules(self, [["conv", "bn"]], inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Concatenation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.cat((x,x,x,x,x,x,x,x,x), dim=1)

class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.clip(x, min=-128, max=127)

class DepthToSpace(nn.Module):
    def __init__(self, block_size:int=3):
        super().__init__()
        self.block_size = block_size
    
    def forward(self, x):
        n, c, h, w = x.size()
        return x.view(n, c//self.block_size**2, h * self.block_size, w * self.block_size)

        
class edgeSR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.concatenation = Concatenation()
        self.feature_extraction = nn.Sequential(
            ConvBNReLU(3, 28),
            ConvBNReLU(28, 28),
            ConvBNReLU(28, 28),
            ConvBNReLU(28, 28),
            ConvBNReLU(28, 28),
            ConvBNReLU(28, 27),
            ConvBNReLU(27, 27, False)
        )
        self.add = nn.quantized.FloatFunctional()
        self.depth_to_space = nn.PixelShuffle(upscale_factor=3)
        #self.depth_to_space = DepthToSpace()
        self.clip = Clip()
        self.final_act = nn.ReLU()
    def forward(self, lr):
        cc = self.concatenation(lr)
        fe = self.feature_extraction(lr)
        added = self.add.add(cc, fe)
        dts = self.depth_to_space(added)
        clipped = self.clip(dts)
        return self.final_act(clipped)
        
 
if __name__ == "__main__":
    cbnrelu = ConvBNReLU(3, 27)
    a = torch.randn(size=(1, 3, 32, 32))
    b = cbnrelu(a)
    print(b.shape)

    cbnrelu.eval()
    cbnrelu.fuse_modules()
    # print(cbnrelu)
    
    fullmodel = edgeSR()
    lr = torch.randn(size=(1, 3, 360, 640))
    hr = fullmodel(lr)
    print(hr.shape)
    
    x = [[1], [2], [3], [4]]
    x = torch.Tensor(x).unsqueeze(dim=-1).unsqueeze(0)
    print(x.shape)
    dts = DepthToSpace(2)
    y = dts(x)
    print(y.shape)
    print(y)
    
    ps = nn.PixelShuffle(2)
    y2 = ps(x)
    print(y2)
    
    