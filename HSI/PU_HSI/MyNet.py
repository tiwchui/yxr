import torch
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def pair1(t,v):
    return t,v

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size=9, patch_size=3, num_classes=9, dim=64, depth=6, heads=8, mlp_dim=128, pool = 'cls', channels =120, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        #print('img', img.shape)
        x = self.to_patch_embedding(img)
        #print('x', x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        #print('cls_tokens', cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        #print('x', x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        #print('x', x.shape)
        x = self.dropout(x)
        #print('x', x.shape)

        x = self.transformer(x)
        #print('x', x.shape)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #print('x', x.shape)

        x = self.to_latent(x)
        #print('x', x.shape)
        return self.mlp_head(x)


NUM_CLASS = 9


class MyNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=120, depth=1, heads=8, mlp_dim=240,
                 dropout=0.1, emb_dropout=0.1):
        super(MyNet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=240, out_channels=120, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )
        self.ViT = ViT()

        self.conv7 = nn.Conv2d(in_channels=120, out_channels=24, kernel_size=(3, 3))
        self.bn7 = nn.BatchNorm2d(24)

        self.conv8 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=(1, 1))
        self.bn8 = nn.BatchNorm2d(24)
        self.conv9 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=(1, 1))
        self.bn9 = nn.BatchNorm2d(24)

        self.conv10 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=(1, 1))
        self.bn10 = nn.BatchNorm2d(24)
        self.conv11 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=(1, 1))
        self.bn11 = nn.BatchNorm2d(24)

        self.Avg = nn.AvgPool2d(kernel_size=(7,7))
        self.linear = nn.Linear(in_features=24, out_features=num_classes)

        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(18, num_classes)  # ,
            # nn.Softmax()
        )

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        #print('x', x.shape)
        x0 = self.conv3d_features(x)
        # print('x0', x0.shape)
        x0 = rearrange(x0, 'b c h w y -> b (c h) w y')
        #print('x0', x0.shape)
        x0 = self.conv2d_features(x0)
        #print('x0', x0.shape)
        #         x = rearrange(x,'b c h w -> b (h w) c')
        #         print('x', x.shape)
        x1 = self.ViT(x0)
        #print('x1', x1.shape)

        x2 = self.conv7(x0)
        #print('x2', x2.shape)
        x2 = self.bn7(x2)
        #print('x2', x2.shape)

        x3 = self.conv8(x2)
        # print('x3', x3.shape)
        x3 = self.bn8(x3)
        #print('x3', x3.shape)
        x3 = self.conv9(x3)
        # print('x3', x3.shape)
        x3 = self.bn9(x3)
        #print('x3', x3.shape)
        x3 = torch.add(x2, x3)
        #print('x3', x3.shape)

        x4 = self.conv10(x3)
        #print('x4', x4.shape)
        x4 = self.bn10(x4)
        #print('x4', x4.shape)
        x4 = self.conv11(x4)
        #print('x4', x4.shape)
        x4 = self.bn11(x4)
        #print('x4', x4.shape)
        x4 = torch.add(x3, x4)
        #print('x4', x4.shape)

        x5 = self.Avg(x4)
        #print('x5', x5.shape)
        x5 = torch.flatten(x5, start_dim=1)
        #print('x5', x5.shape)
        x5 = self.linear(x5)
        #print('x5', x5.shape)

        x_pre = torch.cat((x1, x5), dim=1)
        #print('x_pre', x_pre.shape)
        output = self.full_connection(x_pre)
        #print('output', output.shape)
        return output




if __name__ == '__main__':
    model = MyNet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())

