import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., out_indices=(9, 14, 19, 23)):
        super().__init__()
        self.out_indices = out_indices
        assert self.out_indices[-1] == depth - 1

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        out = []
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x

            if index in self.out_indices:
                out.append(x)

        return out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., out_indices=(9, 14, 19, 23)):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, out_indices=out_indices)

        self.out = Rearrange("b (h w) c->b c h w", h=image_height // patch_height, w=image_width // patch_width)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        out = self.transformer(x)

        for index, transformer_out in enumerate(out):
            # delete cls_tokens and transform output to [b, c, h, w]
            out[index] = self.out(transformer_out[:, 1:, :])

        return out



class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out



class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out



class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False) # out = (bn, 64, 128, 128)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)  # out = (bn, 128, 64, 64)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)  # out = (bn, 256, 32, 32)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)  # out = (bn, 512, 16, 16)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)  # out = (bn, 512, 8, 8)
        self.v = ViT(image_size=(256, 256), patch_size=256 // 16, dim=1024, depth=24, heads=16, mlp_dim=2048, dropout=0.1,
                emb_dropout=0.1, out_indices=(9, 14, 19, 23))
        self.conv6 = ConvBlock(1024,512)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)    # out = (bn, 512, 16, 16)
        self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)  # out = (bn, 256, 32, 32)
        self.deconv6 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)  # out = (bn, 128, 64, 64)
        self.deconv7 = DeconvBlock(num_filter * 2 * 2, num_filter)  # out = (bn, 64, 128, 128)
        self.deconv8 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)    # out = (bn, 3, 256, 256)


    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.v(x)[3]
        enc5 = enc5+self.conv6(enc6)
        dec1 = self.deconv1(enc5)
        dec1 = torch.cat([dec1, enc4], 1)
        dec2 = self.deconv5(dec1)
        dec2 = torch.cat([dec2, enc3], 1)
        dec3 = self.deconv6(dec2)
        dec3 = torch.cat([dec3, enc2], 1)
        dec4 = self.deconv7(dec3)
        dec4 = torch.cat([dec4, enc1], 1)
        dec5 = self.deconv8(dec4)
        out = torch.nn.Tanh()(dec5)

        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)



class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()
        # input_dim = 6, num_filter = 64, out_dim = 1
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)   # (bn, 64, 128, 128)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)  # (bn, 128, 64, 64)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)  # (bn, 256, 32, 32)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)    # (bn, 512, 31, 31)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)  # (bn, 1, 30, 30)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # sigmoid: (0-1)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.rand((1, 3, 256, 256)).to(device)

    model = Generator(3, 64, 3).to(device)
    output = model(input)
    print(output.shape)
    # summary(model, (3, imsize, imsize))