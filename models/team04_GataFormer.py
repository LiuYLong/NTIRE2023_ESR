import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class MeanShift(nn.Module):
    r"""MeanShift for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, rgb_range: int, sign: int = -1, data_type: str = 'DIV2K') -> None:
        super(MeanShift, self).__init__()

        self.sign = sign

        self.rgb_range = rgb_range
        self.rgb_std = (1.0, 1.0, 1.0)
        if data_type == 'DIV2K':
            # RGB mean for DIV2K 1-800
            self.rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == 'DF2K':
            # RGB mean for DF2K 1-3450
            self.rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f'Unknown data type for MeanShift: {data_type}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.Tensor(self.rgb_std)
        weight = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        bias = self.sign * self.rgb_range * torch.Tensor(self.rgb_mean) / std
        return f.conv2d(input=x, weight=weight.type_as(x), bias=bias.type_as(x))

class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)

class LayerNorm4D(nn.Module):
    r"""LayerNorm for 4D input.

    Modified from https://github.com/sail-sg/poolformer.

    Args:
        num_channels (int): Number of channels expected in input
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5

    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(self, upscale: int, in_channels: int,
                 out_channels: int, upsample_mode: str = 'csr') -> None:

        layer_list = list()
        if upsample_mode == 'csr':  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'Upscale {upscale} is not supported.')
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == 'lsr':  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale ** 2)))
            layer_list.append(nn.PixelShuffle(upscale))
        elif upsample_mode == 'denoising' or upsample_mode == 'deblurring' or upsample_mode == 'deraining':
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        else:
            raise ValueError(f'Upscale mode {upscale} is not supported.')

        super(Upsampler, self).__init__(*layer_list)








class GateModule(nn.Module):
    r"""Gate Module.
    """

    def __init__(self, planes: int, block_type: str) -> None:
        super(GateModule, self).__init__()

        if block_type == '33conv':
            block = nn.Conv2d(planes, planes, (3, 3), padding=(1, 1),
                              groups=planes, bias=False)
        elif block_type == '77conv':
            block = nn.Conv2d(planes, planes, (7, 7), padding=(3, 3),
                              groups=planes, bias=False)
        elif block_type == '1111conv':
            block = nn.Conv2d(planes, planes, (11, 11), padding=(5, 5),
                              groups=planes, bias=False)
        elif block_type == '1515conv':
            block = nn.Conv2d(planes, planes, (15, 15), padding=(7, 7),
                              groups=planes, bias=False)
        elif block_type == '5577conv':
            block = nn.Sequential(
                nn.Conv2d(planes, planes, (5, 5), padding=(2, 2),
                          groups=planes, bias=False),
                nn.Conv2d(planes, planes, (7, 7), padding=(3, 3),
                          groups=planes, bias=False) )

        elif block_type == '33pool':
            block = nn.AvgPool2d((3, 3), padding=(5, 5))
        else:
            block = nn.Identity()

        self.main_branch = Conv2d1x1(in_channels=planes, out_channels=planes)

        self.gate_branch = Conv2d1x1(in_channels=planes, out_channels=planes)
        self.block = block

        self.tail_conv = Conv2d1x1(in_channels=planes, out_channels=planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_output = self.main_branch(x)

        gate_output = self.gate_branch(x)
        gate_output = self.block(gate_output)

        return self.tail_conv(main_output * gate_output)


class GateLayer(nn.Module):
    r"""Gate Layer.
    """

    def __init__(self, planes: int, block_type: str) -> None:
        super(GateLayer, self).__init__()

        self.gate_module = GateModule(planes=planes, block_type=block_type)
        self.norm = LayerNorm4D(num_channels=planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate_module(self.norm(x))


class GateFormer(nn.Module):
    r"""GateFormer
    """

    def __init__(self, upscale: int = 4, num_in_ch: int = 3, num_out_ch: int = 3, task: str = 'lsr',
                 planes: int = 37, num_layers: int = 28, block_type: str = '1111conv', inter_rate: int = 2) -> None:
        super(GateFormer, self).__init__()

        self.sub_mean = MeanShift(255, sign=-1, data_type='DF2K')
        self.add_mean = MeanShift(255, sign=1, data_type='DF2K')

        self.head = Conv2d3x3(num_in_ch, planes)

        self.body = nn.Sequential(*[
            GateLayer(planes=planes,
                      block_type=None if (i + 1) % inter_rate == 0 else block_type)
            for i in range(num_layers)
        ])

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x


# if __name__ == '__main__':
#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
#     net = GateFormer(upscale=4, num_in_ch=3, num_out_ch=3, task='lsr',
#                      planes=54, num_layers=48, block_type='1111conv', inter_rate=2)
#     print(count_parameters(net))
#
#     data = torch.randn(1, 3, 48, 48)
#     print(net(data).size())