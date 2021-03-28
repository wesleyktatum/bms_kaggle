import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=64,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        print('-- Queries --')
        print(q.shape)
        print('-- Keys --')
        print(k.shape)
        print('-- Values --')
        print(v.shape)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        print('-- Positional Embeddings (Q) --')
        print(q_embedding.shape)
        print('-- Positional Embeddings (K) --')
        print(k_embedding.shape)
        print('-- Positional Embeddings (V) --')
        print(v_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        print(' -- Queries dot Q Embeddings --')
        print(qr.shape)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        print('-- Keys dot K Embeddings --')
        print(kr.shape)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        print('-- Queries dot Keys --')
        print(qr.shape)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        print('-- Stacking Query Operations --')
        print(stacked_similarity.shape)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        print('-- BatchNorm Query Operations --')
        print(stacked_similarity.shape)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        print('-- SoftMax Query Operations --')
        print(similarity.shape)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        print('-- Query Product dot Values --')
        print(sv.shape)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        print('-- Query Product dot Value Embeddings --')
        print(sve.shape)
        stacked_output = torch.cat([sv, sve], dim=-1)
        print('-- Stacking Output --')
        print(stacked_output.shape)
        stacked_output = stacked_output.view(N*W, self.out_planes*2, H)
        print('-- Reshaping Stacked Output --')
        print(stacked_output.shape)
        output = self.bn_output(stacked_output)
        print('-- BatchNorm Output --')
        print(output.shape)
        output = output.view(N, W, self.out_planes, 2, H)
        print('-- Reshaping BatchNorm Output --')
        print(output.shape)
        output = output.sum(dim=-2)
        print('-- Summing Along Stacked Dimension --')
        print(output.shape)
        print('\n')

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialAttentionReducedPosEmbeddings(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=64,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 2)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes // 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        print('-- Queries --')
        print(q.shape)
        print('-- Keys --')
        print(k.shape)
        print('-- Values --')
        print(v.shape)

        # Calculate position embedding
        q_embedding = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes // 2, self.kernel_size, self.kernel_size)
        print('-- Positional Embeddings --')
        print(q_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        print(' -- Queries dot Embeddings --')
        print(qr.shape)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        print('-- Queries dot Keys --')
        print(qr.shape)
        stacked_similarity = torch.cat([qk, qr], dim=1)
        print('-- Stacking Query Operations --')
        print(stacked_similarity.shape)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 2, self.groups, H, H).sum(dim=1)
        print('-- BatchNorm Query Operations --')
        print(stacked_similarity.shape)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        print('-- SoftMax Query Operations --')
        print(similarity.shape)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        print('-- Query Product dot Values --')
        print(sv.shape)
        sv = sv.contiguous().view(N*W, self.out_planes, H)
        print('-- Reshaping Output --')
        print(sv.shape)
        output = self.bn_output(sv)view(N, W, self.out_planes, H)
        print('-- Reshaping BatchNorm Output --')
        print(output.shape)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=64):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlockReducedPosEmbeddings(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=64):
        super(AxialBlockReducedPosEmbeddings, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttentionReducedPosEmbeddings(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttentionReducedPosEmbeddings(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.5):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * s), layers[0], kernel_size=64)
        self.layer2 = self._make_layer(block, int(128 * s), layers[1], stride=2, kernel_size=64,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(256 * s), layers[2], stride=2, kernel_size=32,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(512 * s), layers[3], stride=2, kernel_size=16,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, qkv_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=64, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (batch_size, channels - 1024, xdim - 8, ydim - 8)
        x = x.permute(0, 2, 3, 1) # (batch_size, xdim, ydim, channels)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def axial18srpe(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlockReducedPosEmbeddings, [2, 2, 2, 2],
                              s=0.5, **kwargs)
    return model

def axial18s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [2, 2, 2, 2], s=0.5, **kwargs)
    return model

def axial26s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [1, 2, 4, 1], s=0.5, **kwargs)
    return model


def axial50s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    return model


def axial50m(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.75, **kwargs)
    return model


def axial50l(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=1, **kwargs)
    return model
