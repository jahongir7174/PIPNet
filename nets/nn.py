import math

import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, s):
        super().__init__()
        if in_ch == out_ch:
            self.conv1 = torch.nn.MaxPool2d(1, s)
        else:
            self.conv1 = Conv(in_ch, out_ch, torch.nn.Identity(), 1, s)
        self.conv2 = torch.nn.Sequential(torch.nn.BatchNorm2d(in_ch),
                                         Conv(in_ch, out_ch, torch.nn.PReLU(out_ch), 3, 1, 1),
                                         Conv(out_ch, out_ch, torch.nn.Identity(), 3, s, 1))

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)


class PIPNet(torch.nn.Module):
    def __init__(self, args, params, depth,
                 mean_indices, reverse_index1, reverse_index2, max_len):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        width = [3, 64, 128, 256, 512]
        depth = {'18': [2, 2, 2, 2], '50': [3, 4, 14, 3], '100': [3, 13, 30, 3]}[depth]

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.PReLU(width[1]), 3, 2, 1))
        # p2/4
        for i in range(depth[0]):
            if i == 0:
                self.p2.append(Residual(width[1], width[1], 2))
            else:
                self.p2.append(Residual(width[1], width[1], 1))
        # p3/8
        for i in range(depth[1]):
            if i == 0:
                self.p3.append(Residual(width[1], width[2], 2))
            else:
                self.p3.append(Residual(width[2], width[2], 1))
        # p4/16
        for i in range(depth[2]):
            if i == 0:
                self.p4.append(Residual(width[2], width[3], 2))
            else:
                self.p4.append(Residual(width[3], width[3], 1))
        # p5/32
        for i in range(depth[3]):
            if i == 0:
                self.p5.append(Residual(width[3], width[4], 2))
            else:
                self.p5.append(Residual(width[4], width[4], 1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.args = args
        self.params = params
        self.max_len = max_len
        self.mean_indices = mean_indices
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2

        self.score = torch.nn.Conv2d(width[4], params['num_lms'], 1)
        self.offset_x = torch.nn.Conv2d(width[4], params['num_lms'], 1)
        self.offset_y = torch.nn.Conv2d(width[4], params['num_lms'], 1)
        self.neighbor_x = torch.nn.Conv2d(width[4], params['num_nb'] * params['num_lms'], 1)
        self.neighbor_y = torch.nn.Conv2d(width[4], params['num_nb'] * params['num_lms'], 1)

        torch.nn.init.normal_(self.score.weight, std=0.001)
        if self.score.bias is not None:
            torch.nn.init.constant_(self.score.bias, 0)

        torch.nn.init.normal_(self.offset_x.weight, std=0.001)
        if self.offset_x.bias is not None:
            torch.nn.init.constant_(self.offset_x.bias, 0)

        torch.nn.init.normal_(self.offset_y.weight, std=0.001)
        if self.offset_y.bias is not None:
            torch.nn.init.constant_(self.offset_y.bias, 0)

        torch.nn.init.normal_(self.neighbor_x.weight, std=0.001)
        if self.neighbor_x.bias is not None:
            torch.nn.init.constant_(self.neighbor_x.bias, 0)

        torch.nn.init.normal_(self.neighbor_y.weight, std=0.001)
        if self.neighbor_y.bias is not None:
            torch.nn.init.constant_(self.neighbor_y.bias, 0)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        score = self.score(x)
        offset_x = self.offset_x(x)
        offset_y = self.offset_y(x)
        neighbor_x = self.neighbor_x(x)
        neighbor_y = self.neighbor_y(x)
        if self.training:
            return score, offset_x, offset_y, neighbor_x, neighbor_y

        b, c, h, w = score.size()
        assert b == 1

        score = score.view(b * c, -1)
        max_idx = torch.argmax(score, 1).view(-1, 1)
        max_idx_neighbor = max_idx.repeat(1, self.params['num_nb']).view(-1, 1)

        offset_x = offset_x.view(b * c, -1)
        offset_y = offset_y.view(b * c, -1)
        offset_x_select = torch.gather(offset_x, 1, max_idx).squeeze(1)
        offset_y_select = torch.gather(offset_y, 1, max_idx).squeeze(1)

        neighbor_x = neighbor_x.view(b * self.params['num_nb'] * c, -1)
        neighbor_y = neighbor_y.view(b * self.params['num_nb'] * c, -1)
        neighbor_x_select = torch.gather(neighbor_x, 1, max_idx_neighbor)
        neighbor_y_select = torch.gather(neighbor_y, 1, max_idx_neighbor)
        neighbor_x_select = neighbor_x_select.squeeze(1).view(-1, self.params['num_nb'])
        neighbor_y_select = neighbor_y_select.squeeze(1).view(-1, self.params['num_nb'])

        offset_x = (max_idx % w).view(-1, 1).float() + offset_x_select.view(-1, 1)
        offset_y = (max_idx // w).view(-1, 1).float() + offset_y_select.view(-1, 1)
        offset_x /= 1.0 * self.args.input_size / self.params['stride']
        offset_y /= 1.0 * self.args.input_size / self.params['stride']

        neighbor_x = (max_idx % w).view(-1, 1).float() + neighbor_x_select
        neighbor_y = (max_idx // w).view(-1, 1).float() + neighbor_y_select
        neighbor_x = neighbor_x.view(-1, self.params['num_nb'])
        neighbor_y = neighbor_y.view(-1, self.params['num_nb'])
        neighbor_x /= 1.0 * self.args.input_size / self.params['stride']
        neighbor_y /= 1.0 * self.args.input_size / self.params['stride']

        # merge neighbor predictions
        neighbor_x = neighbor_x[self.reverse_index1, self.reverse_index2]
        neighbor_y = neighbor_y[self.reverse_index1, self.reverse_index2]
        neighbor_x = neighbor_x.view(self.params['num_lms'], self.max_len)
        neighbor_y = neighbor_y.view(self.params['num_lms'], self.max_len)

        offset_x = torch.mean(torch.cat((offset_x, neighbor_x), dim=1), dim=1).view(-1, 1)
        offset_y = torch.mean(torch.cat((offset_y, neighbor_y), dim=1), dim=1).view(-1, 1)

        output = torch.cat((offset_x, offset_y), dim=1)
        return torch.flatten(output).cpu().numpy()

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class CosineLR:
    def __init__(self, args, optimizer):
        self.min = 1E-5
        self.max = 1E-4

        self.optimizer = optimizer

        self.epochs = args.epochs
        self.values = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.warmup_epochs = 5
        self.warmup_values = [(v - self.max) / self.warmup_epochs for v in self.values]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.max

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [self.max + epoch * value for value in self.warmup_values]
        else:
            epoch = epoch - self.warmup_epochs
            if epoch < self.epochs:
                alpha = math.pi * (epoch - (self.epochs * (epoch // self.epochs))) / self.epochs
                values = [self.min + 0.5 * (lr - self.min) * (1 + math.cos(alpha)) for lr in self.values]
            else:
                values = [self.min for _ in self.values]

        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value
