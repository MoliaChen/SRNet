# -*- coding: utf-8 -*-
# Author: Molia Chen
import torch
import torch.nn as nn
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, cr):
        super().__init__()
        self.input_ = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=(7, 7), stride=1, padding=3, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(2, 2, kernel_size=(7, 7), stride=1, padding=3, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1)
        )
        self.output = nn.Sequential(
            nn.Linear(896, int(2048/cr))
        )

    def forward(self, data):
        out = self.input_(data)
        out = out.flatten(start_dim=1)
        out = self.output(out)
        return out

class BRMBlock(nn.Module):
    def __init__(self, cr, padding_2, last_block=False):
        super().__init__()
        self.last_block = last_block
        if padding_2:
            self.up_sample_block = nn.Sequential(OrderedDict([
                ('conv_transpose1', nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=2, output_padding=1))
            ]))
        elif cr==4:
            self.up_sample_block = nn.Sequential(OrderedDict([
                ('conv_transpose1', nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1))
            ]))
        else:
            self.up_sample_block = nn.Sequential(OrderedDict([
                ('conv_transpose1', nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1))
            ]))

        self.SR_flow = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('prelu1', nn.PReLU()),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('prelu2', nn.PReLU()),
            ('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        if not last_block:
            if padding_2:
                self.down_sample = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2, bias=False))
                ]))
            elif cr==4:
                self.down_sample = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False))
                ]))
            else:
                self.down_sample = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False))
                ]))
            self.back_projection = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('prelu1', nn.PReLU()),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('prelu2', nn.PReLU()),
                ('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))

    def forward(self, data):
        up_sample = self.up_sample_block(data)
        sr_flow_output = self.SR_flow(up_sample)
        if not self.last_block:
            down_sample = self.down_sample(sr_flow_output)
            subtraction = data - down_sample
            back_projection_output = self.back_projection(subtraction)
            back_projection_output += subtraction
            return sr_flow_output, back_projection_output
        return sr_flow_output


class EBRM(nn.Module):
    def __init__(self, cr, block_num, padding_2):
        super().__init__()
        self.block_num = block_num
        self.ebrm = nn.ModuleList([BRMBlock(cr, padding_2) for _ in range(block_num-1)])
        self.ebrm.append(BRMBlock(cr, padding_2, last_block=True))
        self.fusion_function = nn.ModuleList([nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
                                              for _ in range(block_num-1)])

    def forward(self, data):
        out = data
        sr_flow = []
        for m in self.ebrm[:-1]:
            sr_flow_out, out = m(out)
            sr_flow.append(sr_flow_out)
        sr_flow.append(self.ebrm[-1](out))
        sr_flow = sr_flow[::-1]
        fusion_out = sr_flow[0]
        sr_flow_out_hat = []
        sr_flow_out_hat.append(fusion_out)
        for d, m in zip(sr_flow[1:], self.fusion_function):
            fusion_out = m(d + fusion_out)
            sr_flow_out_hat.append(fusion_out)
        output = torch.cat(sr_flow_out_hat[::-1], dim=1)
        return output

class Decoder(nn.Module):
    def __init__(self, cr, low_resolution, padding_2=False, EBRM_block=4):
        super().__init__()
        self.low_resolution = low_resolution
        self.linear = nn.Linear(int(2048/cr), int(low_resolution[0]*low_resolution[1]*low_resolution[2]))
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(low_resolution[0], 64, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.ebrm = EBRM(cr=cr, block_num=EBRM_block, padding_2=padding_2)
        self.concatmodule = nn.Conv2d(64*EBRM_block, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.middle_path = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.out = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, data):
        data = self.linear(data)
        lr_input = data.reshape(-1, *self.low_resolution)
        feature = self.features(lr_input)
        sr_flow_output = self.ebrm(feature)
        concat_out = self.concatmodule(sr_flow_output)
        middle_out = self.middle_path(concat_out)
        output = self.out(middle_out)
        return output

class AutoEncoder(nn.Module):
    def __init__(self, cr, EBRM_block):
        super().__init__()
        padding_2 = False
        if cr >= 32:
            low_resolution = (1, 15, 6)
        elif cr == 16:
            low_resolution = (2, 15, 6)
        elif cr==4:
            low_resolution = (2, 32, 14)
        else:
            low_resolution = (2, 17, 8)
            padding_2 = True

        self.encoder = Encoder(cr=cr)
        self.decoder = Decoder(cr=cr, low_resolution=low_resolution, padding_2=padding_2, EBRM_block=EBRM_block)

    def forward(self, data):
        out = self.encoder(data)
        out = self.decoder(out)
        return out

def load_network(scenario, cr):
    assert scenario in ['indoor', 'outdoor'], "The scenario is not exist!"
    assert cr in [4, 8, 16, 32, 64], "The compression ratio is not exist!"
    model = AutoEncoder(cr, 6)
    return model