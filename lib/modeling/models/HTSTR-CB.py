import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper
from utils import clones

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret

# 修改SFTB，以前输入的是[n,c,s,p],现在要修改成输入是[n,c,s,h,w],原来的parts_num是分成16块，表示身体部分分成16块，现在修改的parts_num是1，表示一个整体的特征图
class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=1):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
          Input:  x,   [n, c, s, h, w]
          Output: ret, [n, c, h, w]
        """

        n, c, s, h, w = x.size()
        x = x.permute(3, 4, 0, 1, 2).contiguous() # [h, w, n, c, s]
        x = x.view(-1, n, c, s)
        feature = x.split(1, 0)  # [[1, n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(h, w, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(h, w, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [h, w, n, c]
        ret = ret.permute(2, 3, 0, 1).contiguous()  # [n, c, h, w]
        return ret

# adaptive region-based motion extractor (ARME)
class ARME_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, split_param ,m, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 padding=(1, 1, 1),bias=False,**kwargs):
        super(ARME_Conv, self).__init__()
        # m是分割块数
        self.m = m

        self.split_param = split_param

        self.conv3d = nn.ModuleList([
            BasicConv3d(in_channels, out_channels, kernel_size, stride, padding,bias ,**kwargs)
            for i in range(self.m)])


    def forward(self, x):
        '''
            x: [n, c, s, h, w]
            split_param 卷积的通道分割
        '''
        feat = x.split(self.split_param, 3)
        feat = torch.cat([self.conv3d[i](_) for i, _ in enumerate(feat)], 3)
        feat = F.leaky_relu(feat)
        return feat

class MyModel(BaseModel):
    def __init__(self, *args, **kargs):
        super(MyModel, self).__init__(*args, **kargs)
        """
            GaitPart: Temporal Part-based Model for Gait Recognition
            Paper:    https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
            Github:   https://github.com/ChaoFan96/GaitPart
        """

    def build_network(self, model_cfg):
        # 修改加入FConv，就是FPFE,backbone_cfg1，2，3可以一样，也可以不一样
        self.set_block1 = self.get_backbone(model_cfg['backbone_cfg1'])
        self.set_block2 = self.get_backbone(model_cfg['backbone_cfg2'])
        self.set_block3 = self.get_backbone(model_cfg['backbone_cfg3'])
        # 修改加入FC层
        # head_cfg = model_cfg['SeparateFCs']
        # 修改STFB 分块大小，为1
        # head_cfg0 = model_cfg['SeparateFCs0']

        # '''
        # self.HPP0 = SetBlockWrapper(
        #     HorizontalPoolingPyramid(bin_num=model_cfg['bin_num0']))
        # '''
        # 修改加入MTB，就是SFTB，对每一帧和相邻的帧提取一个关键特征
        self.SFTB1 = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=32, parts_num=1))
        self.SFTB2 = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=128, parts_num=1))
        self.SFTB3 = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=128, parts_num=1))
        # gl_block就是上面的FPFE
        self.gl_block2 = copy.deepcopy(self.set_block2)
        # self.gl_block2 = self.get_backbone(model_cfg['backbone_cfg2_1'])
        self.gl_block3 = copy.deepcopy(self.set_block3)

        # 对set_block进行分装
        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.HPP = HorizontalPoolingPyramid(bin_num=[32])

        # 加入ARME，就是SRME，总共有3个块
        in_c = model_cfg['channels']
        # class_num = model_cfg['class_num']
        self.arme1 = nn.Sequential(
            BasicConv3d(32, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.arme2 = nn.Sequential(
            ARME_Conv(in_c[2], in_c[0], split_param=[16, 16], m=2, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ARME_Conv(in_c[0], in_c[2], split_param=[16, 16], m=2, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.arme3 = nn.Sequential(
            ARME_Conv(in_c[2], in_c[1], split_param=[8, 8, 8, 8], m=4, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ARME_Conv(in_c[1], in_c[2], split_param=[8, 8, 8, 8], m=4, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )

        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])



    def forward(self, inputs):
        # ipts, labs, _, _, seqL = inputs
        #
        # sils = ipts[0]  # [n, s, h, w]
        # if len(sils.size()) == 4:
        #     sils = sils.unsqueeze(1)  #  torch.Size([128, 1, 30, 64, 44])
        #
        # del ipts
        # print(sils.shape)
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)
        # 第一组
        # outs就是经过FPFE得到的红色块
        outs = self.set_block1(sils)  # [n, c, s, h, w]
        # print('第一组out')
        # print(outs.shape)  # torch.Size([128, 128, 30, 16, 11])

        # 原始想法，outs经过HPP变成[n, c, s, p]
        # outs = self.HPP0(outs)

        # 第一个gl就是经过SFTB，得到蓝色的块
        gl = self.SFTB1(outs, seqL) # [n, c, h, w]
        # print('第一个gl')
        # print(gl.shape) # torch.Size([128, 128, 16, 11])
        # 第二个gl就是经过SREM，得到黄色的块
        gl = self.arme1(gl.unsqueeze(2)).squeeze(2)  #  [n, c, h, w]
        # print('arme1后的gl')
        # print(gl.shape)
        # 第三个gl就是经过FPFE，得到灰色的块
        gl = self.gl_block2(gl)
        # print('第三个gl')
        # print(gl.shape)
        # 第一组feature，HPM方法,有问题
        feature_gl1 = self.HPP(gl)  # [n, c, p]
        # print('feature_gl1')
        # print(feature_gl1.shape)
        # 第二组
        outs = self.set_block2(outs) #有问题
        # print('第二组outs')
        # print(outs.shape)
        gl = gl + self.SFTB2(outs, seqL)
        # print('第二组gl')
        # print(gl.shape)
        gl = self.arme2(gl.unsqueeze(2)).squeeze(2)
        # print('arm2后的gl')
        # print(gl.shape)
        # 经过FPFE，得到橙色的块
        gl = self.gl_block3(gl)
        # print('第二组第2个gl')
        # print(gl.shape)
        # 第二组feature,HPM方法
        feature_gl2 = self.HPP(gl)  # [n, c, p]
        # print('feature_gl2')
        # print(feature_gl2.shape)

        # 第三组
        outs = self.set_block3(outs)
        # print('第三组outs')
        # print(outs.shape)
        outs = self.SFTB3(outs, seqL)
        # print('第三组outs后SFTB3')
        # print(outs.shape)
        gl = gl + outs
        # print('第三组第一个gl')
        # print(gl.shape)
        gl = self.arme3(gl.unsqueeze(2)).squeeze(2)
        # print('arm3后的gl')
        # print(gl.shape)
        # 经过FPFE，得到绿色的块，这个gl_block3待定
        # gl = self.gl_block3(gl)
        # 经过FPFE，得到红色的块，这个gl_block3待定
        # outs = self.gl_block3(outs)


        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        # print('feature1')
        # print(feature1.shape)
        feature2 = self.HPP(gl)  # [n, c, p]
        # print('feature2')
        # print(feature2.shape)
        # 全体feature
        feature = torch.cat([feature_gl1, feature_gl2, feature1, feature2], -1)  # [n, c, p]
        # print('最后feature')
        # print(feature.shape)
        embs = self.Head(feature)

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval
