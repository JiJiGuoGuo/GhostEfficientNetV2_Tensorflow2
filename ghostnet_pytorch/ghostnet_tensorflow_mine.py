# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import tensorflow as tf
import math

__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    它确保所有的层的通道都能够被8整除
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    # 确保这一整数的下降幅度不超过10%。
    if new_v < (0.9 * v):
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SE(tf.keras.layers.Layer):
    def __init__(self,inputs,se_ratio:int = 4,name:str=""):
        '''
        这个函数是使用Conv1x1实现的SE模块，并使用reduc_mean实现GlobalAveragePooling
        channel_pos：通道的维数位于第二维度还是最后一维度
        '''
        super(SE,self).__init__()
        self.inputs = inputs
        self.se_ratio = se_ratio
        self.name = name
        self.filters = self.inputs.shape[-1]
        self.reduction = self.filters // self.se_ratio
        #第一个FC将输入SE的channel压缩成1/4
        self.conv1 = tf.keras.layers.Conv2D(self.reduction,1,1,use_bias=True,name=self.name+'1_conv')
        self.act1 = tf.keras.layers.Activation('swish')
        self.conv2 = tf.keras.layers.Conv2D(self.filters,1,1,use_bias=True,name=self.name+'2_conv')
        self.act2 = tf.keras.layers.Activation('sigmoid')
        self.multiply = tf.keras.layers.Multiply()
    def call(self):
        x = tf.reduce_mean(self.inputs,(1,2),keepdims=True)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        out = self.multiply([x,self.inputs])
        return out

class SqueezeExcite(tf.keras.layers):
    '''
    SE模块，Squeeze：一个AvePooling，GhostModue的SE全部激活函数换成了relu
    Excitation：一个FC+swish，加一个FC+sigmoid
    参数:
    in_chs：输入张量的channel数
    se_ratio:SE压缩率
    reduced_base_chs:
    '''
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, divisor=4):
        super(SqueezeExcite, self).__init__()
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv1 = tf.keras.layers.Conv2D(reduced_chs,1,1,use_bias=True)
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(in_chs,1,1,use_bias=True)
        self.act2 = tf.keras.layers.Activation('relu')

    def call(self, x):
        xx = tf.reduce_mean(x,(1,2),keepdims=True)
        xx = self.conv1(xx)
        xx = self.act1(xx)
        xx = self.conv2(xx)
        xx = self.act2(xx)
        out = tf.keras.layers.Multiply()([x,xx])
        return out


class ConvBnAct(tf.keras.layers.Layer):
    '''
    这个函数用来封装con，Bn，Act(relu)
    '''
    def __init__(self,out_chs, kernel_size,
                 stride=1):
        super(ConvBnAct, self).__init__()
        self.conv = tf.keras.layers.Conv2D(out_chs,kernel_size,stride,padding='same',use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(tf.keras.layers):
    def __init__(self, oup, kernel_size=1, ratio=2, dw_size=3, stride=1,use_relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 这里可以看到实现了两次卷积，点卷积
        self.primary_conv = tf.keras.Sequential([
            #点卷积的卷积核的组数=上一层的channel数，大小为1x1xM，其中M=input.shape(-1)
            tf.keras.layers.Conv2D(init_channels,kernel_size,stride,padding='same',use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation='relu') if use_relu else tf.keras.Sequential(),
        ])

        self.cheap_operation = tf.keras.Sequential([
            #group用于对channel进行分组，默认是一个channel为一组,分组卷积
            tf.keras.layers.Conv2D(new_channels,dw_size,1,'same',use_bias=False,groups=init_channels),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation='relu') if use_relu else tf.keras.Sequential(),
        ])

    def call(self,x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = tf.concat([x1,x2],axis=-1)
        #第0,1,2维全选，最后一维度从0开始读取到self.oup，步长为1，左闭右开
        return out[...,:self.oup]


class GhostBottleneck(tf.keras.layers):
    '''
    G-Bneck专为小型网络而设计类似于ResidualBlock，包含两个GhostModule，第一个用作扩展层，第二个减少通道数
    stride=1和2的情形不同
    '''
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,stride=1, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.#是否使用SEModule
        self.stride = stride

        #第一个GhostModule用于扩展层
        self.ghost_module1 = GhostModule(in_chs, mid_chs)

        #深度卷积Depthwise Convolution
        if self.stride > 1:#大于2才使用DepthwiseConv
            self.dw_conv = tf.keras.layers.DepthwiseConv2D(dw_kernel_size,strides=2,groups=mid_chs,padding='same',use_bias=False)
            self.dw_bn = tf.keras.layers.BatchNormalization()

        if has_se:#如果使用SE模块
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else: #不使用SE模块
            self.se = None

        #第二个GhostModule模块不再使用ReLU，减少通道数
        self.ghost_module2 = GhostModule(mid_chs, out_chs, use_relu=False)

        #是否使用shortcut
        #先假设规定：要使用shortcut，GBneck的输入特征图channels=输出的特征图channels
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = tf.keras.Sequential()
        else:#使用shotcut，stride>2
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(dw_kernel_size,stride,padding='same',use_bias=False,groups=in_chs),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(out_chs,1,1,padding='same',use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, x):

        #第一个GhostModule
        xx = self.ghost_module1(x)

        # Depth-wise convolution
        if self.stride > 1:
            xx = self.dw_conv(xx)
            xx = self.dw_bn(xx)

        #SE模块
        if self.se is not None:
            xx = self.se(xx)
        #第二个GhostModule
        xx = self.ghost_module2(xx)
        #残差结构
        xx =xx + self.shortcut(x)
        return xx


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        #逆残差结构设置
        self.cfgs = cfgs
        self.dropout = dropout

        #构建第一层
        output_channel = _make_divisible(16 * width, 4)
        self.conv_1 = tf.keras.layers.Conv2D(output_channel,3,2,'same',use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        input_channel = output_channel

        # 构建逆残差块
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))#全局平均池化7x7全局平均池化
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)#1x1卷积
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)#FC/Dense全连接层，输出分类数

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    配置ghostnet
    """
    cfgs = [
        # kernel size, t, out channels, use SE, GhostBottleNeck stride
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == '__main__':
    model = ghostnet()
    model.eval()
    print(model)
    input = torch.randn(32, 3, 320, 256)
    y = model(input)
    print(y.size())