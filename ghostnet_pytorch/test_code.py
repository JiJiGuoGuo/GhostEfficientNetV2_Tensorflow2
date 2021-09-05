import tensorflow as tf
import math
def _make_divisible(v, divisor:int=8, min_value=None):
    """
    确保所有的层的通道都能够被8整除，生成的数字，能被divisor整除
    使用if，来保证new_v相对于原先的v的变化不超过+-10%
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    # 确保这一整数的下降幅度不超过10%。
    if new_v < (0.9 * v):
        new_v += divisor
    return new_v

class MBConv(tf.keras.layers.Layer):
    '''
    MBConv使用了SE,使用一个变量控制是否使用Fused_MBConv
    '''
    def __init__(self,input_ch,out_ch,stride,expand_ratio,shortcut,survival=None,is_Fused=True,se_ratio=4):
        '''

        Args:
            inputs: 输入的张量
            out_ch:输出张量的通道数，tensorflow默认是shape[-1]
            stride:
            expand_ratio:扩展比例，第一个卷积后的channels会将input的channels扩展到几倍，NAS主要有{1,4,6}
            shortcut:是否使用直连/残差
            survival:随机网络深度中的残差网络生存概率
            use_se:是否使用SE模块，在EfficientV1和V2中都使用了
            is_Fused:是否使用Fused_MBConv
        '''
        super(MBConv, self).__init__()
        self.in_ch = input_ch#输入张量的通道数
        self.batch_norm_decay = 0.9
        self.is_Fused = is_Fused
        self.se_ratio = se_ratio is not None
        self.shortcut = shortcut
        self.survival = survival
        self.expand_ratio = expand_ratio

        #分别对Fused情形进行处理
        if is_Fused:#Fused_MBConv
            self.upsample_fused = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.in_ch * expand_ratio, 3, strides=stride, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=self.batch_norm_decay),
                tf.keras.layers.Activation('swish'),
            ])
        elif not is_Fused:#MBConv
            self.upsample_no_fused = tf.keras.Sequential([
                #1x1卷积
                tf.keras.layers.Conv2D(self.in_ch * expand_ratio, 1, padding='same', use_bias=False),# 1x1卷积的strides只能是1
                tf.keras.layers.BatchNormalization(momentum=self.batch_norm_decay),
                tf.keras.layers.Activation('swish'),
                #Depthwise Convolution
                tf.keras.layers.DepthwiseConv2D(3, padding='same', strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=self.batch_norm_decay),
                tf.keras.layers.Activation('swish')
            ])

        if self.se_ratio is not None:
            self.se = SE(self.in_ch * expand_ratio,se_ratio)

        #如果第一个卷积不进行升维且是Fused
        if is_Fused and expand_ratio == 1:
            self.downsample_fused = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_ch,3,strides=stride,padding="same",use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=self.batch_norm_decay),
                tf.keras.layers.Activation('swish')
            ])
        else:
            self.downsample_no_fused = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_ch, 1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=self.batch_norm_decay),
            ])

        if shortcut:
            if survival is not None and survival<1:
                import tensorflow_addons as tfa
                self.stochastic_depth = tfa.layers.StochasticDepth(survival_probability=survival)

    def call(self,inputs):
        if self.is_Fused:
            x0 = self.upsample_fused(inputs)
        elif not self.is_Fused:
            x0 = self.upsample_no_fused(inputs)

        if self.se_ratio is not None:
            x1 = self.se(x0)

        if self.is_Fused and self.expand_ratio == 1:
            x2 =self.downsample_fused(x1)
        else:
            x2 = self.downsample_no_fused(x1)

        print("shape inputs is {},shape x2 is {}".format(inputs.shape,x2.shape))

        if self.shortcut:#如果使用直连/残差结构
            if self.survival is not None and self.survival<1:#生存概率(随机深度残差网络论文中的术语，表示残差支路被激活的概率)
                return self.stochastic_depth([inputs,x2])
            else:
                return x2 + inputs
        else:
            return x2

class SE(tf.keras.layers.Layer):
    def __init__(self,inputs_channels:int,se_ratio:int = 4,name:str=""):
        '''
        这个函数是使用Conv1x1实现的SE模块，并使用reduc_mean实现GlobalAveragePooling
        Args:
            inputs_channels: 输入张量的channels
            se_ratio: 第一个FC会将输入SE的张量channels压缩成的倍率
            name:
            return:一个张量，shape同input
        '''
        super(SE,self).__init__()
        self.se_ratio = se_ratio
        self.filters = inputs_channels
        self.reduction = _make_divisible(inputs_channels / se_ratio,8)
        #第一个FC将输入SE的channel压缩成1/4
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
        self.conv1 = tf.keras.layers.Conv2D(self.reduction,1,1,use_bias=True,name=name+'1_conv')
        self.act1 = tf.keras.layers.Activation('swish')
        self.conv2 = tf.keras.layers.Conv2D(self.filters,1,1,use_bias=True,name=name+'2_conv')
        self.act2 = tf.keras.layers.Activation('sigmoid')
        self.multiply = tf.keras.layers.Multiply()
    def call(self,inputs):
        #由于tf2.6才增加了keep_dim，所以在tf2.3需要手动expand_dims
        x = self.global_pool(inputs)
        x = tf.expand_dims(tf.expand_dims(x,1),1)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        out = self.multiply([x,inputs])
        return out


x = tf.random.uniform([3,5,5,16])
model = MBConv(input_ch=24,out_ch=24,stride=3,expand_ratio=4,shortcut=1,survival=0.5,is_Fused=True,se_ratio=4)
m = model(x)
print(m)