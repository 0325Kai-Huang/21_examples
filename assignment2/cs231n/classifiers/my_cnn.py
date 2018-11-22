import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class DeepLayersConvNet(object):
    """
    A deep layer convolution network with following architecture:
    input - [conv-relu-BN-conv-relu-pool-BN]x2 -[affine-relu-BN] - affine - softmax
    every conv layers we maintain the same shape of the input
    conv1 filters=100 size = 3 stride = 1 pad = 1 # 32
    conv2 filters=50 size = 3 stride = 1 pad = 1 # 32
    conv3 filters=50 size = 3 stride = 1 pad = 1 # 16
    conv4 filters=50 size = 3 stride = 1 pad = 1 # 16
    out = N,50,8,8 reshape N,3200
    affine1_W = 3200,400 affine1_b = 400,
    affine2_W = 400,10  affine1_b = 10,
    softmax(out)
    """
    
    def __init__(self, input_dims=(3,32,32), num_filters=(100,50,50,50), filter_size = 3, hidden_dim = 400,num_classes=10, weight_scale = 1e-3, reg=1e-3, dtype=np.float32):
        
        """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['W1'] = np.random.randn(num_filters[0],input_dims[0],filter_size,filter_size) * weight_scale
        self.params['b1'] = np.zeros((num_filters[0],))
        self.params['W2'] = np.random.randn(num_filters[1],num_filters[0],filter_size,filter_size) * weight_scale
        self.params['b2'] = np.zeros((num_filters[1],))
        self.params['W3'] = np.random.randn(num_filters[2],num_filters[1],filter_size,filter_size) * weight_scale
        self.params['b3'] = np.zeros((num_filters[2],))
        self.params['W4'] = np.random.randn(num_filters[3],num_filters[2],filter_size,filter_size) * weight_scale
        self.params['b4'] = np.zeros((num_filters[3],))
        self.params['W5'] = np.random.randn(3200,400) * weight_scale
        self.params['b5'] = np.zeros((400,))
        self.params['W6'] = np.random.randn(400,10) * weight_scale
        self.params['b6'] = np.zeros((10,))
        self.params['gamma1'] = np.random.randn(num_filters[0])
        self.params['beta1'] = np.random.randn(num_filters[0])
        self.params['gamma2'] = np.random.randn(num_filters[1])
        self.params['beta2'] = np.random.randn(num_filters[1])
        self.params['gamma3'] = np.random.randn(num_filters[2])
        self.params['beta3'] = np.random.randn(num_filters[2])
        self.params['gamma4'] = np.random.randn(num_filters[3])
        self.params['beta4'] = np.random.randn(num_filters[2])
        self.params['gamma4'] = np.random.randn(num_filters[3])
        self.params['gamma5'] = np.random.randn(400)
        self.params['beta5'] = np.random.randn(400)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        
    def loss(self,X,y=None):
        
        
        loss,grads = 0.0,{}
        W1,b1 = self.params['W1'],self.params['b1']
        W2,b2 = self.params['W2'],self.params['b2']
        W3,b3 = self.params['W3'],self.params['b3']
        W4,b4 = self.params['W4'],self.params['b4']
        W5,b5 = self.params['W5'],self.params['b5']
        W6,b6 = self.params['W6'],self.params['b6']
        gamma1,beta1 = self.params['gamma1'],self.params['beta1']
        gamma2,beta2 = self.params['gamma2'],self.params['beta2']
        gamma3,beta3 = self.params['gamma3'],self.params['beta3']
        gamma4,beta4 = self.params['gamma4'],self.params['beta4']
        gamma5,beta5 = self.params['gamma5'],self.params['beta5']
        
        filter_size = W1.shape[2]
        
        conv_param = {"stride":1,"pad":1}
        
        pool_param = {"pool_height":2,"pool_width":2,"stride":2}
        
        bn_param1 = {"mode":"train"}
        bn_param2 = {"mode":"train"}
        bn_param3 = {"mode":"train"}
        bn_param4 = {"mode":"train"}
        bn_param5 = {"mode":"train"}
        
        scores = None
        
        #input - [conv-relu-BN-conv-relu-pool-BN]x2 -[affine-BN-relu] - affine - softmax
        
        
        conv1_out,conv1_cache = conv_relu_forward(X,W1,b1,conv_param)
        
        bn1_out,bn1_cache = spatial_batchnorm_forward(conv1_out,gamma1,beta1,bn_param1)
        
        conv2_out,conv2_cache = conv_relu_pool_forward(bn1_out,W2,b2,conv_param,pool_param)
        
        bn2_out,bn2_cache = spatial_batchnorm_forward(conv2_out,gamma2,beta2,bn_param2)
        
        conv3_out,conv3_cache = conv_relu_forward(bn2_out,W3,b3,conv_param)
        
        bn3_out,bn3_cache = spatial_batchnorm_forward(conv3_out,gamma3,beta3,bn_param3)
        
        conv4_out,conv4_cache = conv_relu_pool_forward(bn3_out,W4,b4,conv_param,pool_param)
        
        bn4_out,bn4_cache = spatial_batchnorm_forward(conv4_out,gamma4,beta4,bn_param4)
        
        affine5_out,affine5_cache = affine_bn_relu_forward(bn4_out.reshape(bn4_out.shape[0],-1),W5,b5,gamma5,beta5,bn_param5)
        
        affine6_out,affine6_cache = affine_forward(affine5_out,W6,b6)
        
        scores = affine6_out - np.max(affine6_out)
        
        if y is None:
            return scores
        
        loss,dL = softmax_loss(scores,y)
        
        dout6,grads['W6'],grads['b6'] = affine_backward(dL,affine6_cache)
        
        dout5,grads['W5'],grads['b5'],grads['gamma5'],grads['beta5'] = affine_bn_relu_backward(dout6,affine5_cache)
        
        dout5 = dout5.reshape(bn4_out.shape)
        
        dout4,grads['gamma4'],grads['beta4'] = spatial_batchnorm_backward(dout5,bn4_cache)
        
        dout3,grads['W4'],grads['b4'] = conv_relu_pool_backward(dout4,conv4_cache)
        
        dout2,grads['gamma3'],grads['beta3'] = spatial_batchnorm_backward(dout3,bn3_cache)
        
        dout1,grads['W3'],grads['b3'] = conv_relu_backward(dout2,conv3_cache)
        
        dout_1,grads['gamma2'],grads['beta2'] = spatial_batchnorm_backward(dout1,bn2_cache)
        
        dout_2,grads['W2'],grads['b2'] = conv_relu_pool_backward(dout_1,conv2_cache)
        
        dout_3,grads['gamma1'],grads['beta1'] = spatial_batchnorm_backward(dout_2,bn1_cache)
        
        dout_4,grads['W1'],grads['b1'] = conv_relu_backward(dout_3,conv1_cache)
        
        return loss,grads
    
pass
            
    
    
    
    