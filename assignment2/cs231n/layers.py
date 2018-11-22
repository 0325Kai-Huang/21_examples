import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
#   pass
#   print(x.shape)
#   print(w.shape)
#   N, D = x.shape[0], int(x.size / x.shape[0])
  out = np.dot(x, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
#   pass
  dx = dout.dot(w.T)
  dw = x.T.dot(dout)
  db = np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
#   pass
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
#   pass
  dout[x < 0] = 0.0
  dx = dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  # bn_param 是一个字典型，get()方法是中第一个参数是键名，第二个参数是默认值，如果指定的键其值不存在，
  # 就用第二个参数的值做为默认值
#   print(N,D)
  running_mean = bn_param.get('running_mean', np.zeros(x.shape[1], dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(x.shape[1], dtype=x.dtype))
#   print(running_mean.shape)

  out, cache = None, None
#   out, cache = np.zeros_like(x), {}
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
#     pass
    sample_mean = np.mean(x,axis=0,keepdims=True)
    sample_var = np.var(x,axis=0,keepdims=True)
#     print(sample_mean.shape)
    out_ = (x - sample_mean) / np.sqrt(sample_var + eps)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    out = gamma * out_ + beta
    
    cache = (out_, x, sample_var, eps, gamma)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
#     pass
    out_ = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * out_ + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  """
  mean = np.mean(x,axis=0)
  variance = np.var(x,axis=0)
  x^ = (x-mean) / np.sqrt(variance + eps)
  out = gamma * x^ + beta
  dout = dout  Upstream derivatives, of shape (N,D)
  dout / dgamma = x^  ==> dgamma = dout / x^
  dout / dbeta = 1 ==> dbeta = dout
  dout / dx^ = gamma
  dx^ / dx = 1 / np.sqrt(variance + eps)
  dout / dx = gamma / np.sqrt(variace+eps) ==> dx = dout * np.sqrt(variance + eps) / gamma
  """
#   pass
  
  out_,x,sample_var,eps,gamma = cache
  N = x.shape[0]
  dbeta = np.sum(dout,axis=0)
  dgamma = np.sum(dout * out_,axis=0)
  dout_ = np.matmul(np.ones((N,1)), gamma.reshape((1,-1))) * dout
  dx = N * dout_ - np.sum(dout_,axis=0) - out_*np.sum(dout_*out_,axis=0)
  dx *= 1/(N*np.sqrt(sample_var + eps))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
#     pass
    mask = np.random.rand(x.shape[0],x.shape[1])
    mask = (mask < p)
    out = x*mask
    out /= p
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
#     pass
    # if the mode is test, just return the input 
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
#     pass
    p = dropout_param['p']
    dx = dout * mask
    dx /= p
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
#   pass
  
  pad,stride = conv_param['pad'],conv_param['stride']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_ = int(1 + (H + 2 * pad - HH) / stride)
  W_ = int(1 + (W + 2 * pad - WW) / stride)
  out = np.zeros((N,F,H_,W_))
  x_padding = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
  for n in range(N):
    for f in range(F):
      for i in range(H_):
        for j in range(W_):
#             print("n,f,i,j",n,f,i,j)
          # line 426~429 下面的四行是重要部分，最主要的计算以及如何计算就是在这
          result = 0
          for c in range(C):
            result += np.sum(x_padding[n,c,i*stride:i*stride+HH,j*stride:j*stride+WW]*w[f,c,:,:])
          out[n,f,i,j] = result + b.reshape(b.shape[0],1,1)[f,:,:]
#             out[n,f,i,j] = np.sum(x_padding[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW]*w[f,:,:,:]+ b.reshape(b.shape[0],1,1)) / w.reshape(w.shape[0],-1).shape[1]
#   print(out.shape)
#   print(out)
#   print(w.reshape(w.shape[0],-1).shape[1])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
#   pass

  x,w,b,conv_param = cache
  dx = np.zeros_like(x) #(N,C,H,W)
  dw = np.zeros_like(w) #(F,C,HH,WW)
  db = np.zeros_like(b) #(F,)  dout(N,F,Hnew,Wnew)
  stride,pad = conv_param['stride'],conv_param['pad']
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  N,F,Hnew,Wnew = dout.shape
    
  db = np.sum(dout,axis = (0,2,3))
  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  dx_pad = np.zeros_like(x_pad)

  for i in range(Hnew):
    for j in range(Wnew):
      x_pad_mask = x_pad[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
      
      for n in range(N):
        dx_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += np.sum(w[:,:,:,:] * (dout[n,:,i,j])[:,None,None,None],axis=0)
      for k in range(F):
        dw[k,:,:,:] += np.sum(x_pad_mask * (dout[:,k,i,j][:,None,None,None]),axis=0)
        
  dx = dx_pad[:,:,pad:-pad,pad:-pad]





#   for i in range(Hnew):
#     for j in range(Wnew):
#       x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
#         for k in range(F): #compute dw
#           dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
#         for n in range(N): #compute dx_pad
#           dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] * (dout[n, :, i, j])[:,None ,None, None]), axis=0)
#   dx = dx_pad[:,:,pad:-pad,pad:-pad]
# 下面的BP是错的，我还自以为是的BB了好久，都是错的。真特么尴尬


#   w_rotate = np.rot90(w,2,(2,3))
# #这是之前出错的地方，多了一个循环，由dx与dw的求解过程可以发现，循环的值都是目标值的维度，比如dx的维度分别是N，C，H，W，
# #因此，循环N，H，W，C，感觉顺序应该都可以。参与计算的向量其中有些维度不在目标维度中都用：符号来表示所以。
#   for n in range(N):
# #     for f in range(F): 
#     for h in range(H):
#       for m in range(W):
#         for c in range(C):
# #           print(m*stride,m*stride+WW)
#           dx[n,c,h,m] = np.sum(w_rotate[:,c,:,:] * dout_padding[n,:,h*stride:h*stride+HH,m*stride:m*stride+WW])
  
#   padding = int(((HH - 1)*stride + Hnew - H) / 2)
#   x_padding = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)),'constant')
#   for f in range(F):
#     for c in range(C):
#       for m in range(HH):
#         for n in range(WW):
#           dw[f,c,m,n] = np.sum(x_padding[:,c,m*stride:m*stride+Hnew,n*stride:n*stride+Wnew]*dout[:,f,:,:])

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
#   pass 
  """
  max pooling layer 其实就是一个卷积核，卷积核在x上进行滑动，每次取覆盖区域中的最大值
  这就是max pooling,还有mean pooling 这是取覆盖区域中的均值。
  还有个L2 pooling 是取所以值平方的和然后取平方根。
  """
  pool_height,pool_width,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
  N,C,H,W = x.shape
  Hnew = int((H - pool_height) / stride + 1)
  Wnew = int((W - pool_width) / stride + 1)
  out = np.zeros((N,C,Hnew,Wnew))
  for n in range(N):
    for c in range(C):
      for h in range(Hnew):
        for w in range(Wnew):
          out[n,c,h,w] = np.max(x[n,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
#   pass
  """
  既然只是取最大值，在原理上基本上和Relu激活函数的BP一样，因此我们可以零初始化dx与x同shape
  然后计算还原FP过程。但是此时我们知道Hnew和Wnew的值，就是已经知道dout的shape，这样循环
  更加方便，唯一的变动在，不是用np.max()函数，而是用np.argmax()方法取到最大值的索引，
  该索引是将方法中的向量reshape(-1)得到的索引。我们将该索引赋给dx中的位置置为1并且与dout
  相乘。先乘是因为dx的shape和dout的shape不一样。来尝试一下吧
  """
  
  x,pool_param = cache
  dx = np.zeros_like(x)
  pool_height,pool_width,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
  N,C,Hnew,Wnew = dout.shape
  for n in range(N):
    for c in range(C):
      for h in range(Hnew):
        for w in range(Wnew):
          # 下面结果不会变，因此reshape方法应该是返回另一个向量，不会对原向量有任何影响。所以dx不会变。
          # 因此要换个方法赋值，获得索引后，除pool_height得到行数，取余获得列数来赋值
#           dx[n,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width].reshape(-1)[np.argmax(x[n,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width])] = dout[n,c,h,w]
          index = np.argmax(x[n,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width])
          index_h = int(index / pool_height)
          index_w = index % pool_height
          dx[n,c,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width][index_h,index_w] = dout[n,c,h,w]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
#   pass
  # 自己实现，没有用之前的方法，因为维度的原因，我也不知道怎么使用上面的方法。但是下面复制了
  # 大佬的实现方法，就5行
#   N,C,H,W = x.shape
#   mode = bn_param['mode']
#   eps = bn_param.get('eps', 1e-5)
#   momentum = bn_param.get('momentum', 0.9)
#   running_mean = bn_param.get('running_mean',np.zeros(C, dtype = x.dtype))
#   running_var = bn_param.get('running_var',np.zeros(C, dtype = x.dtype))
#   if mode == "train":
#       sample_mean = np.mean(x,axis=(0,2,3)) # (C,)
#       sample_var = np.var(x,axis=(0,2,3)) # (C,)
#       out_ = (x - sample_mean[None,:,None,None]) / np.sqrt(sample_var + eps)[None,:,None,None]
#       running_mean = running_mean*momentum + (1-momentum)*sample_mean
#       running_var = running_var*momentum + (1 - momentum)*sample_var
#       out = gamma[None,:,None,None]*out_ + beta[None,:,None,None]
#       cache = (out_, x, sample_var, eps, gamma)
#   elif mode == "test":
#       out_ = (x - running_mean[None,:,None,None]) / np.sqrt(running_var + eps)[None,:,None,None]
#       out = gamma[None,:,None,None]*out_ + beta[None,:,None,None]
#   bn_param['running_mean'] = running_mean
#   bn_param['running_var'] = running_var
    
  # 牛逼的做法
  N, C, H, W = x.shape
  x = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
#   pass
  # 刚刚看的大佬的方法，将x四维的向量转换成二维的，然后在使用之前实现的二维的BN方法来求BP
  # 我也来试试看
  N,C,H,W = dout.shape
  dout = dout.transpose(0,2,3,1).reshape(N * H * W, C)
  dx, dgamma, dbeta = batchnorm_backward(dout,cache)
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
