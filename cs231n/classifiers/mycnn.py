from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *



# def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
#     a, conv_cache = conv_forward_fast(x, w, b, conv_param)
#     an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
#     out, relu_cache = relu_forward(an)
#     cache = (conv_cache, bn_cache, relu_cache)
#     return out, cache


# def conv_bn_relu_backward(dout, cache):
#     conv_cache, bn_cache, relu_cache = cache
#     dan = relu_backward(dout, relu_cache)
#     da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
#     dx, dw, db = conv_backward_fast(da, conv_cache)
#     return dx, dw, db, dgamma, dbeta


class MyCNN(object):
    """
    A three-layer convolutional network with the following architecture:

    ConvBNReLU(3->64) - dropout(0.3)

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        self.paramsNoUp={}

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        dim1 = 16
        dim2 = 64
        dim3 = 256
        self.params["W1"] = np.random.normal(0, 2./3 , (dim1,3,3,3))
        self.params["b1"] = np.zeros((dim1))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma1"] = np.random.normal(0, weight_scale , (dim1,))
        self.params["beta1"] = np.zeros((dim1))
        self.paramsNoUp["running_mean1"] = np.zeros((dim1))
        self.paramsNoUp["running_var1"] = np.zeros((dim1))


        #maxPool
        self.params["W2"] = np.random.normal(0, 2./dim1 , (dim2,dim1,3,3))
        self.params["b2"] = np.zeros((dim2))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma2"] = np.random.normal(0, weight_scale , (dim2,))
        self.params["beta2"] = np.zeros((dim2))
        self.paramsNoUp["running_mean2"] = np.zeros((dim2))
        self.paramsNoUp["running_var2"] = np.zeros((dim2))

        
        #maxPool
        self.params["W3"] = np.random.normal(0, 2./dim2 , (dim3,dim2,3,3))
        self.params["b3"] = np.zeros((dim3))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma3"] = np.random.normal(0, weight_scale , (dim3,))
        self.params["beta3"] = np.zeros((dim3))
        self.paramsNoUp["running_mean3"] = np.zeros((dim3))
        self.paramsNoUp["running_var3"] = np.zeros((dim3))

        # self.params["W4"] = np.random.normal(0, 2./dim3/64 , (dim3*8*8,dim3))
        # self.params["b4"] = np.zeros(dim3)
        self.params["W5"] = np.random.normal(0, 2./dim3 , (dim3*8*8,num_classes))
        self.params["b5"] = np.zeros(num_classes)

        #self.params["conv_param"] = {"stride":1,"pad":0}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        input = X
        result={}
        #print(input.shape,self.params["W1"].shape ,self.params["b1"].shape )
      #   eps: Constant for numeric stability
      # - momentum: Constant for running mean / variance. momentum=0 means that
      #   old information is discarded completely at every time step, while
      #   momentum=1 means that new information is never incorporated. The
      #   default of momentum=0.9 should work well in most situations.
      # - running_mean: Array of shape (D,) giving running mean of features
      # - running_var Array of shape (D,) giving running variance of features
        input,result["cacheconv1"] = conv_bn_relu_forward(input,self.params["W1"] , self.params["b1" ], self.params["gamma1" ], self.params["beta1" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean1"],"running_var":self.paramsNoUp["running_var1"]})
        input,result["cachemaxPool1"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv2"] = conv_bn_relu_forward(input,self.params["W2"] , self.params["b2" ], self.params["gamma2" ], self.params["beta2" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean2"],"running_var":self.paramsNoUp["running_var2"]})
        input,result["cachemaxPool2"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv3"] = conv_bn_relu_forward(input,self.params["W3"] , self.params["b3" ], self.params["gamma3" ], self.params["beta3" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean3"],"running_var":self.paramsNoUp["running_var3"]})
        #input,result["cachemaxPool3"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        #print(input.shape,self.params["W2"].shape,self.params["b2" ].shape)
        # input,result["cacheaff1"] = affine_relu_forward(input,self.params["W4"] , self.params["b4" ])
        scores,result["cacheaff2"] = affine_forward(input,self.params["W5"] , self.params["b5" ])
        #scores,scores_cache = affine_forward(input , self.params["W"+str(self.num_layers - 1)] , self.params["b" + str(self.num_layers - 1)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss , dloss  = softmax_loss(scores,y)
        dup , grads["W5"],grads["b5"], = affine_backward(dloss,result["cacheaff2"])
        # dup , grads["W4"],grads["b4"] =  affine_relu_backward(dup,result["cacheaff1"])
        #dx, dw, db, dgamma, dbeta
        dup, grads["W3"],grads["b3"],grads["gamma3"],grads["beta3"] = conv_bn_relu_backward(dup,result["cacheconv3"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool2"])
        dup, grads["W2"],grads["b2"],grads["gamma2"],grads["beta2"] = conv_bn_relu_backward(dup,result["cacheconv2"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool1"])
        dup, grads["W1"],grads["b1"],grads["gamma1"],grads["beta1"] = conv_bn_relu_backward(dup,result["cacheconv1"])
        for para in self.params:
            #if para in grads:
            loss += self.reg * np.sum(self.params[para]*self.params[para])
            grads[para] += 2 * self.reg * self.params[para]

        # for i in range(self.num_layers):
        #     loss +=  self.reg * np.sum(self.params["W"+str(i)] * self.params["W"+str(i)])
        #     grads["W"+str(i)] -= 2 * self.reg * self.params["W"+str(i)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads




class MyCNN2(object):
    """
    A three-layer convolutional network with the following architecture:

    ConvBNReLU(3->64) - dropout(0.3)

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        self.paramsNoUp={}

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        dim1 = 16
        dim2 = 32
        dim3 = 32
        dim4 = 64
        dim5 = 256
        self.params["W1"] = np.random.normal(0, 2./3 , (dim1,3,3,3))
        self.params["b1"] = np.zeros((dim1))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma1"] = np.zeros((dim1))
        self.params["beta1"] = np.zeros((dim1))
        self.paramsNoUp["running_mean1"] = np.zeros((dim1))
        self.paramsNoUp["running_var1"] = np.zeros((dim1))


        #maxPool
        self.params["W2"] = np.random.normal(0, 2./dim1 , (dim2,dim1,3,3))
        self.params["b2"] = np.zeros((dim2))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma2"] = np.zeros((dim2))
        self.params["beta2"] = np.zeros((dim2))
        self.paramsNoUp["running_mean2"] = np.zeros((dim2))
        self.paramsNoUp["running_var2"] = np.zeros((dim2))

        
        #maxPool
        self.params["W3"] = np.random.normal(0, 2./dim2 , (dim3,dim2,3,3))
        self.params["b3"] = np.zeros((dim3))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma3"] = np.zeros((dim3))
        self.params["beta3"] = np.zeros((dim3))
        self.paramsNoUp["running_mean3"] = np.zeros((dim3))
        self.paramsNoUp["running_var3"] = np.zeros((dim3))

        #maxPool
        self.params["W4"] = np.random.normal(0, 2./dim3 , (dim4,dim3,3,3))
        self.params["b4"] = np.zeros((dim4))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma4"] = np.zeros((dim4))
        self.params["beta4"] = np.zeros((dim4))
        self.paramsNoUp["running_mean4"] = np.zeros((dim4))
        self.paramsNoUp["running_var4"] = np.zeros((dim4))


        self.params["W5"] = np.random.normal(0, 2./dim4 , (dim5,dim4,3,3))
        self.params["b5"] = np.zeros((dim5))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma5"] = np.zeros((dim5))
        self.params["beta5"] = np.zeros((dim5))
        self.paramsNoUp["running_mean5"] = np.zeros((dim5))
        self.paramsNoUp["running_var5"] = np.zeros((dim5))

        # self.params["W4"] = np.random.normal(0, 2./dim3/64 , (dim3*8*8,dim3))
        # self.params["b4"] = np.zeros(dim3)
        self.params["W6"] = np.random.normal(0, 2./dim5/16 , (dim5*4*4,num_classes))
        self.params["b6"] = np.zeros(num_classes)

        #self.params["conv_param"] = {"stride":1,"pad":0}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        
        input = X
        result={}
        input,result["cacheconv1"] = conv_bn_relu_forward(input,self.params["W1"] , self.params["b1" ], self.params["gamma1" ], self.params["beta1" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean1"],"running_var":self.paramsNoUp["running_var1"]})
        input,result["cachemaxPool1"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv2"] = conv_bn_relu_forward(input,self.params["W2"] , self.params["b2" ], self.params["gamma2" ], self.params["beta2" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean2"],"running_var":self.paramsNoUp["running_var2"]})
        input,result["cachemaxPool2"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv3"] = conv_bn_relu_forward(input,self.params["W3"] , self.params["b3" ], self.params["gamma3" ], self.params["beta3" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean3"],"running_var":self.paramsNoUp["running_var3"]})
        input,result["cachemaxPool3"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv4"] = conv_bn_relu_forward(input,self.params["W4"] , self.params["b4" ], self.params["gamma4" ], self.params["beta4" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean4"],"running_var":self.paramsNoUp["running_var4"]})
        input,result["cacheconv5"] = conv_bn_relu_forward(input,self.params["W5"] , self.params["b5" ], self.params["gamma5" ], self.params["beta5" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean5"],"running_var":self.paramsNoUp["running_var5"]})
        scores,result["cacheaff2"] = affine_forward(input,self.params["W6"] , self.params["b6" ])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss , dloss  = softmax_loss(scores,y)
        dup , grads["W6"],grads["b6"], = affine_backward(dloss,result["cacheaff2"])
        
        dup, grads["W3"],grads["b3"],grads["gamma3"],grads["beta3"] = conv_bn_relu_backward(dup,result["cacheconv3"])
        dup, grads["W3"],grads["b3"],grads["gamma3"],grads["beta3"] = conv_bn_relu_backward(dup,result["cacheconv3"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool2"])
        dup, grads["W3"],grads["b3"],grads["gamma3"],grads["beta3"] = conv_bn_relu_backward(dup,result["cacheconv3"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool2"])
        dup, grads["W2"],grads["b2"],grads["gamma2"],grads["beta2"] = conv_bn_relu_backward(dup,result["cacheconv2"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool1"])
        dup, grads["W1"],grads["b1"],grads["gamma1"],grads["beta1"] = conv_bn_relu_backward(dup,result["cacheconv1"])
        for para in self.params:
            #if para in grads:
            loss += self.reg * np.sum(self.params[para]*self.params[para])
            grads[para] += 2 * self.reg * self.params[para]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads






class MyCNN3(object):
    """
    A three-layer convolutional network with the following architecture:

    ConvBNReLU(3->64) - dropout(0.3)

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        self.paramsNoUp={}

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        dim1 = 32
        dim2 = 64
        dim3 = 128
        dim4 = 128
        self.params["W1"] = np.random.normal(0, 2./3 , (dim1,3,3,3))
        self.params["b1"] = np.zeros((dim1))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma1"] = np.zeros((dim1))
        self.params["beta1"] = np.zeros((dim1))
        self.paramsNoUp["running_mean1"] = np.zeros((dim1))
        self.paramsNoUp["running_var1"] = np.zeros((dim1))


        #maxPool
        self.params["W2"] = np.random.normal(0, 2./dim1 , (dim2,dim1,3,3))
        self.params["b2"] = np.zeros((dim2))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma2"] = np.zeros((dim2))
        self.params["beta2"] = np.zeros((dim2))
        self.paramsNoUp["running_mean2"] = np.zeros((dim2))
        self.paramsNoUp["running_var2"] = np.zeros((dim2))

        
        #maxPool
        self.params["W3"] = np.random.normal(0, 2./dim2 , (dim3,dim2,3,3))
        self.params["b3"] = np.zeros((dim3))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma3"] = np.zeros((dim3))
        self.params["beta3"] = np.zeros((dim3))
        self.paramsNoUp["running_mean3"] = np.zeros((dim3))
        self.paramsNoUp["running_var3"] = np.zeros((dim3))
        #maxPool
        self.params["W4"] = np.random.normal(0, 2./dim3 , (dim4,dim3,3,3))
        self.params["b4"] = np.zeros((dim4))# , 32-filter_size+1, 32-filter_size+1))
        self.params["gamma4"] = np.zeros((dim4))
        self.params["beta4"] = np.zeros((dim4))
        self.paramsNoUp["running_mean4"] = np.zeros((dim4))
        self.paramsNoUp["running_var4"] = np.zeros((dim4))

        # self.params["W4"] = np.random.normal(0, 2./dim3/64 , (dim3*8*8,dim3))
        # self.params["b4"] = np.zeros(dim3)
        self.params["W5"] = np.random.normal(0, 2./dim4/16 , (dim4*4*4,num_classes))
        self.params["b5"] = np.zeros(num_classes)

        #self.params["conv_param"] = {"stride":1,"pad":0}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        input = X
        result={}
        #print(input.shape,self.params["W1"].shape ,self.params["b1"].shape )
      #   eps: Constant for numeric stability
      # - momentum: Constant for running mean / variance. momentum=0 means that
      #   old information is discarded completely at every time step, while
      #   momentum=1 means that new information is never incorporated. The
      #   default of momentum=0.9 should work well in most situations.
      # - running_mean: Array of shape (D,) giving running mean of features
      # - running_var Array of shape (D,) giving running variance of features
        input,result["cacheconv1"] = conv_bn_relu_forward(input,self.params["W1"] , self.params["b1" ], self.params["gamma1" ], self.params["beta1" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean1"],"running_var":self.paramsNoUp["running_var1"]})
        input,result["cachemaxPool1"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv2"] = conv_bn_relu_forward(input,self.params["W2"] , self.params["b2" ], self.params["gamma2" ], self.params["beta2" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean2"],"running_var":self.paramsNoUp["running_var2"]})
        input,result["cachemaxPool2"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv3"] = conv_bn_relu_forward(input,self.params["W3"] , self.params["b3" ], self.params["gamma3" ], self.params["beta3" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean3"],"running_var":self.paramsNoUp["running_var3"]})
        input,result["cachemaxPool3"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        input,result["cacheconv4"] = conv_bn_relu_forward(input,self.params["W4"] , self.params["b4" ], self.params["gamma4" ], self.params["beta4" ],{"pad":1,"stride":1},{'mode': 'train',"running_mean": self.paramsNoUp["running_mean4"],"running_var":self.paramsNoUp["running_var4"]})
        #input,result["cachemaxPool3"] = max_pool_forward_fast(input,{"pool_height":2,"pool_width":2,"stride":2})
        #print(input.shape,self.params["W2"].shape,self.params["b2" ].shape)
        #input,result["cacheaff1"] = affine_relu_forward(input,self.params["W4"] , self.params["b4" ])
        scores,result["cacheaff2"] = affine_forward(input,self.params["W5"] , self.params["b5" ])
        #scores,scores_cache = affine_forward(input , self.params["W"+str(self.num_layers - 1)] , self.params["b" + str(self.num_layers - 1)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss , dloss  = softmax_loss(scores,y)
        dup , grads["W5"],grads["b5"], = affine_backward(dloss,result["cacheaff2"])
        #dup , grads["W4"],grads["b4"] =  affine_relu_backward(dup,result["cacheaff1"])
        #dx, dw, db, dgamma, dbeta
        dup, grads["W4"],grads["b4"],grads["gamma4"],grads["beta4"] = conv_bn_relu_backward(dup,result["cacheconv4"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool3"])
        dup, grads["W3"],grads["b3"],grads["gamma3"],grads["beta3"] = conv_bn_relu_backward(dup,result["cacheconv3"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool2"])
        dup, grads["W2"],grads["b2"],grads["gamma2"],grads["beta2"] = conv_bn_relu_backward(dup,result["cacheconv2"])
        dup   =  max_pool_backward_fast(dup,result["cachemaxPool1"])
        dup, grads["W1"],grads["b1"],grads["gamma1"],grads["beta1"] = conv_bn_relu_backward(dup,result["cacheconv1"])
        for para in self.params:
            #if para in grads:
            loss += self.reg * np.sum(self.params[para]*self.params[para])
            grads[para] += 2 * self.reg * self.params[para]

        # for i in range(self.num_layers):
        #     loss +=  self.reg * np.sum(self.params["W"+str(i)] * self.params["W"+str(i)])
        #     grads["W"+str(i)] -= 2 * self.reg * self.params["W"+str(i)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

