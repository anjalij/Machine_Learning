import numpy
import theano
import theano.tensor as T
import theano.tensor.nnet import conv2d

#_____Defining input Tensor_______
X=T.matrix('X') #The shape of the tensor is as follows: [mini-batch size, number of input feature maps, image height, image width].


#_____Useful numbers_______
seq_size=556
stepSi=1 #convolutional step size
X1=10 #filter size of first layer
X2=15 #filter size of second layer
X3=20 #filter size of third layer
noAA=20 #This serves as the number of the number of maps in the first layer
noFil=noAA  #number of maps(colors) in each layer
batch_size=1 #number of images to 

#_____Random Number Generator
rng=numpy.random.RandomState(23455)



class HiddenLayer(object):
	def__init__(self,rng,input,n_in,n_out,W=none,b=None)
		"""
		type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input=input

        if W is None:
        	W_Values=numpy.asarray(
        			rng.uniform(
        				low=-4*numpy.sqrt(6./(n_in+n_out)),
        				high=4*numpy.sqrt(6./(n_in+n_out)),
        				size=(n_in,n_out)
        				),
        				dtype=theano.config.floatX
        			)
        	W=theano.shared(value=W_values,name='W',borrow=True)

        if b is None:
        	b_values=numpy.zeros((n_out,),dtype=theano.config.floatX)
        	b=theano.shared(value=b_values,name='b',borrow=True)

        self.W=W
        self.b=b
        lin_output=T.dot(input,self.W)+self.b
        self.output=T.sigmoid(lin_output)

class ConvLayer(object):
	def__init__(self,rng,input,filter_shape,input_shape)
		"""
	    :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
         """

      	self.input=input


      	f_in=numpy.prod(filter_shape[1:]) #inputs to each unit =num input feature maps*filter heigh*filter width
      	f_out=noFil*numpy.prod(filter_shape[2:]) # This seems wrong: the only difference between fin/fout is that if you change the no filters between layers
      	W_bound=numpy.sqrt(6./(f_in+f_out))
      	self.W=theano.shared(numpy.asarray(rng.uniform(low=-4*W_bound,high=4*W_bound,size=filter_shape),dtype=theano.config.floatX),borrow=True)

      	#the bias, one bias per output feature map
      	b_values=numpy.zeros((filter_shape[0]),dtype=theano.config.floatX)
      	self.b=theano.shared(value=b_values,borrow=True)

      	#convolve input 
      	conv_out=conv2d(
      		input=input,
      		filters=self.W,
      		input_shape=input_shape
      		subsample=stepSi #I think we can adjust the size of the convolutional hop
      		)
      	

      	self.output=T.sigmoid(conv_out+self.b.dimshuffle('x',0,'x','x')) #don't really understand dimshuffle

      	self.params = [self.W, self.b]

      	self.input = input #why?

#___Defining Layers____

layer0_input=X.reshape((batch_size,noFil,seq_size,1))
layer0=ConvLayer(rng)