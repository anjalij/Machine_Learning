import numpy
import theano.tensor as T
from theano import function

#---define the imput tensor

x=T.matrix('x')

#---define the logistic function
s = 1 / (1 + T.exp(-x))
f=function([x],s)
print f(2,3)