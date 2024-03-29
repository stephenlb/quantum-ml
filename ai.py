## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import json
import numpy as np
import neal
import dimod
import dwave
import dwave.system as ds

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Configuration
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
DWAVE_API_KEY = os.getenv('DWAVE_API_KEY')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NeuralNetwork():
    def initalize(
        self
    ,   learn=0.1
    ,   batch=10
    ,   epochs=100
    ,   bias=0.1
    ,   density=5
    ,   high=5
    ,   low=-5
    ):
        self.batch     = batch   ## batch size
        self.epochs    = epochs  ## training iterations using a batch each epoch
        self.learn     = learn   ## learning rate
        self.bias      = bias    ## bias node starting value
        self.density   = density ## number of units "neurons"
        self.high      = high    ## initial weights upper limit
        self.low       = low     ## initial weights lower limit
        self.features  = None    ## input training features
        self.labels    = None    ## output training labels
        self.unbuilt   = []      ## prototype of neural network layers
        self.layers    = []      ## fully built network after data load
        self.loss      = []      ## loss from most recent training
        self.loss_avg  = []      ## loss average from most recent training

    def build(self):
        layers = len(self.unbuilt) - 1
        shape  = [self.shape[0], self.density * self.shape[0], self.shape[1]]
        size   = [0,0]

        for i, layer in enumerate(self.unbuilt):
            size[0] = shape[0] if i == 0      else shape[1]
            size[1] = shape[2] if i == layers else shape[1]
            if layer.shape[0]: size[0] = layer.shape[0]
            if layer.shape[1]: size[0] = layer.shape[1]
            self.layers.append(layer.builder(
                name=layer.name,
                size=size,
                high=self.high,
                low=self.low,
                activation=layer.activation
            ))
        for i, layer in enumerate(self.layers):
            print(i, layer.size, layer.weights.shape, layer.name)

        #raise Exception("asdf")

    def load(self, features=[[1],[0]], labels=[[1],[0]]):
        s, f, l       = len(features), len(features[0]) + 1, len(labels[0])
        bias          = np.full((len(features), 1), self.bias)
        self.features = np.concatenate((features, bias), axis=1)
        self.labels   = np.array(labels)
        self.length   = len(features)
        self.shape    = (f, l)
        self.build()

    ## TODO
    ## TODO
    def loadJSON(self, data):
        pass
        ## TODO
        #layers = json.loads(data)
        #for layer in layers: pass
        ## ... import layers
        #self.build()

    def saveJSON(self):
        if not self.initalized(): return
        return json.dumps([{
            'name'       : layer.name
        ,   'type'       : layer.type
        ,   'weights'    : layer.weights.tolist()
        ,   'activation' : layer.activation
        } for layer in self.layers])

    def add(self, builder, name='Unamed', activation='none', shape=[None,None]):
        self.unbuilt.append(LayerLoader(
            builder=builder
        ,   name=name
        ,   shape=shape
        ,   activation=activation
        ))

    def initalized(self):
        if not len(self.layers):
            raise Exception("Uninitialized Network: use network.load(...)")
            return False
        return True
        
    def batcher(self):
        features = []
        labels   = []
        for index in np.random.randint(self.length, size=self.batch):
            features.append(self.features[index])
            labels.append(self.labels[index])
        return np.array(features), np.array(labels)
        
    def train(self, progress=None, updates=10):
        if not self.initalized(): return

        for epoch in range(self.epochs):
            features, labels = self.batcher()
            result = self.forward(features)
            gradient = 2 * (result - labels)
            self.backward(gradient)
            self.optimize()
            self.loss.append(np.sum(gradient) ** 2)
            self.loss_avg.append(np.average(self.loss))

            if progress and self.epochs > updates and \
            not (epoch % int(self.epochs / updates)):
                progress(epoch)

    def predict(self, features):
        bias   = np.full((len(features), 1), self.bias)
        inputs = np.concatenate((np.array(features), bias), axis=1)
        return self.forward(inputs)

    def forward(self, inputs):
        if not self.initalized(): return
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return self.layers[-1].result

    def backward(self, gradient):
        for layer in self.layers[::-1]:
            gradient = layer.backward(gradient)
        return gradient

    def optimize(self):
        for layer in self.layers:
            layer.weights -= self.learn * layer.gradient

    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumNN(NeuralNetwork):
    def initalize(self, **kwargs):
        super().initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Deep Quantum Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumDeepNN(NeuralNetwork):
    def initalize(self, **kwargs):
        super().initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Deep Classical SVM Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class DeepNN(NeuralNetwork):
    def initalize(self, **kwargs):
        super().initalize(**kwargs)
        self.add(StandardLayer, name='Sigmoid',       activation='essigmoid')
        self.add(StandardLayer, name='ElliotSigmoid', activation='esigmoid')
        self.add(StandardLayer, name='Output',        activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Classical SVM Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class ClassicalNN(NeuralNetwork):
    def initalize(self, **kwargs):
        super().initalize(**kwargs)
        self.add(StandardLayer, name='Hidden', activation='lrelu')
        self.add(StandardLayer, name='Output', activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Layer Loader
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class LayerLoader():
    def initalize(
        self
    ,   builder
    ,   name='Unamed'
    ,   activation='none'
    ,   shape=[None,None]
    ):
        self.builder    = builder
        self.name       = name
        self.activation = activation
        self.shape      = shape

    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Neural Network Layer in a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class BaseLayer():
    def initalize(
        self
    ,   name="My Layer"
    ,   size=(5,5)
    ,   high=5
    ,   low=-5
    ,   activation='esigmoid'
    ):
        self.name       = name
        self.type       = 'undefined'
        self.size       = size
        self.weights    = Tensor(size=size, high=high, low=low).matrix
        self.gradient   = None
        self.result     = None
        self.input      = None
        self.activation = activation
        self.activator  = getattr(BaseLayer, activation)
        self.derivative = getattr(BaseLayer, activation+'d')

    def forward(self, inputs): 
        self.input  = inputs
        self.result = self.activator(inputs @ self.weights)
        return self.result

    def backward(self, gradient):
        self.gradient = self.input.T @ gradient
        return (gradient @ self.weights.T) * self.derivative(self.input)

    def binary(N):     return np.where(N > 0.5, 1, 0)
    def binaryd(N):    return np.where(N > 0.5, 1, 0)

    def linear(N):     return N
    def lineard(N):    return N

    def relu(N):       return np.where(N > 0, N, 0)
    def relud(N):      return np.where(N > 0, 1, 0)

    def lrelu(N):      return np.where(N > 0, N, N * 0.01)
    def lrelud(N):     return np.where(N > 0, 1,     0.01)

    def tanh(N):       return np.tanh(N)
    def tanhd(N):      return 1 - N ** 2

    def sigmoid(N):    return 1 / (1 + np.exp(-N))
    def sigmoidd(N):   return N * (1 - N)

    def esigmoid(N):   return 1 / (1 + np.abs(N))
    def esigmoidd(N):  return 1 / (1 + np.abs(N)) ** 2

    def essigmoid(N):  return 0.5 * N / (1 + np.abs(N)) + 0.5
    def essigmoidd(N): return 1 / (1 + np.abs(N)) ** 2

    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Layer in a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class StandardLayer(BaseLayer):
    def initalize(self, **kwargs):
        super().initalize(**kwargs)
        self.type = 'standard'

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum Layer in a Support Vector Machine - Simulates QPU
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumSimulatorLayer(BaseLayer):
    def initalize(self, learn=0.02, **kwargs):
        super().initalize(**kwargs)
        self.type     = 'quantum-simulator'
        self.learn    = learn
        self.qweights = None
        self.qubo     = None
        self.bqm      = None
        size          = self.weights.shape[0]
        self.hidden   = StandardLayer(
            name="Quantum Weights"
        ,   size=(size,size)
        ,   activation='sigmoid'
        #,   activation='linear'
        ,   high=2.0
        ,   low=-2.0
        )

    def backward(self, gradient):
        self.gradient = self.input.T @ gradient
        gradient = (gradient @ self.weights.T) * self.derivative(self.input)
        gradient = self.hidden.backward(gradient)
        self.hidden.weights -= self.learn * self.hidden.gradient
        return gradient

    def qpu(self, bqm):
        return dimod.ExactSolver().sample(bqm)

    def anneal(self):
        if self.bqm: return# and np.random.random() < 0.9999: return

        qubo = self.qubo = {k: w for k, w in np.ndenumerate(self.hidden.weights)}
        bqm  = self.bqm  = dimod.BinaryQuadraticModel({}, qubo, 0.0, 'BINARY')
        #bqmm = self.bqmm = np.round(np.float64(bqm.to_numpy_matrix()))

        samples  = self.qpu(bqm)
        qweights = np.array([
            [s] for s in samples.first.sample.values()
        ], dtype='float64')

        self.qweights = qweights
        
    def forward(self, inputs): 
        inputs = self.hidden.forward(inputs)

        self.anneal()

        self.input  = inputs
        self.result = self.activator(
            inputs @ (self.weights * (self.qweights + 1e-9))
        )

        return self.result

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum Layer in a Support Vector Machine - REQUIRES QPU HARDWARE
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumHardwareLayer(QuantumSimulatorLayer):
    def qpu(self, bqm):
        sampler = ds.EmbeddingComposite(ds.DWaveSampler(token=DWAVE_API_KEY))
        return sampler.sample(bqm, num_reads=1000)

    def initalize(self, **kwargs):
        super().initalize(**kwargs)
        self.type = 'quantum-hardware'

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Tensor Matrix in a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Tensor():
    def initalize(self, size=(3,3), low=-10, high=10):
        self.matrix = np.random.uniform(size=size, low=low, high=high)

    def __init__(self, **kwargs): self.initalize(**kwargs)
