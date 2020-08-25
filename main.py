## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import json
import numpy as np
#import neal
#import dimod

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Configuration
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
DWAVE_API_KEY = os.getenv('DWAVE_API_KEY')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    features = [[1,1],[0,0],[1,0],[0,1]]
    labels   = [ [1],  [1],  [0],  [0] ]

    nn = ClassicalNN(learn=0.1, bias=0.1, density=1, high=5, low=-5)
    nn.save()
    nn.load(features=features, labels=labels)
    print(nn.predict(features))
    print(nn.dumps())

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NeuralNetwork():
    def initalize(self, learn=0.1, batch=10, bias=0.1, density=5, high=5, low=-5):
        self.batch    = batch   ## batch size
        self.learn    = learn   ## learning rate
        self.bias     = bias    ## bias node starting value
        self.density  = density ## number of units "neurons"
        self.high     = high    ## initial weights upper limit
        self.low      = low     ## initial weights lower limit
        self.features = None    ## input training features
        self.labels   = None    ## output training labels
        self.unbuilt  = []      ## prototype of neural network layers
        self.layers   = []      ## fully built network after data load

    def build(self):
        shape = (self.shape[0], self.density * self.shape[0], self.shape[1])
        self.layers = [
            layer.builder(
                name=layer.name,
                size=(
                    shape[0] if i == 0                   else shape[1],
                    shape[2] if i == len(self.unbuilt)-1 else shape[1]
                ),
                high=self.high,
                low=self.low,
                activation=layer.activation
            ) for i, layer in enumerate(self.unbuilt)
        ]

    ## TODO
    def loads(self, data):
        pass
        ## TODO
        #layers = json.loads(data)
        #for layer in layers: pass
        ## ... import layers
        #self.build()

    def save(self):
        if not len(self.layers):
            raise Exception("Uninitialized Network: use network.load(...)")

        return json.dumps([{
            'name'       : layer.name
        ,   'type'       : layer.type
        ,   'weights'    : layer.weights.tolist()
        ,   'activation' : layer.activation
        } for layer in self.layers])

    def add(self, builder, name='Unamed', activation='none'):
        self.unbuilt.append(LayerLoader(
            builder=builder
        ,   name=name
        ,   activation=activation
        ))

    def load(
        self,
        features=[[1,1],[0,0],[1,0],[0,1]],
        labels=  [ [1],  [1],  [0],  [0] ]
    ):
        s, f, l       = len(features), len(features[0]) + 1, len(labels[0])
        self.features = np.array(features) + np.full((s, 1), self.bias)
        self.labels   = np.array(labels)
        self.length   = len(features)
        self.shape    = (f, l)

        self.build()

    def train(self, data):
        if not len(self.layers):
            raise Exception("Uninitialized Network: use network.load(...)")

    def predict(self, features):
        bias   = np.full((len(features), 1), self.bias)
        train  = np.concatenate((np.array(features),bias), axis=1)
        result = train

        for layer in self.layers:
            result = layer.result = layer.activator(
                result.dot(layer.weights)
            )

        return self.layers[-1].result
        #self.results['hidden'] = self.relu(np.dot(X, self.weights['hidden']))
        #self.results['output'] = np.dot(self.results['hidden'], self.weights['output'])
        #return self.results['output']

    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumNN(NeuralNetwork):
    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Deep Quantum Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumDeepNN(NeuralNetwork):
    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Deep Classical SVM Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class DeepNN(NeuralNetwork):
    def __init__(self, **kwargs):
        self.initalize(**kwargs)
        self.add(StandardLayer, name='Hidden One',   activation='lrelu')
        self.add(StandardLayer, name='Hidden Two',   activation='lrelu')
        self.add(StandardLayer, name='Hidden Three', activation='lrelu')
        self.add(StandardLayer, name='Output',       activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Classical SVM Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class ClassicalNN(NeuralNetwork):
    def __init__(self, **kwargs):
        self.initalize(**kwargs)
        self.add(StandardLayer, name='Hidden', activation='lrelu')
        self.add(StandardLayer, name='Output', activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Layer Loader
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class LayerLoader():
    def initalize(self, builder, name='Unamed', activation='none'):
        self.builder    = builder
        self.name       = name
        self.activation = activation

    def __init__(self, **kwargs): self.initalize(**kwargs)


## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Neural Network Layer in a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class BaseLayer():
    def initalize(
        self,
        name="My Layer",
        size=(5,5),
        high=5,
        low=-5,
        activation='lrelu',
    ):
        self.name       = name
        self.type       = 'undefined'
        self.size       = size
        self.weights    = Tensor(size=size, high=high, low=low).matrix
        self.gradient   = None
        self.result     = None
        self.activation = activation
        self.activator  = getattr(BaseLayer, activation)
        self.deriver    = getattr(BaseLayer, activation+'d')
        #self.activate   = self.activation(activation)
        #self.derive     = self.derivative(activation)

    def forward(self): pass
    def bacwkard(self): pass

    def linear(N):   return N
    def lineard(N):  return N

    def relu(N):     return np.where(N > 0, N, 0)
    def relud(N):    return np.where(N > 0, 1, 0)

    def lrelu(N):    return np.where(N > 0, N, N * 0.01)
    def lrelud(N):   return np.where(N > 0, 1, 1e-9)

    def sigmoid(N):  return 1 / (1 + np.exp(-N))
    def sigmoidd(N): return N * (1 - N)

    """
    def activation(self, method):
        return {
            'relu'    : self.relu
        ,   'lrelu'   : self.lrelu
        ,   'sigmoid' : self.sigmoid
        ,   'linear'  : self.linear
        }[method]

    def derivative(self, method):
        return {
            'relu'    : self.relud
        ,   'lrelu'   : self.lrelud
        ,   'sigmoid' : self.sigmoidd
        ,   'linear'  : self.linear
        }[method]
    """

    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Layer in a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class StandardLayer(BaseLayer):
    def __init__(self, **kwargs):
        self.initalize(**kwargs)
        self.type = 'standard'

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum Layer in a Support Vector Machine - REQUIRES QPU HARDWARE
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumHardwareLayer(BaseLayer):
    def forward(self): pass
    def __init__(self, **kwargs):
        self.initalize(**kwargs)
        self.type = 'quantum-hardware'

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum Layer in a Support Vector Machine - Simulates QPU
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumSimulatorLayer(BaseLayer):
    def forward(self): pass
    def __init__(self, **kwargs):
        self.initalize(**kwargs)
        self.type = 'quantum-simulator'

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Tensor Matrix in a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Tensor():
    def initalize(self, size=(3,3), low=-10, high=10):
        self.matrix = np.random.uniform(size=size, low=low, high=high)

    def __init__(self, **kwargs): self.initalize(**kwargs)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()
