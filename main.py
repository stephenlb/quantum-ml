## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import json
import numpy as np
import matplotlib.pyplot as plt
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

    nn = DeepNN(
        learn=0.005
    ,   epochs=900
    ,   batch=10
    ,   bias=1
    ,   density=3
    ,   high=3.0
    ,   low=-3.0
    )
    nn.load(features=features, labels=labels)
    nn.train()
    results = nn.predict(features)
    print(np.array(nn.loss))
    print(np.column_stack((
        results
    ,   np.where(results > 0.5, 1, 0)
    ,   np.array(labels)
    )))

    #fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    #fig.suptitle('Loss')
    #y = x = [x for x in range(len(nn.loss))]
    #ax.errorbar(x, nn.loss, yerr=nn.loss)
    #plt.show()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Neural Network as a Support Vector Machine
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NeuralNetwork():
    def initalize(
        self
    ,   learn=0.1
    ,   batch=10
    ,   epochs=500
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

    def build(self):
        layers = len(self.unbuilt)-1
        shape  = (self.shape[0], self.density * self.shape[0], self.shape[1])
        self.layers = [
            layer.builder(
                name=layer.name,
                size=(
                    shape[0] if i == 0      else shape[1],
                    shape[2] if i == layers else shape[1]
                ),
                high=self.high,
                low=self.low,
                activation=layer.activation
            ) for i, layer in enumerate(self.unbuilt)
        ]

    def load(
        self
    ,   features=[[1,1],[0,0],[1,0],[0,1]]
    ,   labels=  [ [1],  [1],  [0],  [0] ]
    ):
        s, f, l       = len(features), len(features[0]) + 1, len(labels[0])
        bias          = np.full((len(features), 1), self.bias)
        self.features = np.concatenate((features, bias), axis=1)
        self.labels   = np.array(labels)
        self.length   = len(features)
        self.shape    = (f, l)

        self.build()

    ## TODO
    ## TODO
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

    def add(self, builder, name='Unamed', activation='none'):
        self.unbuilt.append(LayerLoader(
            builder=builder
        ,   name=name
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
        
    def train(self):
        if not self.initalized(): return

        for epoch in range(self.epochs):
            features, labels = self.batcher()

            result   = self.forward(features)
            gradient = 2 * (result - labels)

            self.backward(gradient)
            self.optimize()
            self.loss.append(np.sum(gradient) ** 2)

    def predict(self, features):
        bias   = np.full((len(features), 1), self.bias)
        inputs = np.concatenate((np.array(features), bias), axis=1)

        return self.forward(inputs)

    def forward(self, inputs):
        if not self.initalized(): return

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return self.layers[-1].result

    ## TODO embed in Layer (cuz we have different kinds of layers!)
    ## TODO embed in Layer (cuz we have different kinds of layers!)
    ## TODO embed in Layer (cuz we have different kinds of layers!)
    ## TODO embed in Layer (cuz we have different kinds of layers!)
    def backward(self, gradient):
        for layer in self.layers[::-1]:
            layer.gradient = layer.input.T @ gradient
            gradient       = gradient.dot(layer.weights.T) * layer.derivative(layer.input)

        return gradient

    def optimize(self):
        for layer in self.layers:
            layer.weights -= self.learn * layer.gradient

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
        #self.add(StandardLayer, name='ReLU',      activation='relu')
        #self.add(StandardLayer, name='LeakyReLU', activation='lrelu')
        self.add(StandardLayer, name='Sigmoid',   activation='sigmoid')
        self.add(StandardLayer, name='Sigmoid',   activation='sigmoid')
        self.add(StandardLayer, name='Sigmoid',   activation='sigmoid')
        self.add(StandardLayer, name='Sigmoid',   activation='sigmoid')
        self.add(StandardLayer, name='Sigmoid',   activation='sigmoid')
        self.add(StandardLayer, name='Output',    activation='linear')

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
        self
    ,   name="My Layer"
    ,   size=(5,5)
    ,   high=5
    ,   low=-5
    ,   activation='lrelu'
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

    def backward(self): pass

    def linear(N):   return N
    def lineard(N):  return N

    def relu(N):     return np.where(N > 0, N, 0)
    def relud(N):    return np.where(N > 0, 1, 0)

    def lrelu(N):    return np.where(N > 0, N, N * 0.5)
    def lrelud(N):   return np.where(N > 0, 1, N * 0.01)

    def sigmoid(N):  return 1 / (1 + np.exp(-N))
    def sigmoidd(N): return N * (1 - N)

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
