## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
#from dwave.system import EmbeddingComposite, DWaveSampler

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Basic ML Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NN():
    def weights(self, x=3, y=3):
        self.weights = {
            'hidden' : np.random.rand(x,y), ## Hidden Layer Weights
            'output' : np.random.rand(y,1), ## Output Layer Weights
        }
        self.weights['hidden'] -= 0.5
        self.weights['output'] -= 0.5

    def data(self, data):
        H = self.H = np.array(data[0])                        ## Headers / Column Names
        X = self.X = np.array([x[0:3]    for x in data[1:]])  ## Input Features for Training
        Y = self.Y = np.array([int(y[3]) for y in data[1:]])  ## Output Labels (Tartget Answers) for Training
        #print(H)                                              ## Headers
        #print(X)                                              ## Features
        #print(Y)                                              ## Labels (Target Answer)

    def train(self):
        pass

    def predict(self, X):
        a = self.relu(np.dot(self.X, self.weights['hidden']))
        return np.dot(a, self.weights['output'])

    def relu(self, N):  return np.maximum(N, 0)
    def relud(self, N): return (N > 0) * 1

    def __init__(self): self.weights()
    def __str__(self):
        return '\nHidden Layer:\n' + str(self.weights['hidden']) + '\n' + \
               '\nOutput Layer:\n' + str(self.weights['output']) + '\n'

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Init Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
nn_model = NN()
print(nn_model)

## Training Data (XOR)
data = [
    ['A', 'B', 'Bias', 'Answer'],
    [ 0,   0,   1,      True,  ],
    [ 1,   1,   1,      True,  ],
    [ 1,   0,   1,      False, ],
    [ 0,   1,   1,      False, ],
]

nn_model.data(data)
print(nn_model.predict(data))



## Calculate edges to fit into a QUBO
#Q = dict()
#Q.update(dict(((k, k), np.sum(v)) for (k, v) in enumerate(X)))

#print(Q)


#def predict(X):


#Q = np.array()

# x or

Q = {('A','A'):   1,
     ('A','B'):  -1,
     ('B','A'):  -1,
     ('B','B'):   1,
     ('C','C'):   1,
     ('C','A'):   1,
     ('C','B'):   1}
#print(Q)
#print(Q)

def nvm():
    # Define the sampler that will be used to run the problem
    sampler = EmbeddingComposite(DWaveSampler())

    # Run the problem on the sampler and print the results
    # postprocess='sampling' answer_mode='histogram',
    sampleset = sampler.sample_qubo(Q, num_reads=1000, postprocess='sampling')

    #print(sampleset.info)
    print(sampleset)
    for sample in sampleset:
        print(sample)
    
    print(sampleset.data)

