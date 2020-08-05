## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
#from dwave.system import EmbeddingComposite, DWaveSampler

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    ## Training Data (XOR)
    data = [
        ['A', 'B', 'Bias', 'Answer'],
        [ 0,   0,   1,      True,  ],
        [ 1,   1,   1,      True,  ],
        [ 1,   0,   1,      False, ],
        [ 0,   1,   1,      False, ],
    ]

    ## Training Model
    nn_model = NN()
    nn_model.load(data)
    nn_model.train()

    print(nn_model)

    print("Predictions:")
    print(nn_model.predict(nn_model.X))

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Basic ML Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NN():
    def initalize(self, x=3, y=3):
        self.epochs    = 100
        self.learning  = 0.2
        self.results   = {}
        self.gradients = {}
        self.weights   = {
            'hidden' : np.random.rand(x, y) - 0.5, ## Hidden Layer Weights
            'output' : np.random.rand(y, 1) - 0.5, ## Output Layer Weights
        }

    def load(self, data):
        H = self.H = np.array(data[0])                         ## Headers / Column Names
        X = self.X = np.array([x[0:3]      for x in data[1:]]) ## Input Features for Training
        Y = self.Y = np.array([[int(y[3])] for y in data[1:]]) ## Output Labels (Tartget Answers) for Training

    def train(self):
        for epoch in xrange(self.epochs):
            self.predict(self.X)
            #error = np.square(self.Y - self.results['output'])
            error = (self.Y - self.results['output'])

            self.gradients['output'] = error * self.learning
            self.weights['output'] += np.dot(self.results['hidden'].T, self.gradients['output'])

            self.gradients['hidden'] = np.dot(self.gradients['output'], self.weights['output'].T) * \
                self.relud(self.results['hidden'])
            self.weights['hidden'] += np.dot(self.X.T, self.gradients['hidden'])

            #if not (epoch % (self.epochs / 10)): print(epoch)
        #if self.epochs > epoch: self.train(epoch + 1)

    def predict(self, X):
        self.results['hidden'] = self.relu(np.dot(X, self.weights['hidden']))
        self.results['output'] = np.dot(self.results['hidden'], self.weights['output'])

        return self.results['output']

    def relu(self, N):    return np.where(N > 0, N, N * 0.01) #0.01 * N if N <= 0 else N
    def relud(self, N):   return (N > 0) * 1
    def sigmoid(self, N): return 1 / (1 + np.exp(-N))

    def __init__(self, **kwargs): self.initalize(**kwargs)
    def __str__(self):
        return '\nHidden Layer:\n' + str(self.weights['hidden']) + '\n' + \
               '\nOutput Layer:\n' + str(self.weights['output']) + '\n'

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()



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

