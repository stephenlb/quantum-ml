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

    prediction = nn_model.predict(nn_model.X)
    accuracy = 100.0 - np.sum(np.round(prediction) - nn_model.Y)
    print("Predictions:\n%s" % prediction)

    print("\nAccuracy: %s%%" % accuracy)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Basic ML Model using Leaky ReLU
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NN():
    def initalize(self, epochs=100, learning=0.03, features=3, units=10, labels=1):
        self.epochs    = epochs
        self.learning  = learning
        self.results   = {}
        self.gradients = {}

        f, u, l = features, units, labels

        self.weights   = {
            'hidden' : np.random.uniform(size=(f, u), low=-0.5), ## Hidden Weights
            'output' : np.random.uniform(size=(u, l), low=-0.5), ## Output Weights
        }

    def load(self, data):
        H = self.H = np.array(data[0])                         ## Headers / Column Names
        X = self.X = np.array([x[0:3]      for x in data[1:]]) ## Input Features for Training
        Y = self.Y = np.array([[int(y[3])] for y in data[1:]]) ## Output Labels (Target Answers) for Training

    def train(self):
        for epoch in xrange(self.epochs):
            self.predict(self.X)
            error = self.Y - self.results['output']

            ## Train Output Layer
            self.gradients['output'] = error * self.learning
            self.weights['output']  += np.dot(
                self.results['hidden'].T,
                self.gradients['output']
            )

            ## Train Hidden Layer
            self.gradients['hidden'] = np.dot(
                self.gradients['output'],
                self.weights['output'].T
            ) * self.relud(self.results['hidden'])
            self.weights['hidden'] += np.dot(self.X.T, self.gradients['hidden'])

    def predict(self, X):
        self.results['hidden'] = self.relu(np.dot(X, self.weights['hidden']))
        self.results['output'] = np.dot(self.results['hidden'], self.weights['output'])

        return self.results['output']

    def relu(self, N):    return np.where(N > 0, N, N * 0.01)
    def relud(self, N):   return np.where(N > 0, 1, 0)
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

