## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
from dwave_qbsolv import QBSolv
from dwave.system import EmbeddingComposite, DWaveSampler

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
    nn = NN()
    nn.load(data)
    nn.train()

    ## Predict
    prediction = nn.predict(nn.X)
    answers    = np.round(prediction)
    accuracy   = (1 - np.average(np.abs((np.round(prediction) - nn.Y))))*100

    print(nn)
    print("Results:\n%s"             % prediction)
    print("\nAccuracy: %s%%"         % accuracy)
    print(np.column_stack((answers, nn.Y)))
    print("")

    ## Quantum Model
    #qnn = QuantumNN()
    #nn.load(data)
    #qnn.train2()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Basic ML Model using Leaky ReLU
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NN():
    def initalize(self, epochs=500, learning=0.03, units=10, features=3, labels=1):
        self.epochs    = epochs
        self.learning  = learning
        self.results   = {}
        self.gradients = {}

        f, u, l = features, units, labels

        self.weights   = {
            'hidden' : np.random.uniform(size=(f, u), low=-0.4), ## Hidden Weights
            'output' : np.random.uniform(size=(u, l), low=-0.4), ## Output Weights
        }

    def load(self, data):
        H = self.H = np.array(data[0])                         ## Headers / Column Names
        X = self.X = np.array([x[0:3]      for x in data[1:]]) ## Input Features for Training
        Y = self.Y = np.array([[int(y[3])] for y in data[1:]]) ## Output Labels (Target Answers) for Training

    def train(self):
        for epoch in range(self.epochs):
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
## Quantum ML Model using QUBO
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumNN(NN):
    def train(self):
        #Q = {(0, 0): 1, (1, 1): 1, (0, 1): 1}
        Q = {('A','A'):   1,
             ('A','B'):  -1,
             ('B','A'):  -1,
             ('B','B'):   1,
             ('C','C'):   1,
             ('C','A'):   1,
             ('C','B'):   1}
        response = QBSolv().sample_qubo(Q, num_reads=1000, postprocess='sampling')
        print(response)
        print("samples=%s" % list(response.samples()))
        print("energies=%s" % list(response.data_vectors['energy']))

        for sample in response:
            print(sample)

        print(response.data)

    def train2(self):
        Q = {('A','A'):   1,
             ('A','B'):  -1,
             ('B','A'):  -1,
             ('B','B'):   1,
             ('C','C'):   1,
             ('C','A'):   1,
             ('C','B'):   1}
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

