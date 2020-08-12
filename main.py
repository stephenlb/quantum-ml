## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import numpy as np

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
    #nn = NN()
    #nn.load(data)
    #nn.train()

    ## Predict
    #prediction = nn.predict(nn.X)
    #answers    = np.round(prediction)
    #accuracy   = (1 - np.average(np.abs((np.round(prediction) - nn.Y))))*100

    #print(nn)
    #print("Results:\n%s"             % prediction)
    #print("\nAccuracy: %s%%"         % accuracy)
    #print(np.column_stack((answers, nn.Y)))
    #print("")

    ## Quantum Model
    print("Running Quantum Model")
    qnn = QuantumNN()
    qnn.load(data)
    qnn.train()

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
        import dimod

        X = self.X
        Y = self.Y
        
        print("X\n%s\n" % X)
        print("Y\n%s\n" % Y)

        y = np.where( Y > 0.0, Y, -1.0)
        x = np.where( X > 0.0, X, -1.0)
        q = x.dot(x.T)
        Q = dict()
        Q.update(dict((k, v) for k, v in np.ndenumerate(q)))

        print("x\n%s\n" % x)
        print("y\n%s\n" % y)
        print("q\n%s\n" % q)
        print("Q\n%s\n" % Q)

        bqm = dimod.BinaryQuadraticModel(y, Q, 0.0, dimod.SPIN)
        print("bqm: \n%s\n" % bqm.adj)

        #solver = dimod.ExactSolver()
        #solver.sample_ising





    def train3(self):
        import neal
        import dimod
        from dwave_qbsolv import QBSolv
        from dwave.system import LeapHybridSampler
        from dwave.system import EmbeddingComposite

        #Q = {(0, 0): 1, (1, 1): 1, (0, 1): 1}
        Q1 = {
            (0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1,
            (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1,
            (2, 0): 1, (2, 1): -1, (2, 2): 1, (2, 3): 1,
            (3, 0): -1, (3, 1): -1, (3, 2): 1, (3, 3): 1,
            (4, 0): 1, (4, 1): -1, (4, 2): 1, (4, 3): 1,
            (5, 0): 1, (5, 1): -1, (5, 2): -1, (5, 3): 1,
            (6, 0): 1, (6, 1): -1, (6, 2): -1, (6, 3): 1,
            (7, 0): 1, (7, 1): -1, (7, 2): -1, (7, 3): 1,
        }
        Q2 = {
            (0, 0): -1, (0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1, (0, 5): 1,
            (1, 0): 1, (1, 1): -1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1,
            #(2, 0): 1, (2, 1): 1, (2, 2): 1, (2, 3): 1, (2, 4): 1, (2, 5): 1,
            #(3, 0): 1, (3, 1): 1, (3, 2): 1, (3, 3): 1, (3, 4): 1, (3, 5): 1,
            #(4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1, (4, 5): 1,
            #(5, 0): 1, (5, 1): 1, (5, 2): 1, (5, 3): 1, (5, 4): 1, (5, 5): 1,
            #(6, 0): 1, (6, 1): 1, (6, 2): 1, (6, 3): 1, (6, 4): 1, (6, 5): 1,
            #(7, 0): 1, (7, 1): 1, (7, 2): 1, (7, 3): 1, (7, 4): 1, (7, 5): 1,
        }
        """Q = {('A','A'):   1,
             ('A','B'):  -1,
             ('B','A'):  -1,
             ('B','B'):   1,
             ('C','C'):   1,
             ('C','A'):   1,
             ('C','B'):   1,
             ('D','D'):   1,
             ('D','A'):   1,
             ('D','B'):   1,
             ('D','C'):   1}
         """

        hybrid    = LeapHybridSampler()
        simulator = neal.SimulatedAnnealingSampler()
        exact     = dimod.ExactSolver()
        solver    = QBSolv()
        #response = exact.sample_qubo(Q2)
        response  = exact.sample_qubo(
            Q2,
            #solver=simulator,
            #solver=simulator,
            #num_reads=1000,
            #postprocess='sampling'
        )

        print(response)
        print(response.info)
        print("samples=%s" % list(response.samples()))
        print("energies=%s" % list(response.data_vectors['energy']))
        #for sample in response[0:2]: print(sample)
        samples = np.array([[samp[k] for k in range(6)] for samp in response])
        print(samples)
        #print(response.data)

    def train2(self):
        from dwave.system import EmbeddingComposite, DWaveSampler
        # from dwave.system.samplers import LeapHybridSampler

        Q = {('A','A'):   1,
             ('A','B'):  -1,
             ('B','A'):  -1,
             ('B','B'):   1,
             ('C','C'):   1,
             ('C','A'):   1,
             ('C','B'):   1}

        # Define the sampler that will be used to run the problem
        ## """solver={'qpu': True})"""
        ## LeapHybridSampler
        sampler = EmbeddingComposite(DWaveSampler(token=os.getenv('DWAVE_API_KEY')))

        ## Quantum QPU Paramaters
        params = {
            'num_reads': 1000,
            #'auto_scale': True,
            #'answer_mode': 'histogram',
            #'num_spin_reversal_transforms': 10,
            #'annealing_time': 10,
            #'postprocess':'optimization',
            'postprocess': 'sampling',
        }

        # Run the problem on the sampler and print the results
        # postprocess='sampling' answer_mode='histogram',
        sampleset = sampler.sample_qubo(Q, **params)
        #sampleset = sampler.sample_ising({}, Q, **params)

        print("print(sampleset)")
        print(sampleset)

        print("print(sampleset.info)")
        print(sampleset.info)

        print("print(sampleset.data)")
        print(sampleset.data)

        print("for sample in sampleset: print(sample)")
        for sample in sampleset: print(sample)

        print("print(dir(sampleset))")
        print(dir(sampleset))

        #print("print(sampleset.to_pandas_dataframe())")
        #print(sampleset.to_pandas_dataframe())

        print("print(sampleset.to_serializable())")
        print(sampleset.to_serializable())

        #print("print(Q.to_numpy_matrix())")
        #print(Q.to_numpy_matrix())

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()
