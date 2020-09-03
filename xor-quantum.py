## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import ai
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum XOR Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumXOR(ai.NeuralNetwork):
    def initalize(self):
        density  = 4
        features = 3 ## change for training features!
        super().initalize(
            learn   =  0.02
        ,   epochs  =  2000
        ,   batch   =  density * features
        ,   bias    =  1
        ,   density =  4
        ,   high    =  2.0
        ,   low     = -2.0
        )

        self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        self.add(ai.QuantumSimulatorLayer, name='QuSim', activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    qnn = quantum()
    qx  = [x for x in range(len(qnn.loss))]

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Quantum XOR')

    ax1.set_ylabel('Quantum Error')
    ax1.set_yscale('log')
    ax1.plot(qx, qnn.loss, '.-')

    plt.show()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum Training
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def quantum():
    features = [[1,1],[0,0],[1,0],[0,1]]
    labels   = [ [1],  [1],  [0],  [0] ]

    def updates(epoch): print(epoch, qnn.loss_avg[-1], qnn.loss[-1])

    qnn = QuantumXOR()
    qnn.load(features=features, labels=labels)
    qnn.train(progress=updates)

    results = qnn.predict(features)

    print(np.column_stack((
        results
    ,   np.round(results)
    ,   np.where(results > 0.5, 1, 0)
    ,   np.array(labels)
    )))

    return qnn

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()
