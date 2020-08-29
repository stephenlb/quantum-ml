## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import ai
import numpy as np
import matplotlib.pyplot as plt

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum XOR Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumXOR(ai.NeuralNetwork):
    def initalize(self):
        super().initalize(
            learn   =  0.00001
        ,   epochs  =  1200
        ,   batch   =  9 # must = density * features
        ,   bias    =  1
        ,   density =  3
        ,   high    =  1.0
        ,   low     = -1.0
        )

        self.add(ai.StandardLayer, name='ElliotScaled',  activation='essigmoid')
        self.add(ai.QuantumSimulatorLayer, name='QuSim', activation='linear')
        self.add(ai.StandardLayer, name='ElliotSigmoid', activation='esigmoid')
        self.add(ai.StandardLayer, name='ElliotSigmoid', activation='sigmoid')
        self.add(ai.StandardLayer, name='Output',        activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    features = [[1,1],[0,0],[1,0],[0,1]]
    labels   = [ [1],  [1],  [0],  [0] ]

    nn = QuantumXOR()
    nn.load(features=features, labels=labels)
    nn.train(
        progress=lambda epoch : print(epoch, np.average(nn.loss), nn.loss[-1])
    )
    results = nn.predict(features)

    print(np.column_stack((
        results
    ,   np.round(results)
    ,   np.array(labels)
    )))

    x = [x for x in range(len(nn.loss))]
    plt.plot(x, nn.loss, 'bo')
    plt.yscale('log')
    plt.show()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()
