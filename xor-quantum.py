## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import ai
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum SVM Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
##  ◀ ▶ ▲ ▼ ×
## ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽ ⎽
#  
#  In        (batch × units) 
#   ▼
#  HiddenSig (units × units)
#   ▼
#  QUBO      (units × units)
#   ▼
#  QOut      (1 x batch)
#   ▼
#  QLinear   (batch x units)
#   ▼
#  HiddenSig (units x units)
#   ▼
#  Output    (1 x batch)
#  

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
## Classical XOR Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class ClassicalXOR(ai.NeuralNetwork):
    def initalize(self):
        super().initalize(
            learn   =  0.02
        ,   epochs  =  2000
        ,   batch   =  10
        ,   bias    =  1
        ,   density =  6
        ,   high    =  2.0
        ,   low     = -2.0
        )

        #self.add(ai.StandardLayer, name='ElliotScaled',  activation='essigmoid')
        #self.add(ai.StandardLayer, name='ElliotSigmoid', activation='esigmoid')
        self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        self.add(ai.StandardLayer, name='Output',        activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    nn  = classical()
    qnn = quantum()
    cx   = [x for x in range(len(nn.loss))]
    qx   = [x for x in range(len(qnn.loss))]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Quantum ML vs. Classical ML')

    ax1.set_ylabel('Quantum Error')
    ax1.set_yscale('log')
    ax1.plot(qx, qnn.loss, '.-')

    ax2.set_xlabel('Training Iteration (epoch)')
    ax2.set_ylabel('Classical Error')
    ax2.set_yscale('log')
    ax2.plot(cx, nn.loss, 'o-')

    plt.show()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Classical Training
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def classical():
    features = [[1,1],[0,0],[1,0],[0,1]]
    labels   = [ [1],  [1],  [0],  [0] ]

    def updates(epoch): print(epoch, nn.loss_avg[-1], nn.loss[-1])

    nn = ClassicalXOR()
    nn.load(features=features, labels=labels)
    nn.train(progress=updates)

    results = nn.predict(features)

    print(np.column_stack((
        results
    ,   np.round(results)
    ,   np.array(labels)
    )))

    return nn

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
