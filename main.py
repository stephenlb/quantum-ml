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
#  Inputs    (batch × units) 
#   ▼
#  HiddenSig (units × units)
#   ▼
#  QUBO      (units × units)
#   ▼
#  QOut      (1 x batch)
#   ▼
#  QLinear   (batch x units)
#   ▼
#  Output    (1 x batch)
#  

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum XOR Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumXOR(ai.NeuralNetwork):
    def initalize(self):
        density  = 4
        features = 3
        super().initalize(
            learn   =  0.1
        ,   epochs  =  3000
        ,   batch   =  density * features
        ,   bias    =  1
        ,   density =  density
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
        density  = 4
        features = 3
        super().initalize(
            learn   =  0.1
        ,   epochs  =  3000
        ,   batch   =  density * features
        ,   bias    =  1
        ,   density =  density
        ,   high    =  2.0
        ,   low     = -2.0
        )

        self.add(ai.StandardLayer, name='Sigmoid', activation='sigmoid')
        self.add(ai.StandardLayer, name='Sigmoid', activation='sigmoid')
        self.add(ai.StandardLayer, name='Output',  activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    nn  = classical()
    qnn = quantum()
    cx  = [x for x in range(len(nn.loss))]
    qx  = [x for x in range(len(qnn.loss))]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Quantum ML vs. Classical ML')

    ax1.set_ylabel('Quantum Error')
    ax1.set_yscale('log')
    ax1.plot(qx, qnn.loss, '.-')
    ax1b = ax1.twinx()
    ax1b.axis(ymin=0, ymax=nn.loss_avg[200])
    ax1b.set_ylabel('mse', color='tab:red')
    ax1b.tick_params(axis='y', labelcolor='tab:red')
    ax1b.text(len(nn.loss)+6, nn.loss_avg[200]-nn.loss_avg[200]*.14, str(qnn.loss_avg[-1]), color='tab:red', horizontalalignment='right', verticalalignment='top')
    ax1b.plot(qx, qnn.loss_avg, '.-', color='tab:red')

    ax2.set_xlabel('Training Iteration (epoch)')
    ax2.set_ylabel('Classical Error')
    ax2.set_yscale('log')
    ax2.plot(cx, nn.loss, 'o-')
    ax2b = ax2.twinx()
    ax2b.axis(ymin=0, ymax=nn.loss_avg[200])
    ax2b.set_ylabel('mse', color='tab:red')
    ax2b.tick_params(axis='y', labelcolor='tab:red')
    ax2b.text(len(nn.loss)+6, nn.loss_avg[200]-nn.loss_avg[200]*.14, str(nn.loss_avg[-1]), color='tab:red', horizontalalignment='right', verticalalignment='top')
    ax2b.plot(cx, nn.loss_avg, '.-', color='tab:red')

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
