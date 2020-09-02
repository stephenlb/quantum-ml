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
#  In (batch × units) 
#   ▼
#  HiddenSig (units × units)
#   ▼
#  QUBO (units × units)
#   ▼
#  QOut (1 x batch)
#   ▼
#  QLinear (batch x units)
#   ▼
#  HiddenSig (units x units)
#   ▼
#  Output (1 x batch)
#  
#  

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Quantum XOR Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class QuantumXOR(ai.NeuralNetwork):
    def initalize(self):
        super().initalize(
            learn   =  0.02
        ,   epochs  =  2000
        ,   batch   =  12 # must = density * features
        ,   bias    =  1
        ,   density =  4
        ,   high    =  2.0
        ,   low     = -2.0
        )

        #self.add(ai.StandardLayer, name='ElliotScaled',  activation='essigmoid')
        #self.add(ai.StandardLayer, name='ElliotSigmoid', activation='esigmoid')
        #self.add(ai.StandardLayer, name='ElliotSigmoid', activation='esigmoid')
        self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        #self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        self.add(ai.QuantumSimulatorLayer, name='QuSim', activation='linear')
        #self.add(ai.StandardLayer, name='Output',        activation='linear')#, shape=[1, None])
        #self.add(ai.StandardLayer, name='Sigmoid',       activation='sigmoid')
        #self.add(ai.QuantumOutputLayer, name='QuOut', activation='sigmoid', shape=[1, None])
        #self.add(ai.StandardLayer, name='ElliotSSig',    activation='essigmoid')
        #self.add(ai.StandardLayer, name='Output',        activation='linear')#, shape=[1, None])
        #self.add(ai.StandardLayer, name='ElliotSSig',    activation='essigmoid')#, shape=[1,None,1])
        #self.add(ai.QuantumHardwareLayer, name='QuHard', activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    features = [[1,1],[0,0],[1,0],[0,1]]
    labels   = [ [1],  [1],  [0],  [0] ]

    def updates(epoch): print(epoch, nn.loss_avg[-1], nn.loss[-1])

    nn = QuantumXOR()
    nn.load(features=features, labels=labels)
    nn.train(progress=updates, updates=100)
    print("")
    print("PREDICTING..........")
    print("")
    results = nn.predict(features)

    print(np.column_stack((
    #print(((
        results
    ,   np.round(results)
    ,   np.where(results > 0.5, 1, 0)
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
