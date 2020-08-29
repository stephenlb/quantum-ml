## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import ai
import numpy as np

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

        self.add(ai.StandardLayer, name='ElliotScaled',  activation='essigmoid')
        self.add(ai.StandardLayer, name='ElliotSigmoid', activation='esigmoid')
        self.add(ai.StandardLayer, name='Output',        activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    features = [[1,1],[0,0],[1,0],[0,1]]
    labels   = [ [1],  [1],  [0],  [0] ]

    nn = ClassicalXOR()
    nn.load(features=features, labels=labels)
    nn.train(progress=lambda epoch : print(epoch, np.average(nn.loss)))
    results = nn.predict(features)

    print(np.column_stack((
        results
    ,   np.round(results)
    ,   np.array(labels)
    )))

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()
