## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Imports
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import ai
import numpy as np
import matplotlib.pyplot as plt
import mnist

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Classical MNIST Model
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class ClassicalMNIST(ai.NeuralNetwork):
    def initalize(self):
        super().initalize(
            learn   =  0.0002
        ,   epochs  =  1000
        ,   batch   =  10
        ,   bias    =  0.1
        ,   density =  2
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
    samples      = 20
    train_images = mnist.train_images()[0:samples]
    train_labels = mnist.train_labels()[0:samples]
    resolution   = len(train_images[0]) * len(train_images[0])
    labels       = [[1 if l > 4 else 0] for l in train_labels]
    features     = [
        np.where(np.reshape(symbol, resolution) > 1, 1, 0)
        for symbol in train_images
    ]

    nn = ClassicalMNIST()
    nn.load(features=features, labels=labels)
    nn.train(progress=lambda epoch : print(epoch, np.average(nn.loss)))
    results = nn.predict(features)

    print(features[0])
    print(len(features[0]))
    print(labels)
    print(np.array(nn.loss))
    print(np.column_stack((
        results
    ,   np.round(results)
    ,   np.array(labels)
    )))

    #x = [x for x in range(len(nn.loss))]
    #plt.errorbar(x, nn.loss, yerr=nn.loss, errorevery=100)
    #plt.yscale('log')
    #plt.show()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()
