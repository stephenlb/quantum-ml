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
class QuantumMNIST(ai.NeuralNetwork):
    def initalize(self):
        super().initalize(
            learn   =  0.00001
        ,   epochs  =  100000
        ,   batch   =  10
        ,   bias    =  0.1
        ,   density =  1
        ,   high    =  2.0
        ,   low     = -2.0
        )

        self.add(ai.StandardLayer, name='Sigmoid', activation='sigmoid')
        self.add(ai.StandardLayer, name='Sigmoid', activation='sigmoid')
        #self.add(ai.StandardLayer, name='ElliotScaled',  activation='essigmoid')
        #self.add(ai.StandardLayer, name='ElliotSigmoid', activation='esigmoid')
        self.add(ai.StandardLayer, name='Output',        activation='linear')
        #self.add(ai.QuantumSimulatorLayer, name='QuSim', activation='linear')

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main():
    samples      = 20
    train_images = mnist.train_images()[0:samples]
    train_labels = mnist.train_labels()[0:samples]
    labels       = [[1 if l > 4 else 0] for l in train_labels]
    features     = vectorize(train_images)

    #print(train_images[5])
    #print(features)
    #for feature in features: print(feature)
    #return
    #print(train_labels[0],labels[0])
    #print(len(features[0]))
    #print(labels)
    #return

    def updates(epoch): print(epoch, nn.loss_avg[-1], nn.loss[-1])

    nn = QuantumMNIST()
    nn.load(features=features, labels=labels)
    nn.train(progress=updates, updates=100)
    results = nn.predict(features)

    #print(np.array(nn.loss))
    print(np.column_stack((
        results
    ,   np.round(results)
    ,   np.array(labels)
    )))

    x = [x for x in range(len(nn.loss))]
    plt.errorbar(x, nn.loss, yerr=nn.loss, errorevery=100)
    plt.yscale('log')
    plt.show()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Preprocess images before training
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def vectorize(images):
    #print(onehot(flatten(images), edge=1))
    #raise Exception("SDF")
    return onehot([
        downsample(image)
        for image in onehot(flatten(images), edge=1)
    ], edge=10)

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Flatten 2D -> 1D array
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def flatten(images):
    resolution = len(images[0]) * len(images[0][0])
    return [
        np.reshape(image, resolution)
        for image in images
    ]

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## OneHot with Edge Threshold
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def onehot(images, edge=1):
    return [[
        1 if pixel > edge else 0
        for pixel in image
    ] for image in images ]

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Downsample by Average
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def downsample(image, by=5):
    for a in range(by):
        image = [
            v + (image[p+1] if p+1 < len(image) else 0)
            for p, v in enumerate(image)
            if not p % 2
        ]
    return image

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run Main
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__": main()
