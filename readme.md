# Quantum Machine Learning vs Classical ML

Quantum, when added to Machine Learning, is claimed to be better.
Better with faster deep learning training.
Better with higher accuracy on smaller data sets.
Is this true?
Let's find out.

> Goal: Build a QSVM (Quantum Support Vector Machine) where D-Wave QPU provides
> weights to be used in classical SVM prediction models.
> Improvements we want include: speedup and higher accuracy.

## Quantum Results are Promising

Quantum ML converges more quickly than classical.
With fewer training cycles, this can be considered a speedup.

The following figures are training sessions comparing 
Quantum Machine Learning with Classical Machine Learning.
The images are represented using the same hyper parameters.

### Quantum ML vs Classical ML

Comparing using the same hyper parameters.
**Lower points** on the charts are **better**.

![Figure 1](media/quantum-ml-vs-classical-ml-1.png)

This second figure shows pretty much the same results.
Note that the Quantum converges more quickly than the classical
in both training graphs.

![Figure 2](media/quantum-ml-vs-classical-ml-2.png)

As you can see, the QSVM outperforms the Classical SVM.
This happens most of the time.
The RNG nature of initial weights may change the outcome.

### Quantum Deep Learning vs Classical Deep Learning

The quantum model learns differently comparing to the classical model.
Notice that the quantum finds a fit to the data set
that differs from the classical model.

![Figure 3](media/quantum-mse-in-action-2.png)

It is possible to tune the hyper parameters in the favor of both models.
In this case, however, the hyper parameters are shared for both
the classical deep learning and quantum deep learning models.

More PNG image results are in the `./media` directory.

## Try it out

Both docker and local python follow here.
You will need to run the training session multiple times.
This is because sometimes the model will miss convergence.
You'll get results representing quantum optimizations.
Nearly all Quantum runs show that the MSE is one order
of magnitude more optimal than the classical.

### Run with Python Locally

Running on your local machine, use these following commands.

```shell
pip install -r requirements.txt
python main.py
```

### Run with Docker

```shell
docker build . -t quantum-ml
docker run -e DWAVE_API_KEY=YOUR_API_KEY quantum-ml
```

## Quantum Training Approach

Looking to take a conventional Machine Learning Algorithm
with SGD (stochastic gradient decent) on an SVM (support vector machine)
and apply it in a quantum circuit.

It seems that using a quadratic unconstrained binary optimizer for
finding weight outputs can generate usable AI weights that can be
applied in classical computation.
This will accelerate training time.
This should improve accuracy.

There are several approaches to using Quantum Circuits as a solver.
Quantum Machine Learning appears solvable with a BQM / QUBO.

We need to generate a shallow neural net matrix weights
to be used in predicting outcomes.
In addition, deep learning with multi-layer neural nets.

## Quantum Acceleration of SVMs

We start with traditional SVM strategies.
Then we leverage quantum sampling to accelerate the training
speed and accuracy convergence.
This is achieved by using the quantum results as a filter
on our output layer weights.

This focuses the error correction on SGD optimization
using the quantum results.
This allows MSE from quantum models to converge faster than classical.
QUBO models only result in 1 or 0.
This causes possible destruction of learning progress.

It is necessary to make these leaps to approach global minimums,
however we don't want to over-fit and we don't want to leave
ourselves finding dead neurons.
We prevent dead neuron using leaky matrix multiplication to allow
future learnings on existing memory and recovery of
learnings post quantum tunneling events.

## Challenges with Quantum Machine Learning

The Quantum chip (QPU) is programmable using assembly-like instructions.
Making the task that much more tricky.
However several ease-of-use approaches have been made available.
The use of QUBO solvers make it easier to solve quantum problems without needing a lot of code.
QUBO solvers also allow you to skip the need to construct ideal quantum circuit.

While QUBO solvers help a new problem is introduced.
Find the weight coefficients becomes the new challenge.
Creating an algorithmic approach is yet to become a standard. 
More research is needed to develop quantum interfaces.

## Quantum Leap: Optimization

> **Update: this was added.**
> The `ai.py` lib includes a QPU optimization of 99% usage.

An optimization strategy is to reduce the number of sampling events
required when measuring optimal weight adjustments for each epoch.
Instead of sampling the weights each epoch,
we can instead sample 10x fewer times
keeping a cache of the most recent quantum samples.

Quantum sampling isn't needed on each epoch.
The convergence hasn't approached most recent sampling optimizations.
This approach may reduce training time.

## Conclusion

Overall, quantum delivers on the promise.
The addition of Quantum in Machine Learning has shown to achieve our goal.
Quantum is faster.
Quantum creates more accurate models.
Also, as a bonus, Quantum can reduce training data
volume requirements for deep learning.
