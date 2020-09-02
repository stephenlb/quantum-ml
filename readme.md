# Quantum Machine Learning vs Classical

This is a learn repostiory for exploring.
Code is for exploration, not production.

> Goal: Build a QSVM (Quantum Support Vector Machine) where D-Wave QPU provides
> weights to be used in classical SVM prediction models.

## Initial Results are Promising

Quantum ML converges more quickly than classical.
With fewer training cycles, this can be considered a speedup.

The following figures are two training sessions comparing 
Quantum Machine Learning with Classical Machine Learning.
The images are represented using the same hyper parameters.

![Figure 1](media/quantum-ml-vs-classical-ml-1.png)
![Figure 2](media/quantum-ml-vs-classical-ml-2.png)

As you can see, the QSVM outperforms the Classical SVM.
This happens most of the time.
The RNG nature of initial weights may change the outcome.
Try it yourself!

### Run with Python Locally

Running on your local machine, use these following commands.

```shell
pip install -r requirements.txt
python xor-quantum.py
```

### Run with Docker

```shell
docker build . -t quantum-ml
docker run -e DWAVE_API_KEY=YOUR_API_KEY quantum-ml
```

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
In addtion, deep learning with multi-layer neural nets.

## Challenges with Quantum Machine Learning

The Quantum chip (QPU) is programmable using assembly-like instructions.
Making the task that much more tricky.
