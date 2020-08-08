# Quantum Machine Learning

This is a learn repostiory for exploring.
Code is for exploration, not production.

> Goal: Build a QSVM (Quantum Support Vector Machine) where D-Wave QPU provides
> weights to be used in classical SVM prediction models.

```shell
docker build . -t quantum-ml
```

```shell
docker run quantum-ml -e DWAVE_API_KEY=YOUR_API_KEY
```

Looking to take a conventional Machine Learning Algorithm
with SGD (stochastic gradient decent) on an SVM (support vector machine)
and apply it in a quantum circuit.

It seems that using a quadratic unconstrained binary optimizer for
finding weight outputs can generate usable AI weights that can be
applied in classical computation.

There are many approaches to using Quantum circuits as a solver.
Quantum Machine Learning appears solvable with a BQM / QUBO.

We need to generate a shallow neural net ( one layer ) matrix weights
to be used in predicting outcomes.
The next step will be deep learning with multi-layer neural nets.
To get started though, single layer weights w/ embedded bias is sufficient.

## Challenges with Quantum Machine Learning

The Quantum chip (QPU) is programmable using assembly-like instructions.
Making the task that much more tricky.
