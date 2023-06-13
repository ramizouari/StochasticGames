
Broadly speaking, the pipeline will consist of the following steps:

## 1. Value Estimation

In this step, we will estimate the values of states using a variant of Monte Carlo Tree Search.

The selection policy will be based on a helper deep learning model that will estimate the value of a state.

Those estimates will control the behaviour of the selection process.

We will apply many estimations of the monte carlo simulations, potentially on a cluster of computers

## 2. Model Training
With the simulations done, we will build an estimation of the values of states. And we will fit the model $\mathcal{M}$ to these values.

## 3. Agent Evaluation
We will evaluate the agent against the older version of the model, and potentially against other opponents.

The acceptance of the recently fitted model is determined by its performance against these opponents.
