
## 1. Chapters
The report will contain the following chapters:
### 1.0 Introduction


### 1.1 Analysis of Mean Payoff Games
- Definition
- How the game is well defined
- Symmetries of the game
- Why standard Game Theory algorithms does not apply (MinMax,etc..)
- Strategies
	- Deterministic (Calculating Winner)
	- Fractional (Calculating Winner)
- Knowing the Winner based on pair of (deterministic/fractional) strategies
- Solving the game with a given strategy of a player.

### 1.2 Mean Payoff Games Library


### 1.3 Solving a Mean Payoff Games
- Why standard Game Theory algorithms does not apply (MinMax,etc..)
- Slow convergence of Known algorithms (Policy Iteration, Value Iteration)
- 

### 1.4 Generation of Mean Payoff Games
- Generating a Graph
- Fast Implementation of generating a graph
- Implementation and Design
- Deployment on HPC node.
- Solving Mean Payoff Instances using CSP.
- Limits of the CSP approach
- Generating MPG dataset: sparse/dense
- Annotating the dataset using the CSP solver, using adequate heuristics
- Deployment on a HPC Cluster


### 1.5 Model Design
- Feature Extraction: Adjacency Matrix, Weights Matrix, starting position
- Preprocessing Layers:
	- Random Connections
	- Weight Noise
	- Permutation Layer
	- Weight Normalisation
- Problem with Multi Layer Perceptrons
- Graph Neural Network Approach
- Adding Weighted Graph Convolutions
- Model Properties
	- Equivariance under permutation
	- Stability under Padding

### 1.6 Pipeline Architecture
- Supervised Training pipeline
- Problem with the supervised approach
- Alpha Zero approach.
- Problem with open_spiel's implementation of Alpha Zero
- Implementation


### 1.7 Training & Results
- Hyper-parameters
- Training Options
- Monitoring Training
- Results

