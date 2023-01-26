## 1. Notations
- Let $\mathcal{R}\in\{\mathbb{R},\mathbb{Q},\mathbb{Z}\}$ 
- Let $\mathcal{G}=(\mathcal{V},\mathcal{E},\alpha)$ be a labeled digraph, with $\alpha: \mathcal{E}\rightarrow  \mathcal{R}$

## 2. Mean Pay Off game *(To verify)*
- We are interested in the games defined by the quadruplet $\mathtt{G}=(\mathcal{V},\mathcal{E},\alpha,v_0),$ with $v_0\in \mathcal{V}.$
- This game is a turn-based $2$-player game, defined as follow:
	1. The game starts at $s=v_0,$ with $R=0.$ 
	2. For a state $s\in\mathcal{V},$ the current player can choose a state $s'$ as a next state with $(s,s')\in\mathcal{E}.$ The valuation of such choice is:
		$$
		R\leftarrow R+\alpha(s,s')
		$$
	3. The first player want to maximize $R,$ while the second wants to minimize
	4. If no choice is possible, the game terminates


The objective is to find an approximate optimal solution for the first player.

## 3. Transforming finite games to an infinite ones
For each vertex $s$ that has no adjacent state, we can simply add ah edge $s\rightarrow s$, with $\alpha(s,s)=0.$
With that, every potentially finite game is transformed to an equivalent infinite one.--
It is equivalent in the sense that a strategy for one leads to a strategy for other, with the same valuations.


