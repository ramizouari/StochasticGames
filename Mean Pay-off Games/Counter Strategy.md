$$
\DeclareMathOperator{\adj}{adj}
$$
## 1. Ideas
- First of all, an optimal strategy for a player does only depend on the current state. *(To prove)*
- So for estimating performance, we will only consider strategies that depend on the current state.

## 2. Equation
- We will augment the game to an equivalent infinite one.
- Let $S:\mathcal{V}\rightarrow \mathcal{V}$  be the strategy of the first player, which sends the next state depending of the current state.
- Formally, $S\subseteq \mathcal{E}$
- Let $(f_n)_{n\in\mathbb{N}}$ a family of valuations such that:
	$$
	\begin{cases}
	\displaystyle \forall u\in\mathcal{V} &  f_0(u)=0\\
		\displaystyle \forall n\in\mathbb{N}^*,\forall u\in\mathcal{V} &  f_n(u)=\displaystyle \min_{v\in\adj(S(u))}f_{n-1}(v)+\alpha(u,S(u))+\alpha(S(u),v)
	\end{cases}
	$$

## 3. Fixed Point algorithm
- Let $\bar{\mathcal{R}} = \mathcal{R} \cup \{\pm\infty\}$ 
- We will apply the following algorithm, starting from $f(u)=0\quad \forall u\in\mathcal{V}$:
$$
\forall u\in\mathcal{V},\quad f(u)\leftarrow \min_{v\in\adj(S(u))}f(v)+\alpha(u,S(u))+\alpha(S(u),v)
$$

If it converges on $\mathscr{F}(\mathcal{V},\bar{\mathcal{R}})$  to $f^*$ (with a suitable metric, like $d(f,g)= \max_{u\in\mathcal{V}}\lvert \tanh f(u)-\tanh f(u)\rvert$), then $f^*(u)$ will be the valuation of a game that starts at $u\in\mathcal{V}$ using the first player's agreed on strategy, and the second player's optimal counter-strategy.

## 4. Matrix Formulation
### 4.1 Arithmetic
We will use a matrix multiplication consistent with our ring $\bar{\mathcal{R}}$
That is:
$$
\forall M,N\in\mathcal{M}_{\mathcal{V}}(\bar{\mathcal{R}}),\forall u,v \in\mathcal{V},\quad (M\odot N)_{u,v} = \min_{w\in\mathcal{V}} M_{u,w}+N_{w,v}
$$

### 4.2 Equation
We will create a matrix $M$ with:
$$
\forall u,v\in\mathcal{V},\quad  M_{u,v}= \begin{cases}
\alpha(u,S(u))+\alpha(S(u),v) & \text{if}\space v\in \adj(S(u)) \\
+\infty &\text{otherwise}
\end{cases}
$$
Also, the valuation $(f_n)$ can be thought as of a matrix.
With that, the algorithm will be **(To Verify)**:
$$
f_n=M \odot f_{n-1}
$$

With that, we will apply the following fixed point algorithm:
$$
f\leftarrow M\odot f
$$

### 4.3 Fast Method
We can apply an exponential search:
- If $M\odot f = (M\odot M)\odot f,$ then returns $M\odot f$
- Else, sets $M\leftarrow M\odot M$
In case of convergence, the value $f^*$  should be the same as that of $(3.)$


## 5. Tests
To test our algorithms, we generated test files with the given format:
```bash
|V| |E|
u1 v1
u2 v2
...
...
...
um vm
s1 s2 ... sn
```
with:
- $n=\lvert \mathcal{V}\rvert$
- $m=\lvert \mathcal{E} \rvert$
- $\mathcal{V}=\{0,\dots,n-1\}$
- $\mathcal{E}=\{(u_1,v_1),\dots,(u_m,v_m)\}$
- $S:i\rightarrow s_i$

