
## 1. Ideas
- For each player, the optimal strategy depends only on the current state *(To prove)*

## 2. Equation
- Let $(f_n)_{n\in\mathbb{N}}$ be a family of valuations such that:
	$$
	\DeclareMathOperator{\Adj}{Adj}
	\DeclareMathOperator{\adj}{adj}
		\begin{cases}
	\displaystyle \forall u\in\mathcal{V} &  f_0(u)=0\\
		\displaystyle \forall n\in\mathbb{N}^*,\forall u\in\mathcal{V} &  f_n(u)=\displaystyle \begin{cases} 
		\displaystyle \max_{v\in\adj(u)}f_{n-1}(v)+\alpha(u,v)  & \text{if}\space n\space \text{is odd} \\
		\displaystyle  \min_{v\in\adj(u)}f_{n-1}(v)+\alpha(u,v) & \text{otherwise}
		\end{cases}
	\end{cases}
	
	$$
- we divide the sequence in two $(g_n)_{n\in\mathbb{N}}=(f_{2n})_{n\in\mathbb{N}}$ and $(h_n)_{n\in\mathbb{N}^*}=(f_{2n-1})_{n\in\mathbb{N}^*}$. We have:
	$$
	\begin{align*}
	\forall n\in\mathbb{N}^*, \forall u\in\mathcal{V},\quad g_n(u)&= \min_{v\in\adj(u)}h_{n}(v)+\alpha(u,v)\\
	&=\min_{v\in\adj(u)}\max_{w\in\adj(v)} g_{n-1}(w)+\alpha(u,v)+\alpha(v,w)\\
	\forall n\in\mathbb{N}^*, \forall u\in\mathcal{V},\quad h_n(u)&=\max_{v\in\adj(u)}g_{n-1}(v)+\alpha(u,v)\\
	&=\max_{v\in\adj(u)}\min_{w\in\adj(v)} h_{n-1}(w)+\alpha(u,v)+\alpha(v,w)
	\end{align*}
	$$

## 3. Fixed Point algorithm
- Let $\bar{\mathcal{R}} = \mathcal{R} \cup \{\pm\infty\}$ 
- We will apply the following algorithm, starting from $g(u)=0\quad \forall u\in\mathcal{V}$:
$$
\begin{align}
\text{1.}\quad &\forall u\in\mathcal{V},\quad h(u)\leftarrow \max_{v\in\adj(u)}g(v)+\alpha(u,v)\\
\text{2.}\quad&\forall u\in\mathcal{V},\quad g(u)\leftarrow \min_{v\in\adj(u)}h(v)+\alpha(u,v)\\
\end{align}
$$

Both sequences converge on $\mathscr{F}(\mathcal{V},\bar{\mathcal{R}})$  to $(g^*,h^*)$ (with a suitable metric, like $d(f,g)= \max_{u\in\mathcal{V}}\lvert \tanh f(u)-\tanh g(u)\rvert$), then: 
- $g^*(u)$ will be the valuation of a game that starts at $u\in\mathcal{V}$ on which the second player plays first.
- $h^*(u)$ will be the valuation of a game that starts at $u\in\mathcal{V}$ on which the first player plays first.

## 4. Independent Fixed Point algorithm
This algorithm will calculate $g$ and $h$ independently. 

- Let $\bar{\mathcal{R}} = \mathcal{R} \cup \{\pm\infty\}$ 
- We will apply the following algorithm, starting from $g(u)=h(u)=0\quad \forall u\in\mathcal{V}$:
$$
\begin{align}
\text{1.}\quad &\forall u\in\mathcal{V},\quad h(u)\leftarrow \max_{v\in\adj(u)}\min_{w\in\adj(v)} h(w)+\alpha(u,v)+\alpha(v,w)\\
\text{2.}\quad&\forall u\in\mathcal{V},\quad g(u)\leftarrow \min_{v\in\adj(u)}\max_{w\in\adj(v)} g(w)+\alpha(u,v)+\alpha(v,w)\\
\end{align}
$$

Both sequences converge on $\mathscr{F}(\mathcal{V},\bar{\mathcal{R}})$  to $(g^*,h^*)$ (with a suitable metric, like $d(f,g)= \max_{u\in\mathcal{V}}\lvert \tanh f(u)-\tanh g(u)\rvert$), then: 
- $g^*(u)$ will be the valuation of a game that starts at $u\in\mathcal{V}$ on which the second player plays first.
- $h^*(u)$ will be the valuation of a game that starts at $u\in\mathcal{V}$ on which the first player plays first.
## 5. Tests
To test our algorithms, we generated test files with the given format:
```bash
|V| |E|
u1 v1 l1
u2 v2 l2
...
...
...
um vm lm
```
with:
- $n=\lvert \mathcal{V}\rvert$
- $m=\lvert \mathcal{E} \rvert$
- $\mathcal{V}=\{0,\dots,n-1\}$
- $\mathcal{E}=\{(u_1,v_1),\dots,(u_m,v_m)\}$
- $L:(u_i,v_i)\rightarrow l_i$

