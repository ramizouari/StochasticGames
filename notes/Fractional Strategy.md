$$
\DeclareMathOperator{\Adj}{Adj}
$$
## 1. Definition
- Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ be a mean-payoff game
- For $u\in\mathcal{V},$ Let $\mathscr{P}(u)$ be the set of probability distributions over the set $\text{Adj}(u)$
- We define a fractional strategy as a function $\Phi\in \mathscr{P}$

## 2. As a matrix
- Let $n=\lvert \mathcal{V}\rvert$
- Let $u_1,\dots,u_n$ an enumeration of elements of $\mathcal{V}$
A fractional strategy can be represented as a matrix $A$ such that:
$$
\mathcal{P}(\Phi(u_i)=u_j)=A_{i,j}
$$


## 3. Mean Payoff of a pair of fractional strategies
### 3.1 Notations
- Let $A,B$ be a pair of fractional strategies
- Let $P_m,Q_m$ two random variables defining the mean-payoffs for the respective players after turn $m$


### 3.2 Expected Cost of a pli
Let $\Pi \in \{A,B\}$

We have:
$$
\begin{align*}
\mathbb{E}\left[w(u,\Pi(u))\right]&=\sum_{v\in \Adj u} w(u,v)\cdot \mathcal{P}(\Pi(u)=v) \\
\end{align*}
$$

### 3.3 Expected cost of a turn
Let $h$ be the cost of a turn

We have:
$$
\begin{align*}
\mathbb{E}\left[h(u,A(u),B \circ A(u))\right]&= \mathbb{E}[w(u,A(u))]+\sum_{v\in \Adj u} \mathbb{E}[w(v,B(v))]\cdot \mathcal{P}(A(u)=v) \\
\end{align*}
$$

### 3.4 Expected Total Payoff
- Let $\Pi=B\circ A$
- Let $(X_m)_{m\in\mathbb{N}}$ defined as follow:
	$$
	\begin{cases}
	X_0&=s\\
	\forall m\in\mathbb{N}^*,\quad X_m&= \Pi(X_{m-1})
	\end{cases}
	$$
- Let $(R_m)_{m\in\mathbb{N}}$ defined as follow:
	$$
	\begin{cases}
	R_0&=0\\
	\forall m\in\mathbb{N}^*,\quad  R_m&= R_{m-1}+\displaystyle\sum_{u\in V}h(u,A(u),\Pi(u)) \cdot \mathcal{P}(X_{m-1}=u)
	\end{cases}
	$$
We have:
$$
\begin{align*}
\mathbb{E}[R_m]&= \mathbb{E}[R_{m-1}]+\sum_{u\in V}\mathbb{E}[h(u,A(u),\Pi(u))] \cdot \mathcal{P}(X_{m-1}=u) \\
&=\mathbb{E}[R_{m-1}]+\sum_{u\in V}P^{m-1}(s,u)\times q(u)\\ 
&=\mathbb{E}[R_{m-1}]+(P^{m-1}\cdot q)(s)\quad \text{(Matrix Multiplication)} \\
&=\sum_{k=1}^m(P^{k-1}\cdot q)(s)+ \mathbb{E}[R_0]  \\
&=\left(\sum_{k=0}^{m-1}P^{k}\cdot q\right)(s)+ \mathbb{E}[R_0] \\
&=\left(\sum_{k=0}^{m-1}P^{k}\cdot q\right)(s)
\end{align*}
$$
Now, we may see that the formula is easy generalisable to any starting vertex:
$$
\mathbb{E}[R_m]=\sum_{k=0}^{m-1}P^{k}\cdot q
$$

### 3.5 Expected Mean Payoff
The mean-payoff is defined as:
$$
K_m=\frac{R_m}{m}
$$
We define $K_\infty$ as:
$$
K_\infty=K_{+\infty}=\lim_{m \rightarrow +\infty} \frac{R_m}{m}
$$

Let $m\in\mathbb{N}\cup\{+\infty\}$
Now, the expected mean-payoff can act as a the judge for who is winning:
- Player $0$ wins if $\mathbb{E}[K_m]>0$
- Player $1$ wins if $\mathbb{E}[K_m]<0$
- Else, it is a tie

Now, if $m$ is finite, we can calculate $\mathbb{E}[K_m]$ directly.
Otherwise, we have:
$$
\mathbb{E}[K_{\infty}]=\lim_{m\rightarrow +\infty}\frac{1}{m} \sum_{k=0}^m P^k \cdot q
$$
Now, $P$ can be seen as a stochastic matrix.
Thus it has a simple eigenvalue of value $1$, and all its other eigenvalue $\lambda$ satisfies:
$$
\lambda \neq 1 \wedge \lvert\lambda \rvert \le 1
$$
Also, we have (Proof?):
$$
\lvert \lambda \rvert =1\implies \lambda \space\text{is a simple eigenvalue}
$$

With that, it can be proven that the $\lim_{m\rightarrow +\infty}\frac{1}{m} \sum_{k=0}^m P^k$ converges so some matrix $T$

This matrix can be constructed as follow.
Let $P=VJV^{-1}$ the jordan normal form of $P$

Without a loss of generality, we will suppose that the first eigenvalue of this decomposition is $1$
We have then:
$$
T=V\begin{pmatrix}1 & 0\\
0&\boldsymbol{0}_{n-1}\end{pmatrix}V^{-1}
$$


## 4. Discounted Payoffs
Using the same approach as the mean payoffs. Let $R_m$ be the discounted payoff.
It can be shown that:
$$
\mathbb{E}[R_m]=\sum_{n\in\mathbb{N}} \gamma^nP^n \cdot q=(\text{Id}-\gamma P)^{-1}q
$$