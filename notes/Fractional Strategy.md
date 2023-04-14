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
- Let $A,B$ be a pair of fractional strategies
- Let $P,Q$ two random variables defining the mean-payoffs for the respective players

We have:
$$
\begin{align*}
\mathbb{E}[P^{k}_i]&=\sum_{j}A_{i,j}\mathbb{E}[P_j^{k+1}]+\sum_{j}A_{i,j}W_{i,j}\\
\mathbb{E}[P^k]&=A\cdot\mathbb{E}[P^{k+1}]+A\odot W\\
\mathbb{E}[P^0]&= \sum_{r=0}^k
\end{align*}
$$
$$
\begin{align*}
\mathbb{E}[P^{k+1}_j]&=\sum_{i}A_{i,j}\mathbb{E}[P_i^{k}]+\sum_{i}A_{i,j}W_{i,j}\\
\mathbb{E}[P^k]&=A^T\cdot\mathbb{E}[P^{k+1}]+A\odot W\\
\mathbb{E}[P^k]&= \sum_{r=0}^{k-1}\left(A^T\right)^r \cdot (A\odot W)+\left(A^T\right)^k \mathbb{E}[P^0] \\
\frac{1}{k}\mathbb{E}[P^k]&=\frac{1}{k}\sum_{r=0}^{k-1}\left(A^T\right)^r \cdot (A\odot W)+ \frac{1}{k}\left(A^T\right)^k \mathbb{E}[P^0]
\end{align*}
$$

As $A$ is a markov transistion matrix, then it has a simple eigenvalue of $1$, and all other complex eigenvalues are strictly less than $1$ in norm.

Also, let $DJD^{-1}$ be the Jordan normal form of $A$ with:
$$
J=\begin{pmatrix}D_s & 0 & \dots &0  \\
0& J_2 & \dots  &0\\
0 & 0 & \ddots &0 \\
0 &0 & \dots& J_{n}
\end{pmatrix}
$$

With $D_s$ a diagonal matrix of size $s$, with elements $\pm 1,$ $J_2,\dots,J_n$ are jordan blocks with associated eigenvalues $\lambda_2,\dots,\lambda_n$ such that:
$$
\max_{i\in\{2,\dots,n\}}\lvert \lambda_i \rvert <1
$$
With that:
$$
\forall i\in\{2,\dots,n\},\quad \lim_{r\rightarrow +\infty}\frac{1}{r}J_i^r=0
$$
With that, it can be proven that:
$$
\lim_{r\rightarrow \infty} \frac{1}{r}\left(A^T\right)^r=0
$$

Also:
$$
\begin{align*}
\sum_{k=0}^{r-1}(D_s)^k&=\sum_{k=0}^{r-1}D_{s,+}^k+D_{s,-}^k \\
&=\sum_{k=0}^{r-1}D_{s,+}+(-1)^kD_{s,-} \\
&=rD_{s,+}+\frac{1}{2}\cdot(1-(-1)^r)D_{s,-} \\
\lim_{r\rightarrow +\infty}\sum_{k=0}^{r-1}(D_s)^k&=\lim_{r\rightarrow +\infty} D_{s,+}+\frac{1}{2r}\cdot(1-(-1)^r)D_{s,-} \\
&=D_{s,+}
\end{align*}
$$
With that:
$$
\lim_{r\rightarrow +\infty}\frac{1}{r}\mathbb{E}[P^r]= B\cdot(A\odot W)
$$
With:
$$
B=DJ^{\star}D^{-1}=D\begin{pmatrix}D_{s,+} & 0 & \dots &0  \\
0& 0 & \dots  &0\\
0 & 0 & \ddots &0 \\
0 &0 & \dots& 0
\end{pmatrix} D^{-1}
$$
