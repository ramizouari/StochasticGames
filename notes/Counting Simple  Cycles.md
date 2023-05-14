## 1. Definition
$$
\DeclareMathOperator{\RotationallyEqual}{\mathcal{R}}
$$
Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ be a di-graph

A simple cyclic sequence of $\mathcal{G}$ is a tuple $(u_0,\dots,u_m)\in \mathcal{V}^*$  with:
- $\forall i\in\{1,\dots,m\},\quad (u_{i-1},u_{i})\in \mathcal{E}$ 
- $\forall i,j\in\{0,\dots,m\},\quad u_i=u_j\implies \{i,j\}\subseteq \{0,m\}$

A rotation of a simple cyclic sequence $S=(u_0,\dots,u_m)$ is a simple cyclic sequence $S'=(v_0,\dots,v_m)$ such that:
$$
 \exists r\in \{1,\dots,m\}/\quad (u_0,\dots,u_m)=(v_r,\dots,v_m,v_1,\dots,v_{r})
$$


Two simple cyclic sequences $S$ and $S'$ are said to be rotationally equal, if one is a rotation of the other.
We will denote this relation by $S \RotationallyEqual S'$

Now it is trivial that $\mathcal{R}$ is an equivalence relation

We will define a simple cycle as an element the quotient set $\mathcal{S}/\RotationallyEqual,$ where $\mathcal{S}\subset \mathcal{V}^*$ is the set of simple cyclic sequences.


## 2. Theorem
- Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ be a finite di-graph.
- Let $\mathscr{C}(\mathcal{G})$ the number of simple cycles in $\mathcal{G}$

We will prove that:
$$
\mathscr{C}(\mathcal{G})\le e\cdot \lvert \mathcal{V} \rvert!
$$

## 3. Proof
- Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ be a graph with $n=\lvert \mathcal{V} \rvert$ vertices
- Let $\mathcal{K}_n$ be the complete di-graph with $n$ vertices, containing loops

Let $C$ be a simple cycle of $\mathcal{G},$ then $C$ is also a simple cycle of $\mathcal{K}_n.$ And with that:
$$
\mathscr{C}(\mathcal{G})\le \mathscr{C}(\mathcal{K}_n)
$$


Now, it is easy to enumerate all simple cycles of $\mathcal{K}_n$:
1. We choose $k$ vertices from $\mathcal{V}$
2. Each cyclic permutation of those $k$ vertices determine a unique simple cycle of $\mathcal{K}_n$
It is easy to see that this method will enumerate all simple cycles of $\mathcal{K}_n$.

With that said:
$$
\begin{align*}
\mathscr{C}(\mathcal{K}_n)&=\sum_{k=1}^n {n \choose k}\times (k-1)! \\
&= \sum_{k=1}^n \frac{n!(k-1)!}{k!(n-k)!} \\
&=\sum_{k=1}^n \frac{n!}{k(n-k)!}\\
\implies & e\cdot (n-1)!\le \mathscr{C}(\mathcal{K}_n)\le e \cdot n! \quad  \blacksquare
\end{align*}
$$