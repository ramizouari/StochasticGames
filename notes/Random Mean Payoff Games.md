## 1. Approach

## 2. Choosing $k$ elements from $S$ without replacement
### 2.1 Algorithm
We will propose an efficient algorithm to choose $m$ elements from a set $S$ **uniformly** and without replacement.
Let $n=\lvert S \rvert$ 
```python
def choose(S:list,m:int) -> set:
	n=len(A)
	C=set()
	while len(C) < m:
		k=np.random.integers(0,n)
		C.add(A[k])
	return C
```

For $S=\{0,\dots,n-1\}$ we propose this version:
```python
def choose_integers(n:int,m:int) -> set:
	while len(C) < m:
		k=np.random.integers(0,n)
		C.add(k)
	return C
```

### 2.2 Expected Runtime
We will estimate the running time of both approaches.
Let $X_{n,m}$ the running time of an execution of the algorithm `choose_integers(n,m)`.
We have:
$$
\begin{align*}
X_{n,0} & \space \text{is deterministic}\\
X_{n,0}&=\mathcal{O}(1) \\
\mathbb{E}[X_{n,m}]&=1+\frac{1}{n}\sum_{k=0}^{n-1} \mathbb{E}[X_{n,m} \mid \text{The last drawn number is}\space k] \\
&=1+\frac{1}{n}\sum_{k=0}^{m-2} \mathbb{E}[X_{n,m}]+\frac{1}{n}\sum_{k=m-1}^{n-1} \mathbb{E}[X_{n,m-1}] \\
&= 1+\frac{m-1}{n}\mathbb{E}[X_{n,m}]+\frac{n-m+1}{n}\mathbb{E}[X_{n,m-1}]
\\
\implies \frac{n-m+1}{n}\mathbb{E}[X_{n,m}]&=\frac{n-m+1}{n}\mathbb{E}[X_{n,m-1}] +1\\
\implies \mathbb{E}[X_{n,m}]&=\frac{n-m+1}{n-m+1}\mathbb{E}[X_{n,m-1}]+\frac{n}{n-m+1}\\
&=\mathbb{E}[X_{n,m-1}]+\frac{n}{n-m+1} \\
&=\sum_{k=1}^m\frac{n}{n-k+1}\\
&=\sum_{k=0}^{m-1}\frac{n}{n-k} \\
&=n\sum_{k=n-m+1}^n\frac{1}{k}\\
&=n(H_n-H_{n-m})
\end{align*}
$$
Where $(H_n)_{n\in\mathbb{N}^*}$ is the harmonic series and $H_0=0$

### 2.3 Expected Complexity
#### 2.3.1 $m=o(n)$
In this case, we have a time complexity of:
$$
\mathcal{O}(m)
$$
### 2.3.2 $m\sim kn$ with $k\in]0,1[$
With that, we can prove that the time complexity is:
$$
\mathcal{O}(n)
$$
### 2.3.3 $m\sim n$
With that, we have a time complexity of:
$$
\mathcal{O}(n\ln n)
$$

## 3. Random Graph
### 3.1 Naive Algorithm

To build such graph, we start with $E\leftarrow\varnothing.$
For each vertex $u\in V,$ while $\Adj u=\varnothing:$
1. For each $v\in V$, we create a bernoulli random variable $B\sim\mathcal{B}(p)$
2. If $B=1$ we add the edge $(u,v):E\leftarrow E\cup\{(u,v)\}$

Now if $p>0,$ this program halts with probability $1$

Let $R(v)$ be the number of times the outer loop is executed.
That is, it is the number of times that $\Adj(v)=\varnothing$ 
We can prove that $R(v)\sim \mathcal{G}(1-(1-p)^n).$ and:
$$
\mathbb{E}[R(v)]=\frac{1}{1-(1-p)^n}
$$
Now the expected runtime of the algorithm is:
$$
\begin{align*}
\mathbb{E}\left[\sum_{v\in V}R(v)f(n,p)\right] &=\sum_{v\in V}f(n,p)\mathbb{E}\left[R(v)\right] \\
&=\frac{nf(n,p)}{1-(1-p)^n} \\
\end{align*}
$$
Where $f(n,p)$ is the cost of adding the random edges.

Naively, $f(n,p)=n$ and we have a performance of:
$$
\mathcal{O}\left(\frac{n^2}{1-(1-p)^n}\right)
$$
Now, let $p=\frac{c}{n}$
We have:
$$
\begin{align*}
(1-p)^n&\sim (1-\tfrac{c}{n})^n\\
&\sim e^{-c}
\end{align*}
$$
With that, the complexity will be:
$$\mathcal{O}\left(\frac{n^2}{1-e^{-c}}\right)
$$


### 3.2 Faster Algorithm

To build such graph, we start with $E\leftarrow\varnothing.$
For each vertex $u\in V,$ while $\Adj u=\varnothing:$
1. For each $v\in V$, we create a binomial random variable $X\sim\mathcal{B}(n,p)$ representing the degree of $v$
2. Now we will draw $X$ vertices uniformly from $V$ without replacement, and put them into a set $S$
3. $\Adj u=S$

Now if $p>0,$ this program halts with probability $1$

Let $R(v)$ be the number of times the outer loop is executed.
That is, it is the number of times that $\Adj(v)=\varnothing$ 
We can prove that $R(v)\sim \mathcal{G}(1-(1-p)^n).$ and:
$$
\mathbb{E}[R(v)]=\frac{1}{1-(1-p)^n}
$$
Now the expected runtime of the algorithm is:
$$
\begin{align*}
\mathbb{E}\left[\sum_{v\in V}R(v)f(n,p)\right] &=\sum_{v\in V}f(n,p)\mathbb{E}\left[R(v)\right] \\
&=\frac{nf(n,p)}{1-(1-p)^n} \\
\end{align*}
$$
Where $f(n,p)$ is the cost of adding the random edges.

We have:
$$
f(n,p)=\mathcal{O}\left(np\right)
$$
And so, the time complexity is:
$$
\mathcal{O}\left(\frac{n^2p}{1-(1-p)^n}\right)
$$

Now, let $p=\frac{c}{n}$
We have:
$$
\begin{align*}
(1-p)^n&\sim (1-\tfrac{c}{n})^n\\
&\sim e^{-c}
\end{align*}
$$
With that, the complexity will be:
$$\mathcal{O}\left(\frac{cn}{1-e^{-c}}\right)=\mathcal{O}(n)
$$


