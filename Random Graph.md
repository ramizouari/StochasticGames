## 1. $\mathcal{G}(n,p)$
### Probability of a sinkless graph
#### Formula
- Let $G\sim \mathcal{G}(n,p)$
- Let $v$ a vertex of $G$

$$
\DeclareMathOperator{\Adj}{Adj}
\begin{align*}
\mathcal{P}(\Adj v=\varnothing) 
&=\mathcal{P}(\deg v=0) \\
&= (1-p)^n \\
\implies \mathcal{P}(\Adj v\ne \varnothing) &= 1-(1-p)^n\\
\implies \mathcal{P}(\Adj v\ne \varnothing\quad\forall v)&=\prod_{v}\mathcal{P}(\Adj v\ne \varnothing) \\
&=\left(1-(1-p)^n\right)^n
\end{align*}
$$ 

#### Asymptotic Analysis
Let:
$$
\begin{align*}
f:\mathbb{R}_+^*\times \mathbb{R}_+\times \mathbb{R} & \rightarrow \mathbb{R}_+\\
x,k,c&\rightarrow (1-g(x,k,c))^x\\
g:\mathbb{R}_+^*\times \mathbb{R}_+\times \mathbb{R} & \rightarrow \mathbb{R}_+\\
x,k,c&\rightarrow \left(1-\frac{k \ln x+c}{x}\right)^x
\end{align*}
$$
In fact, we have $f(n,k,c)$ is the probability of a sinkless graph following $\mathcal{G}(n,\tfrac{k\ln n+c}{n})$

We have:
$$
\begin{align*}

\ln g(k,x,c)&=x\ln\left(1-\frac{k\ln x+c}{x}\right)\\
&=-k\ln x-c -\frac{(k(\ln x)+c)^2}{2x}+o\left(\frac{(\ln x)^3}{x^2}\right)\\
\implies g(x,k,c)&=\exp\left(-k\ln x-c -\frac{(k\ln x+c)^2}{2x}+o\left(\frac{(\ln x)^3}{x^2}\right)\right) \\
&=\frac{e^{-c}}{x^k}\times e^{\frac{-(k\ln x+c)^2}{2x}+o\left(\frac{(\ln x)^3}{x^2}\right)}\\
&=\frac{e^{-c}}{x^k}\left(1-\frac{(k \ln x+c)^2}{2x}+o\left(\frac{(\ln x)^3}{x^2}\right)\right)  \\
&=\frac{e^{-c}}{x^k}-e^{-c}\frac{k^2(\ln x)^2}{2x^{k+1}}+o\left(\frac{(\ln x)^3}{x^{k+2}}\right)\\
&=\frac{e^{-c}}{x^k}+o\left(\frac{1}{x^k}\right)\\
\implies 1- g(x,k,c)&=1-\frac{e^{-c}}{x^{k}} +o\left(\frac{1}{x^k}\right)\\
\implies x\ln(1-g(x,k,c))&= -\frac{e^{-c}}{x^{k-1}}+o\left(\frac{1}{x^{k-1}}\right) \\
&\sim -\frac{e^{-c}}{x^{k-1}}
\end{align*}
$$
Now with that:
$$
\lim_{x\rightarrow +\infty} x\ln (1-g(x,k,c))=\begin{cases}
-\infty  & \text{if} \space k\in[0,1[ \\
-e^{-c} & \text{if}\space k=1  \\
0 & \text{otherwise if}\space k\in ]1,+\infty[ 
\end{cases}
$$
Finally, we can conclude that:
$$
\lim_{x\rightarrow +\infty} f(x,k)\begin{cases}
0  & \text{if} \space k\in[0,1[ \\
e^{-e^{-c}} & \text{if}\space k=1  \\
1 & \text{otherwise if}\space k\in ]1,+\infty[ 
\end{cases}
$$


## $\mathcal{G}(n,m)$
- Let $G_m\sim \mathcal{G}(n,m)$
- Let $v_m$ a vertex of $G_m$

$$
\DeclareMathOperator{\Adj}{Adj}
\begin{align*}
\mathcal{P}(\Adj v_m=\varnothing) 
&=\mathcal{P}(\deg v=0) \\
&= \prod_{k=0}^{m-1}\frac{n^2-n-k}{n^2-k} \\
&=\prod_{k=0}^{m-1}1-\frac{n}{n^2-k} \\
\mathcal{P}(\exists v_m\in G_m,\quad \Adj v_m= \varnothing) &=\sum_{k=1}^n {n \choose }
\end{align*}
$$

### $\mathcal{G}(n,p)$ such that $\min_{v}\deg v>0$

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



