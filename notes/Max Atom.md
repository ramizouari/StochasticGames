## 1. Ternary Max Atom system
### 1.1 Definition
- Let $\mathcal{X}$ be a finite set of variables
- Let $D=I\cup \{-\infty\},$ with $I\subseteq \mathbb{R}$   
- For $x,y,z\in \mathcal{X},c\in I$, let $\text{MA}(x,y,z,c)$ be defined as follow:
	$$
	\text{MA}(x,y,z,c)\iff x\le \max(y,z)+c
	$$
A ternary max atom system is $\text{CSP}(D,\Gamma)$ where:
$$
\begin{align*}
\Gamma&=\left\{\text{MA}(x,y,z,c),\quad (x,y,z,c)\in \mathscr{R}\right\}\\
\mathscr{R}&\subseteq \mathcal{X}^3\times I\\
\mathscr{R}& \space \text{is finite}
\end{align*}
$$
### 1.2 Examples
Let $S=\text{CSP}(\mathcal{X},D,\Gamma)$  with:
- $\mathcal{X}=\{x,y,z\}$
- $D=\mathbb{R}\cup \{-\infty\}$
- $\Gamma$ is generated as follow:
	$$
	\begin{align*}
	x &\le \max(y,z)-1\\
	y & \le \max(x,z)-1\\
	z & \le \max(x,y)-1
	\end{align*}
	$$


## 2. Max Atom System
### 2.1 Definition
- Let $\mathcal{X}$ be a finite set of variables
- Let $D=I\cup \{-\infty\},$ with $I\subseteq \mathbb{R}$   
- For $x\in \mathcal{X},Y\subseteq\mathcal{X}^m,c\in I$, let $\text{MA}(x,Y,c)$ be defined as follow:
	$$
	\text{MA}(x,Y,c)\iff x\le \max Y+c
	$$
A ternary max atom system is $\text{CSP}(D,\Gamma)$ where:
$$
\begin{align*}
\Gamma&=\left\{\text{MA}(x,Y,c),\quad (x,Y,c)\in \mathscr{R}\right\}\\
\mathscr{R}&\subseteq \mathcal{X}\times \left(\mathscr{P}(\mathcal{X}) \setminus \{\varnothing\}\right)\times I \\
\mathscr{R}&\space \text{is finite}
\end{align*}
$$

### 2.2 Ternary Max Atom $\le$ Max Atom
This is immediate

### 2.3 Max Atom $\le$ Ternary Max Atom
- Let $S=\text{CSP}(\mathcal{X},D,\Gamma)$ a max atom system.
- Let $R\in \Gamma$
- Let $x\in \mathcal{X},Y\in\mathscr{P}(\mathcal{X}),c\in I$ such that $R=\text{MA}(x,Y,c)$ such that $\lvert Y \rvert >2$

#### 2.3.1 Recursive Reduction
We will reduce the arity of $R$ as follow:
- Let $y,z\in Y$ such that $y\ne z$
- We introduce a variable $w\notin \mathcal{X}$
- Let $\mathcal{X}'=\mathcal{X}\cup\{w\}$
- Let $Y'=(Y\cup \{w\})\setminus\{y,z\}$
- Let $R'=\text{MA}(x,Y',c)$
- Let $R_w=\text{MA}(w,\{y,z\},0)$
- Let $\Gamma'=(\Gamma\cup\{R',R_w\})\setminus \{R\}$
- Let $S'=\text{CSP}(\mathcal{X}',D,\Gamma)$

We will prove that $S'$ is equivalent to $S.$

Without a loss of generality:
- we will order $\mathcal{X}$ such that $x_0\le\dots\le x_{n-1}$ with $n=\lvert \mathcal{X}\rvert$ 
- $x_{n-2}=y$
- $x_{n-1}=z$
- We will set $x_n=w$
- Let $i\in\{0,\dots,n-1\}$ such that $x_i=x$

##### Implication
Let $s_0,\dots,s_{n}$ an assignment of $S'.$ It is trivial that $s_0,\dots,s_{n-1}$ is an assignment of $S$ 

##### Equivalence
Let $s_0,\dots,s_{n-1}$ an assignment of $S'$
Let $s_n=\max(s_{n-1},s_{n-2})$

Then, $s_0,\dots,s_n$ is an assignment of $S'$

#### 2.3.2 Induction
Since the number of variables is finite. The number of constraints is finite, and the arity of each constraint is finite. Applying such reduction iteratively will eventually give a system $S^*$ equivalent to $S$ with:
- $\mathcal{X}^*$ the set of variables with $\mathcal{X}\subseteq \mathcal{X}'$ 
- $\Gamma^*$ is the set of constraints:
	- Each constraint is of the form $\text{MA}(x,Y,c)$ with $x\in \mathcal{X}',Y\subseteq \mathcal{X}',c\in I$ with $\lvert Y\rvert \le 2$   

Now such system can be transformed to a ternary system $S_3$ as follow:
- The set of variables is $\mathcal{X}^*$
- The domain is $D$
- For every relation $R=\text{MA}(x,Y,c)$ we map it to the relation $R_3=\text{MA}(x,y,z,c)$ as follow:
	- $y,z$ constitute the elements of $Y$ if $\lvert  Y \rvert=2$
	- $z=y$ if $\lvert Y \rvert=1$

It is trivial that $S^*$ is equivalent to $S_3.$
With that, $S$ is equivalent to $S_3.$