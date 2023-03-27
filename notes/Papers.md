
## 1. Tropically Convex Constraints Satisfaction

### 1.1 Constraint Satisfaction Problem
#### 1.1.1 Proposed Definition
- Let $D$ be a set
- Let $\tau=\{R_1,\dots\}$ be a set of predicate symbols
- Let $\Gamma= (D,R_1^\Gamma,\dots)$ with $R_i^\Gamma \subseteq D^{k_i}.$ That is, $R_i$ is a $k_i$-arity relation over $D$ in $\Gamma$
- $\Gamma$ is called a $\tau$-structure
- We will suppose that $\lvert \tau \rvert$ is finite

#### 1.1.2 Instance
- A set of formal variables $x_1,\dots,x_n$
- Each expression is of the form:
	$$
	\mathtt{E}_i=R_{h(i)}^\Gamma (x_{\delta_i(1),\dots},x_{\delta_i(k_{h(i)})})
	$$
	With:
	- $h(i)\in\{1,\dots,\lvert \tau \rvert\}$
	- $\delta_i:\{1,\dots,k_{h(i)}\}\rightarrow \{1,\dots,n\}$
#### 1.1.3 Admissible Solution
It is a solution of the CSP, that is, a set of variables $(x_1,\dots,x_n)\in D$ such that:
$$
\bigwedge_{i} \mathtt{E}_i \quad \text{is True}
$$
### 1.2 Semi Linear Program
*(Why the word semi linear?)*
#### 1.2.1 Semi-linear Relation
A relation $R$ is **by definition**, semi-linear if, $R$ can be defined using first order logic over (the ordered field ?)  $(\mathbb{Q},+,\le,1)$

#### 1.2.2 Semi-linear CSP
$\Gamma$ is called semi-linear when each relation over $\Gamma$ is semi-linear

### 1.3 Primive Positive Definability
In formal logic, it is a first order logic fomula defined as follow:
$$
\exists x_1,\dots,x_n / \bigwedge_{i=1}^m \psi_i
$$
Where $\psi_1,\dots,\psi_m$ are atomic formulas that depends on $x_1,\dots,x_n$

##### Theorem
> Every expansion of a CSP $\Gamma$ by finitely many primitive positive definable formulas can be reduced in polynomial time to the same CSP.

### 1.4 Polymorphism
**By definition**, We say that $\Phi\in \mathscr{F}(\Gamma^k,\Gamma)$ is a polymorphism when:
$$
\forall R\in \tau,\forall \boldsymbol{a}_1,\dots,\boldsymbol{a}_k\in R^\Gamma,\quad \left(\Phi(\boldsymbol{a}^{1}), \Phi(\boldsymbol{a}^m)\right) \in R^\Gamma
$$
Where:
- $m$ is the arity of the relation $R$
- $\forall i\in\{1,\dots,m\},\quad \boldsymbol{a}^i=(a_{1,i},\dots,a_{k,i})$

##### Theorem
> For finite structures $\Gamma$ (when $D$ is finite?), a relation $R$ is primitive positive definable in $\Gamma$ *if and only if* $R$ is **preserved** by all polymorphisms of $\Gamma$

**Theorem**
> A semi-linear relation is is convex *if and only if* the cyclic function $\Phi:(x,y)\rightarrow \frac{x+y}{2}$ is a polymorphism

**Conjecture (Tractibility Conjecture)** 
> $\text{CSP}(\Gamma)$ is in $\mathtt{P}$ *if and only if* $\Gamma$ has a cyclic polymorphism.

#### 1.4.1 Max Atom problem
It is defined as:
$$
M_c \triangleq \left\{(x_1,x_2,x_3)/ \quad x_1 +c \le \max(x_2,x_3) \right\}
$$
- It is open whether $M_c$ is in $\mathtt{P}$
- It is equivalent to a mean-payoff game
- $\max$ is a polymorphism of $M_c$
- It is known to be in $\mathtt{NP}\cap \text{co-}\mathtt{NP}$    

#### 1.4.2 Max closed Semi-Linear programs
**By definition**, it is a semi-linear program that has $\max$ as a polymorphism


When $\Gamma=(D,\tau)$ has a finite $D,$ and $D$ has a total order $\le,$ and $\Gamma$ is max-closed, then $\text{CSP}(\Gamma)$ is in $\mathtt{P}.$
For semi-linear max-closed CSPs, the problem is open.


#### 1.4.3 Tropically Convex CSP
A translation invariant CSP is a CSP on which $x\rightarrow x+c$ is a polymorphism forall $c.$
**By definiton**, a tropical convex CSP is translation-invariant max-closed semi-linear CSP.

##### Result
> A tropically convex CSP is in $\mathtt{NP}\cap \text{co-}\mathtt{NP}$


### 1.5 Semi Linear Horn clause
#### 1.5.1 Definition
a semi-linear horn clause is defined as a clause of the form:
$$
\bigvee_{i=1}^m \boldsymbol{a}_i^T\boldsymbol{x} \succ_i c_i
$$
With:
- $\boldsymbol{a}_1,\dots,\boldsymbol{a}_m\in\mathbb{Q}^n$ satisfying the following condition:
	$$
	\exists k\in\{1,\dots,n\}/\forall i\in\{1,\dots,m\},\forall j\in\{1,\dots,n\}\setminus\{k\},\quad a_{i,j}\ge 0
	$$
	That is to say, that there exists at most a unique variable whose coefficients in each clause can be negative
- $\boldsymbol{x}=(x_1,\dots,x_n)$ a vector of variables
- $c_1,\dots,c_m\in\mathbb{Q}$
- $\succ_1,\dots,\succ_m \in\{>,\ge\}$

#### 1.5.2 Max Atom Clause
$M_c$ is equivalent to:
$$
\{(x,y,z) / \quad x +c < y \vee x+c < z\} = \{(x,y,z)/\quad y-x >c \vee z-x > c\}
$$
Which is essentially a semi-linear horn clause.
