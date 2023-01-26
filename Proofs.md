
## 1. Max-closed Linear Program
> A linear inequality is max-closed if and only if it is of the form
> $$
 \sum_{i=1}^n a_ix_i> c
 $$
 With $a_1,\dots,a_n$ are all positive except potentially one term. 

- Let $\sum_{i=1}^na_ix_i \succ c$ be a linear inequality.
- Let $\boldsymbol{x}=(x_1,\dots,x_n),\boldsymbol{y}=(y_1,\dots,y_n)$ be two vectors satisfying the constraints.
- Let $\boldsymbol{z}=\max(\boldsymbol{x},\boldsymbol{y})$ applied component wise

### 1.1 Case 1:
Suppose that $\forall i\in\{1,\dots,n\}, \quad a_i\ge 0$

We have:
$$
\max\left(\sum_{i=1}^na_ix_i,\sum_{i=1}^n a_iy_i\right)\le \sum_{i=1}^n\max(a_ix_i,a_iy_i)=\sum_{i=1}^na_i\max(x_i,y_i)=\sum_{i=1}^na_iz_i
$$
But as bot $\boldsymbol{x},\boldsymbol{y}$ satisfies the inequality, then:
$$
\sum_{i=1}^n a_ix_i \succ c
$$
### 1.2 Case 2:
- Suppose that only one $\exists ! j\in\{1,\dots,n\}/\quad a_j <0$ 
- Let $b=-a_n > 0$
- Without a loss of generality, we will suppose that $j=n$

We have the following:
$$
\begin{cases}
bx_n+c \prec \sum_{i=1}^{n-1}a_ix_i \\
by_n+c \prec \sum_{i=1}^{n-1}a_iy_i 
\end{cases} \implies \max(x_n+c,y_n+c)\prec \max\left(\sum_{i=1}^{n-1}a_ix_i,\sum_{i=1}^{n-1}a_iy_i\right)
$$
But we have:
$$
\max\left(\sum_{i=1}^{n-1}a_ix_i,\sum_{i=1}^{n-1}a_iy_i\right)  \le \sum_{i=1}^{n-1}a_iz_i \quad \text{as}\space \forall i\in\{1,\dots,n-1\},\quad a_i\ge0
$$
With that we have:
$$
\max(bx_n+c,by_n+c)=\max(bx_n,by_n)+c=bz_n+c \prec \sum_{i=1}^{n-1}a_iz_i
$$
With proves that:
$$
\sum_{i=1}^n a_iz_i \succ c
$$
### 1.3 Case 3 (To finish, we should prove that max is not a polymorphism):
- Suppose that there are $m>1$ variables with negative coefficients, and $n$ variables with positive coefficients

- Without a loss of generality, we will denote the inequality as:
	$$
	\sum_{i=1}^{m}b_ix^-_i +c \prec \sum_{i=1}^n a_ix^+_i
	$$
	
- Let $\boldsymbol{x}=(x_1^+,\dots,x_n^+,x_1^-,\dots,x_m^-),\boldsymbol{y}=(y_1^+,\dots,y_n^+,y_1^-,\dots,y_m^-)$ be two vectors satisfying the inequality above