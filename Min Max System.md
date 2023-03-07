## 1. Max Atom System
### 1.1 Definition
- Let $\mathcal{X}$ be a finite set of variables
- Let $D=I\cup \{-\infty\},$ with $I\subseteq \mathbb{R}$   
- For $x\in \mathcal{X},Y\subseteq\mathcal{X}^m,c\in I$, let $\text{MA}(x,Y,c)$ be defined as follow:
	$$
	\text{MA}(x,Y,c)\iff x\le \max Y+c
	$$
- For $x\in \mathcal{X},Y\subseteq\mathcal{X}^m,c\in I$, let $\text{MI}(x,Y,c)$ be defined as follow:
	$$
	\text{MI}(x,Y,c)\iff x\le \min Y+c
	$$


A min max system is $\text{CSP}(D,\Gamma)$ where:
$$
\begin{align*}
\Gamma&=\left\{O(x,Y,c),\quad (O,x,Y,c)\in \mathscr{R}\right\}\\
\mathscr{R}&\subseteq \{\text{MA},\text{MI}\}\times\mathcal{X}\times \left(\mathscr{P}(\mathcal{X}) \setminus \{\varnothing\}\right)\times I \\
\mathscr{R}&\space \text{is finite}
\end{align*}
$$

### 1.2 Equivalence with Max Atom
A Max Atom system is trivially a Min Max system. So we will only prove the latter implication.

Let $S'=\text{CSP}(D,\Gamma)$ be a Min Max system, and let:
- $\Gamma_{\text{MI}}$ be the constraints that has $\text{MI}$ 
- $\Gamma_{\text{MA}}$ be the constraints that has $\text{MA}$

For each $\text{MI}(x,Y,c)\in \Gamma_{\text{MI}}.$ we replace it with the following constraints:
$$
\Gamma^{x,Y,c}=\bigcup_{y\in Y}\left\{\text{MA}(x,\{y\},c)\right\}
$$
Now, let:
$$
\Gamma'=\Gamma_{\text{MA}}\cup\bigcup_{\text{MI(x,Y,c)}\in \Gamma_{\text{MI}}} \Gamma^{x,Y,c}
$$
The system $\text{CSP}(D,\Gamma')$ is a max system