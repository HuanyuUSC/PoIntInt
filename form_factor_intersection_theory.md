# Intersection Volume via Form Factor Fields — Theory Summary

This note collects formulas you can **paste into Notion**. All math is written in LaTeX (`$...$`, `$$...$$`).

---

## 1) Core definitions

- Shape $\Omega \subset \mathbb{R}^3$, indicator $\chi_\Omega(x)$.
- **Form factor field** (Fourier transform of the indicator):
  $$
  F(k)=\widehat{\chi_\Omega}(k)=\int_{\Omega} e^{i\,k\cdot x}\,dx,\qquad k\in\mathbb{R}^3.
  $$

- **Intersection volume** of two shapes $\Omega_1,\Omega_2$ with relative translation $t$:
  $$
  V_{\cap}(t)=\int_{\mathbb{R}^3}\chi_{\Omega_1}(x)\,\chi_{\Omega_2}(x-t)\,dx
  \;=\;\frac{1}{(2\pi)^3}\int_{\mathbb{R}^3} F_1(k)\,\overline{F_2(k)}\,e^{i\,k\cdot t}\,dk.
  $$

- **Surface-only form** (closed surfaces): define
  $$
  A(k)=\int_{\partial\Omega} n(x)\,e^{i\,k\cdot x}\,dS(x),\qquad
  F(k)=\frac{i}{\|k\|^2}\,k\cdot A(k).
  $$
  A useful scalar is $A_{\parallel}(k)=\dfrac{k\cdot A(k)}{\|k\|}$, so
  $$
  F(k)=\frac{i}{\|k\|}\,A_{\parallel}(k).
  $$

---

## 2) Transform rules (translation / rotation / affine)

Let $R\in SO(3)$, $A\in \mathrm{GL}(3)$, and $c\in\mathbb{R}^3$. Then
- **Translation**: $F_{\Omega+c}(k)=e^{\,i\,k\cdot c}\,F_\Omega(k)$.
- **Rotation**: $F_{R\Omega}(k)=F_\Omega(R^\top k)$.
- **Affine** ($x\mapsto Ax+c$):
  $$
  \boxed{\,F_{A\Omega+c}(k)=|\det A|\;e^{\,i\,k\cdot c}\;F_\Omega(A^\top k)\, .}
  $$

These give exact $F$ for ellipsoids, oriented boxes/cylinders, etc., from axis-aligned formulas.

---

## 3) Ewald / Gaussian split (stability & speed)

Insert
$$
1=e^{-\|k\|^2/\kappa^2}+\big(1-e^{-\|k\|^2/\kappa^2}\big).
$$
Then
$$
\begin{aligned}
V_{\cap}(t)
&=\frac{1}{(2\pi)^3}\int e^{-\|k\|^2/\kappa^2}\,F_1(k)\,\overline{F_2(k)}\,e^{i\,k\cdot t}\,dk\\
&\quad+\iint \chi_{\Omega_1}(x)\,\chi_{\Omega_2}(y)\;g_\kappa(x-y-t)\,dx\,dy,
\end{aligned}
$$
with the Gaussian
$$
g_\kappa(r)=\frac{\kappa^3}{8\pi^{3/2}}\;e^{-(\kappa^2/4)\|r\|^2}.
$$
- The **$k$-part** converges rapidly (small cutoff $K$ suffices).
- The **real-space** term is analytic/separable for boxes and very cheap for surfels (disks/Gaussians).

---

## 4) Element formulas (triangles, disks, Gaussian surfels)

### 4.1 Triangle patch

Vertices $a,b,c$; edges $e_1=b-a$, $e_2=c-a$. Parameterize $x=a+u\,e_1+v\,e_2$ with $u\ge0$, $v\ge0$, $u+v\le1$.
Define the scalar simplex integral
$$
\begin{aligned}
J_\triangle(k;a,e_1,e_2)
&=\int_{u\ge0}\!\!\int_{v\ge0,\ u+v\le1} e^{i\,k\cdot(a+u e_1+v e_2)}\,du\,dv \\
&= e^{i\,k\cdot a}\;\frac{e^{i\,k\cdot e_2}\,\dfrac{e^{i\,k\cdot(e_1-e_2)}-1}{i\,k\cdot(e_1-e_2)} - \dfrac{e^{i\,k\cdot e_1}-1}{i\,k\cdot e_1}}{i\,k\cdot e_2}\;.
\end{aligned}
$$

- **Surface-normal transform** of the triangle (no unit-normal needed):
  $$
  \boxed{\,A_\triangle(k)=(e_1\times e_2)\;J_\triangle(k;a,e_1,e_2)\, .}
  $$
- **Area (indicator) contribution** of the triangle:
  $$
  S_\triangle(k)=\|e_1\times e_2\|\;J_\triangle(k;a,e_1,e_2).
  $$

**Stability near zeros**: use the sinc-series
$$
\frac{e^{i\alpha}-1}{i\alpha}=1-\frac{i\alpha}{2}-\frac{\alpha^2}{6}+\cdots\quad (|\alpha|\ll 1).
$$

### 4.2 Circular disk surfel

Center $c$, radius $\rho$, unit normal $n$. Let $\Pi=I-n n^\top$ and $k_\perp=\Pi k$, $r=\|k_\perp\|$.
- **Area transform**:
  $$
  S_{\mathrm{disk}}(k)=e^{\,i\,k\cdot c}\;\frac{2\pi\,\rho\,J_1(\rho r)}{r},\qquad \text{with } \lim_{r\to0}\frac{J_1(\rho r)}{r}=\frac{\rho}{2}.
  $$
- **Surface-normal transform**:
  $$
  A_{\mathrm{disk}}(k)=n\;S_{\mathrm{disk}}(k).
  $$

### 4.3 Gaussian surfel (planar Gaussian footprint)

Let the surfel carry **area weight** $w$ distributed by a *normalized* planar Gaussian of std $\sigma$ in its tangent plane.
With $k_\perp=(I-n n^\top)k$,
- **Area transform**:
  $$
  S_{\mathrm{gauss}}(k)=w\,e^{\,i\,k\cdot c}\;e^{-\tfrac12\,\sigma^2\|k_\perp\|^2}.
  $$
- **Surface-normal transform**:
  $$
  A_{\mathrm{gauss}}(k)=n\;S_{\mathrm{gauss}}(k).
  $$

> If instead you use *unnormalized* density $w\cdot (2\pi\sigma^2)^{-1}\!\int\!e^{-\|x_\perp\|^2/(2\sigma^2)}dx_\perp=w$, the same closed form holds.

---

## 5) Closed-form $F(k)$ for basic shapes

### 5.1 Axis-aligned box (centered), half-lengths $(a,b,c)$
$$
\boxed{\,F_{\mathrm{box}}(k)=\prod_{j\in\{x,y,z\}}\frac{2\sin(a_j\,k_j)}{k_j},\qquad \text{use } \frac{2\sin(a_j k_j)}{k_j}\Big|_{k_j=0}=2a_j.}
$$

**Unit cube** $[-\tfrac12,\tfrac12]^3$:
$$
F_{\mathrm{cube}}(k)=\prod_{j=1}^3 \frac{2\sin(k_j/2)}{k_j},\qquad F(0)=1.
$$

### 5.2 Solid ball (radius $R$)
Let $k=\|k\|$.
$$
\boxed{\,F_{\mathrm{ball}}(k)=4\pi\,\frac{\sin(kR)-kR\cos(kR)}{k^3},\qquad \lim_{k\to0}F_{\mathrm{ball}}(k)=\frac{4\pi R^3}{3}.}
$$

### 5.3 Right circular cylinder (radius $a$, height $h$, axis $\hat z$)
Let $k_\perp=\sqrt{k_x^2+k_y^2}$.
$$
\boxed{\,F_{\mathrm{cyl}}(k)=\frac{2\sin\!\big(\tfrac{k_z h}{2}\big)}{k_z}\;\cdot\;\frac{2\pi a\,J_1(a k_\perp)}{k_\perp},}
$$
with the obvious limits at $k_z=0$ or $k_\perp=0$. Rotations/translations follow by the affine rule.

### 5.4 Ellipsoid and general affine images
If $\Omega=\{x=c+A u:\ \|u\|\le 1\}$ (ellipsoid), then by the affine rule
$$
\boxed{\,F_{\mathrm{ellipsoid}}(k)=|\det A|\;e^{\,i\,k\cdot c}\;F_{\mathrm{ball}}\!\left(\|A^\top k\|\right).}
$$
This also yields oriented boxes (diagonal $A$ plus rotation), capsules (Minkowski sums via products in $k$-space), etc.

---

## 6) Gradients / Hessians of $V_{\cap}$

Let parameters $q$ affect only $\Omega_1$ (shape $\Omega_2$ fixed). Define
$$
V_{\cap}(t,q)=\frac{1}{(2\pi)^3}\int F_1(k;q)\,\overline{F_2(k)}\,e^{i\,k\cdot t}\,dk.
$$

### 6.1 Translation derivatives
$$
\nabla_t V_{\cap}(t)=\frac{1}{(2\pi)^3}\int i\,k\;F_1(k)\,\overline{F_2(k)}\,e^{i\,k\cdot t}\,dk,
$$
$$
H_t(t)=\frac{\partial^2 V_{\cap}}{\partial t\,\partial t^\top}= -\frac{1}{(2\pi)^3}\int (k\,k^\top)\;F_1(k)\,\overline{F_2(k)}\,e^{i\,k\cdot t}\,dk.
$$

### 6.2 Shape-parameter derivatives (chain rule)
$$
\boxed{\,\frac{\partial V_{\cap}}{\partial q}=\frac{1}{(2\pi)^3}\int \frac{\partial F_1}{\partial q}(k;q)\;\overline{F_2(k)}\;e^{i\,k\cdot t}\,dk.}
$$

Using the **surface path**, $F=\dfrac{i}{\|k\|^2}k\cdot A$,
$$
\frac{\partial F}{\partial q}(k)=\frac{i}{\|k\|^2}\,k\cdot\frac{\partial A}{\partial q}(k).
$$

- **Triangle element** $A_\triangle(k)=(e_1\times e_2)\,J_\triangle$:
  $$
  \frac{\partial A_\triangle}{\partial a}=(e_1\times e_2)\;\frac{\partial J_\triangle}{\partial a},\qquad
  \frac{\partial J_\triangle}{\partial a}= i\,k\,J_\triangle.
  $$
  $$
  \frac{\partial A_\triangle}{\partial e_1}=(\partial_{e_1}(e_1\times e_2))\,J_\triangle+(e_1\times e_2)\,\frac{\partial J_\triangle}{\partial e_1},
  \quad \partial_{e_1}(e_1\times e_2)[v]=v\times e_2.
  $$
  For the scalar $J_\triangle$ with $\alpha=k\cdot e_1$, $\beta=k\cdot e_2$,
  $$
  \frac{\partial J_\triangle}{\partial e_1}=\frac{\partial J_\triangle}{\partial \alpha}\,k,\qquad
  \frac{\partial J_\triangle}{\partial e_2}=\frac{\partial J_\triangle}{\partial \beta}\,k,
  $$
  where $\partial J_\triangle/\partial \alpha$, $\partial J_\triangle/\partial \beta$ come from differentiating the closed form; use the sinc-series for small $|\alpha|,|\beta|$.

- **Disk / Gaussian surfel**: $A_e(k)=n\,S_e(k)$.
  - If only the center $c$ moves: $\partial S/\partial c = i\,k\;S$.
  - If radius $\rho$ (disk) changes: $\partial S/\partial \rho = e^{i k\cdot c}\,2\pi\!\left[\frac{J_1(\rho r)}{r} + \rho\,J_1'(\rho r)\right]$ (chain rules for $r$).
  - For Gaussian: $\partial S/\partial \sigma = -\sigma\,\|k_\perp\|^2\,S$.

### 6.3 Hessians
Differentiate the gradient again. In practice, use **automatic differentiation** through the exact element formulas; for optimization near quadratic regimes, a **Gauss–Newton** approximation (dropping second-order terms in $\partial F/\partial q$) is very effective.

---

## 7) Coefficient-space (Laguerre + Spherical Harmonics) shortcut

Expand with an orthonormal basis
$$
F(k)\approx \sum_{n,\ell,m} c_{n\ell m}\,R_{n\ell}(\|k\|)\,Y_{\ell m}(\hat k)\quad\Rightarrow\quad
V_{\cap}(t)=\sum_{n,\ell,m} c^{(1)}_{n\ell m}\;\overline{c^{(2)}_{n\ell m}}\;e^{i\,\phi_{n\ell m}(t)}.
$$
Then
$$
\frac{\partial V_{\cap}}{\partial q}=\mathrm{Re}\big[J(q)^\top\,\overline{\mathbf c^{(2)}}\big],\qquad
J=\frac{\partial \mathbf c^{(1)}}{\partial q},
$$
with $J$ assembled from **polynomial moments** of triangle/disc/Gaussian elements (no runtime $k$-quadrature).

---

## 8) Numerical stability tips

- Use **Ewald/Gaussian** damping to keep bandlimits modest and stabilize $t=0$.
- In triangle formulas, replace $\dfrac{e^{i\alpha}-1}{i\alpha}$ by its **sinc-series** for $|\alpha|\lesssim 10^{-4}$.
- For disks, define $J_1(\rho r)/r$ at $r=0$ via its limit ($\rho/2$).
- Prefer $A_\triangle=(e_1\times e_2)J_\triangle$ to avoid unit-normal normalization.
- Cache $k$-dependent factors (Bessel, exponentials, radial polynomials) across elements.
