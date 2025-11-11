# PoIntInt: PoINT-based INTersection Volume as Boundary INTegral

本项目使用 `uv` 作为包管理器，已配置好 PyTorch CUDA 12.6 和 Kaolin 的索引源。

## 环境安装

1. 确保已安装 [uv](https://docs.astral.sh/uv/)
2. 在项目根目录执行：
   ```bash
   uv sync
   ```
## 运行脚本

示例脚本位于 `simplicits_demo.py`，用于 CLI 环境重现 Demo：

- `uv run python simplicits_demo.py --mesh assets/fox.obj --export fox_deformed.obj`
- 需要可用的 GPU（CUDA 12.6 对应驱动）以及与 `pyproject.toml` 匹配的 PyTorch、Kaolin 版本。
- 可以通过参数调整采样数、训练步数、模拟步数等；运行 `--help` 查看完整选项。

# Fast Intersection Volumes

This repo explores how to compute **intersection volumes between 3D solids** using a **boundary-only formulation** and evaluates it numerically with a **Fourier (k-space) method**.

The pipeline is:

1. Express the intersection volume of two solids as a **double surface integral** over their boundaries.  
2. Rewrite the $1/\|x-y\|$ kernel via its **Fourier representation**.  
3. Discretise k-space (Gauss–Legendre × Lebedev) and precompute per-object Fourier data.  
4. Recover all pairwise overlap volumes from a single matrix product $J^\top D J$.

---

## 1. Mathematical background

### 1.1 Indicator via solid angle / double layer

For a bounded region $\Omega \subset \mathbb{R}^3$ with boundary $\partial\Omega$ and outward normal $n(x)$, its indicator function $\chi_\Omega(p)$ can be written as a **double-layer potential** (solid angle of $\partial\Omega$ as seen from $p$):

$$
\chi_\Omega(p)
  = -\frac{1}{4\pi}
    \int_{\partial\Omega}
      \frac{(p - x)\cdot n(x)}{\|p - x\|^3}\, dS_x.
$$

This equals $1$ when $p$ is inside $\Omega$ and $0$ outside, assuming a well-behaved closed surface. This is the same object that appears in generalized winding number formulations in geometry processing.

### 1.2 Double surface integral for intersection volume

For two solids $\Omega_1,\Omega_2$, the intersection volume is

$$
\text{Vol}(\Omega_1 \cap \Omega_2)
  = \int_{\mathbb{R}^3}
      \chi_{\Omega_1}(p)\,\chi_{\Omega_2}(p)\, dp.
$$

Plugging the double-layer representation of $\chi_{\Omega_2}$ into the integral and using the divergence theorem on $\Omega_1$ converts the volume integral into a **double surface integral** over the two boundaries:

$$
\text{Vol}(\Omega_1 \cap \Omega_2)
  = \frac{1}{4\pi}
    \int_{\partial\Omega_1}
    \int_{\partial\Omega_2}
      \frac{n_1(x)\cdot n_2(y)}{\|x - y\|}\,
    dS_x\, dS_y.
$$

This is the main continuous formula behind the project: overlap is a **Coulomb-type interaction energy** between two surfaces, with kernel $1/\|x-y\|$ and a normal–normal dot product.

This structure is closely related to:

- **Generalized winding numbers** and solid angles in computer graphics,  
- **Double-layer potentials** / boundary integral methods in potential theory,  
- **View-factor integrals** in radiative heat transfer, which also integrate normal products over $1/\|x-y\|^2$.

---

## 2. Fourier formulation (main numerical method)

The project does **not** evaluate the double surface integral directly. Instead it uses a **Fourier representation** of the Newtonian kernel:

$$
\frac{1}{\|x-y\|}
  = \frac{1}{(2\pi)^3}
    \int_{\mathbb{R}^3}
      \frac{e^{i k\cdot(x-y)}}{\|k\|^2}\, dk.
$$

Inserting this into the surface–surface formula and rearranging integrals yields a k-space expression

$$
\text{Vol}(\Omega_1 \cap \Omega_2)
  = C
    \int_{\mathbb{R}^3}
      \frac{A_1(k)\, A_2(-k)}
           {\|k\|^2}\, dk,
$$

where for each object $\Omega_n$

$$
A_n(k)
  = \int_{\partial\Omega_n} e^{ik\cdot x}\, n(x)\, dS_x
  = i k\, F_n(k),
$$

and

$$
F_n(k)
  = \int_{\Omega_n} e^{ik\cdot x}\, dx
$$

is the Fourier transform of the indicator function $\chi_{\Omega_n}$. In words:

> Each solid is summarized by a **form-factor field** $F_n(k)$ in Fourier space; overlap is an inner product of these fields with weight $1/\|k\|^2$.

This is the main numerical object the code computes.

---

## 3. Discretising k-space

We parameterize frequencies in spherical coordinates

$$
k = (k_x,k_y,k_z),
\quad
k_x = r\sin\theta\cos\phi,\;
k_y = r\sin\theta\sin\phi,\;
k_z = r\cos\theta.
$$

The integral over $k$ is discretised by a tensor-product quadrature:

- **Radial part** $r$:
  - Gauss–Legendre nodes $t_\ell$ mapped to $[0,\infty)$,
  - $\ell = 1,\dots,L$ with $L \in \{16,32,64,96,128\}$.
- **Angular part** $(\theta_j,\phi_j)$:
  - Lebedev spherical quadrature with $M \approx 10^2\!-\!10^4$ nodes.

Total number of k-nodes is $N_k = L \times M \approx 10^3\!-\!10^6$.

The integral turns into a sum

$$
\text{Vol}(\Omega_1 \cap \Omega_2)
  \approx
    \sum_{q=1}^{N_k}
      w_q\,
      \frac{A_1(k_q)\, A_2(-k_q)}{\|k_q\|^2},
$$

where $w_q$ already includes Jacobian and quadrature weights.

---

## 4. Matrix view for many objects

If we have many solids $\Omega_1,\dots,\Omega_n$, we precompute their form-factor fields on the *same* set of k-nodes:

- For each object $n$ and node $q$:
  - Evaluate $F_n(k_q)$ or $A_n(k_q) = i k_q F_n(k_q)$.
- Stack these into a matrix $J \in \mathbb{C}^{N_k \times n}$, one column per object.
- Let $D \in \mathbb{R}^{N_k \times N_k}$ be diagonal, containing the weights
  $w_q / \|k_q\|^2$.

Then all pairwise intersection volumes are approximated (up to a global factor) by

$$
V \approx J^\top D J.
$$

So computing all $n\times n$ overlaps reduces to a **single symmetric matrix product**, which is GPU-friendly and easy to optimize.

---

## 5. Analytic test shapes

To debug and validate the k-space discretisation, the repo includes shapes with known closed-form Fourier transforms.

### 5.1 Axis-aligned cube $[-1/2,1/2]^3$

$$
F_{\text{cube}}(k)
  = \prod_{i=x,y,z}
      \frac{\sin(k_i/2)}{k_i/2}
  = \operatorname{sinc}\!\left(\frac{k_x}{2}\right)
    \operatorname{sinc}\!\left(\frac{k_y}{2}\right)
    \operatorname{sinc}\!\left(\frac{k_z}{2}\right),
$$

so $F_{\text{cube}}(0) = 1$ (unit volume).

### 5.2 Unit ball

$$
F_{\text{ball}}(k)
  = 4\pi
    \frac{
      \sin\|k\|
      - \|k\|\cos\|k\|
    }{\|k\|^3}.
$$

We compare numerical quadrature against these analytic formulas to check convergence vs. $L,M$.

---

## 6. Moments from $F(k)$

A Taylor expansion of $F_n(k)$ around $k=0$

$$
F_n(k)
  = F_n(0)
    + k\cdot \nabla F_n(0)
    + \tfrac12 k^\top \nabla^2 F_n(0)\, k + \dots
$$

encodes geometric quantities of $\Omega_n$:

- $F_n(0) = \text{Vol}(\Omega_n)$,
- $\nabla F_n(0)$ gives the **barycentre**,
- $\nabla^2 F_n(0)$ contains **second moments / inertia tensor**.

So once we have the Fourier data, we can also recover basic shape descriptors “for free”.

---

## 7. Relation to other approaches

The double surface integral and its Fourier rewrite connect this project to several existing ideas:

- **Generalized winding numbers** (Jacobson et al., Barill et al.): use the same double-layer potential for robust inside/outside tests.  
- **Boundary integral methods**: represent harmonic functions via layer potentials; our formula is essentially an interaction energy of two double layers.  
- **View-factor integrals** in radiative heat transfer: similar normal–normal kernels for energy exchange between surfaces.

Here, the focus is specifically on **intersection volume** and **pairwise overlap matrices** for arbitrary solids, evaluated mainly via the **Fourier method** rather than direct spatial quadrature or FMM.

---

## 8. Status and roadmap

Planned / current components:

- [x] Theory note: derivation of double surface integral and literature survey  
- [x] Analytic reference implementations for cube / sphere form factors  
- [ ] Mesh-based discretisation of $\partial\Omega$ and numerical evaluation of $F_n(k_q)$  
- [ ] GPU implementation of the k-space sampling and $J^\top D J$  
- [ ] Experiments on:
  - shape similarity based on overlap volume,
  - clustering / kernels built from $V$,
  - potential applications in contact energies.

---

## 9. License

TBD.
