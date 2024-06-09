![status](https://img.shields.io/badge/status-beta-red.svg)
[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code-style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Gram-Schmidt-Networks

Perceptron-wise Inter-relational Artificial Neural Networks based on Gram Schmidt orthogonalization process. 

## Explanation
### Vector projections

Given two vectors $u, v \in V$ being $V$ a linear space, the projection of $v$ onto $u$ is defined as follows:

$proj_uv = u \frac{(v, u)}{(u, u)}$

for a given inner product $(⋅ , ⋅)$

### Frobenius inner product
The inner product for the vector space of linear transformations.

$(A,B)_F = Tr(A^TB)$

where $Tr$ denotes the trace of a matrix.
### Gram-Schmidt Orthogonalization process
To create an orthogonal set $\left( u_k \right)_{k = 1}^n$ from any set of vectors $\left( v_k \right)_{k = 1}^n$

$u_1 = v_1$

$u_2 = v_2 - proj_{u_1}v_2$

$u_3 = v_3 - proj_{u_2}v_3 - proj_{u_1}v_3$

⋅

⋅

⋅

$u_k = v_k - \sum_{i = 1}^{k-1} proj_{u_i}v_k$

### Implementation
The idea behind GSNs is to orthogonalize the linear transformations of a whole layer as if it were a set of elements from a given vector space. After orthogonalization, the output from each one is summed.

## Contact  

- [Linkedin](https://www.linkedin.com/in/jorge-david-enciso-mart%C3%ADnez-149977265/)
- [GitHub](https://github.com/Jorgedavyd)
- Email: jorged.encyso@gmail.com

## Citation
Upcoming...
```
@misc{gsn,
  author = {Jorge Enciso},
  title = {GSNs: Gram-Schmidt Networks},
  howpublished = {\url{https://github.com/Jorgedavyd/Gram-Schmidt-Networks}},
  year = {2024}
}
```
