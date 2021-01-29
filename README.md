# Bessel functions in Pytorch

### Jean-Remy Conti, 2021

Pytorch implementation of modified Bessel functions (only of the 1st kind for now i.e. I(nu, z)). Alternative libraries for this kind of computation, as `scipy`, are quite limited. For nu >= 500, `scipy.special.ive`cannot even generate a non-NaN finite number for I(nu, z) when z < 100 [[1]](#1). This repo is mainly based on `scipy` but extends its definition domain using accurate numerical approximations, involving ratios of Bessel functions. Those ratios can be computed in a much more precise way than libraries such as `scipy`do, in addition to be suited for parallel computing with GPUs [[2]](#2).

<p align="center">
  <img src="https://github.com/JRConti/Bessel-functions-Pytorch/blob/main/images/methods_high_nu.png">
</p>




## References
<a id="1">[1]</a> 
Changyong Oh et al. (2019). 
[Radial and Directional Posteriors for Bayesian Neural Networks](https://arxiv.org/pdf/1902.02603.pdf).
AAAI Conference on Artificial Intelligence (AAAI-20).

<a id="2">[2]</a> 
Diego Ruiz-Antolin et al. (2016). 
[A new type of sharp bounds for ratios of modified Bessel functions](https://arxiv.org/pdf/1606.02008.pdf).
Journal of Mathematical Analysis and Applications, Volume 443, Issue 2, 15 November 2016, Pages 1232-1246.
