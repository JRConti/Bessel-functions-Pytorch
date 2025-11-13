# Bessel functions in Pytorch

### Jean-Remy Conti, 2021

Pytorch implementation of modified Bessel functions (only of the 1st kind for now i.e. I(nu, z)). Alternative libraries for this kind of computation, as `scipy`, are quite limited. For nu >= 500, `scipy.special.ive`cannot even generate a non-NaN finite number for I(nu, z) when z < 100 [[1]](#1). This repo is mainly based on `scipy` but extends its definition domain using accurate numerical approximations, involving ratios of Bessel functions. Those ratios can be computed in a much more precise way than libraries such as `scipy` do [[1](#1),[2](#2)], in addition to be suited for parallel computing with GPUs.

<p align="center">
  <img src="https://github.com/JRConti/Bessel-functions-Pytorch/blob/main/images/methods_high_nu.png">
</p>

## How to use ?

For numerical reasons, the logarithm of Bessel functions is rather considered.

* Import the functions:

```from logbessel_I import logbessel_I, Ak_approx```

* Use documentation:

The following function computes the logarithm of the modified Bessel function of the 1st kind I(nu, z). It uses the `scipy` method and extends it when it is not tractable, using 2 different methods (1 slow/precise, 1 fast/rough approximation). Note that, contrary to `scipy`, you can pass a tensor of shape (N,) as the z argument.
```
def logbessel_I(nu, z, fast = False, check = True):
        Parameters
	----------
	nu: positive int, float
		Order of modified Bessel function of 1st kind.
	z: int/float or tensor, shape (N,) 
		Argument of Bessel function.
	fast: bool
		If True, use asymptotic behavior as approximation when main 
		scipy method is not tractable. If False, use tight bounds for 
		the ratio of Bessel functions:
		https://arxiv.org/pdf/1902.02603.pdf
	check: bool
		If True, check if argument of log is non zero and not NaN.
    	
	Return
	------
	result: tensor, shape (N,)
```

The following function computes ratios of the form I(nu, z)/I(nu-1, z). It is much more accurate than using `scipy`[[1](#1),[2](#2)], in addition to be a way simpler/faster computation suited for parallel computing with GPUs. Note that both nu and z can be tensors.
```
def Ak_approx(nu, z):
	Approximation of ratio of modified Bessel functions of 1st kind.

        Parameters
	----------
	nu: tensor, shape (N0,)
		Order of modified Bessel functions of 1st kind.
	z: tensor, shape (N1,) 
		Argument of Bessel function. Positive values only.
	
	Return
	------
	tensor, shape (N1, N0)

```

## Details

#### Why I(nu, z) is difficult to compute ?

As explained in [[1]](#1), for any finite C > 0 and any delta > 0, there exists nu such that for any z in [0,C], I(nu, z) < delta. Intuitively, a larger nu makes I(nu, z) arbitrarily small on longer intervals.
The ratio of Bessel functions is thus also difficult to compute.

#### Asymptotic approximation

A fast but not so accurate way to compute log[I(nu, z)] is to use asymptotic behavior.

#### Approximation of Bessel ratios

[[2]](#2) find very simple and tight bounds for the ratio I(nu, z)/I(nu-1, z) which are used in [[1]](#1) to approximate those ratios in a much better way than with libraries such as `scipy`. This computation allows to compute I(nu, z) using telescoping products.

The quality of the approximation is quantified here using the relative error with respect to `scipy` methods (when they are tractable).

<p align="left">
  <img src="https://github.com/JRConti/Bessel-functions-Pytorch/blob/main/images/ratio_approximation_error_low_nu.png" width="400">
  <img src="https://github.com/JRConti/Bessel-functions-Pytorch/blob/main/images/ratio_approximation_error_high_nu.png" width="420">
</p>

For nu = 1000 (right), the `scipy`computation is not tractable for z <~ 600. 

#### Computation time:

Although the previous approximation is very accurate, the complexity of the computation is proportional to nu (telescoping product [[1]](#1)). 

As an example (which can be run with main script of `logbessel_I.py`), here are the computation times for nu = 1000 and 1 <= z <= 200 000 (z is a tensor of shape (200 000,)) on a simple CPU:

* Pytorch adaptation of `scipy` (function `logbessel_I_scipy`): **0.21s** (can produce NaN values for low z values)
* Asymptotic computation (function `logbessel_I_asymptotic`): **0.01s** (not so accurate but fast)
* Approximation with ratios of Bessel functions (function `logbessel_I_approx`): **11.60s** (precise and slow)

#### Proposed method:

Because of the computation time of the approximation via ratios of Bessel functions, the main function `logbessel_I` copies the Pytorch adaptation of `scipy`. When the `scipy` method does not output a finite result, 2 methods are proposed via the option `fast`:
 
- `fast = True`: use asymptotic computation (not really accurate)
- `fast = False`: use approximation with ratios of Bessel functions (accurate but slow)

Here are the results of the computation of function `logbessel_I` with `fast = False` (green dashed plot), alongside the `scipy`(blue curve) and asymptotic (orange dashed curve) methods:
<p align="left">
  <img src="https://github.com/JRConti/Bessel-functions-Pytorch/blob/main/images/methods_low_nu.png" width="400">
  <img src="https://github.com/JRConti/Bessel-functions-Pytorch/blob/main/images/methods_high_nu.png" width="400">
</p>

For low z values (left), the `scipy` version is always defined so our function is identical. There are 2 asymptotes (around 0 and for +inf).

For higher z values (right), the `scipy`version is not defined for z <~600, so the approximation using ratios of Bessel functions extends it. There is only 1 asymptote that is well defined in this case.

As an example (which can be run with main script of `logbessel_I.py`), here are the computation times for nu = 1000 and 1 <= z <= 200 000 (z is a tensor of shape (200 000,)) on a simple CPU:

* Pytorch adaptation of `scipy` (function `logbessel_I_scipy`): **0.21s** (can produce NaN values for low z values)
* Asymptotic computation (function `logbessel_I_asymptotic`): **0.01s** (not so accurate but fast)
* Approximation with ratios of Bessel functions (function `logbessel_I_approx`): **11.60s** (precise and slow)

* Fast proposed method (function `logbessel_I` with `fast = True`): **0.15s** (fast extension of `scipy`)
* Accurate proposed method (function `logbessel_I` with `fast = False`): **0.18s** (precise extension of `scipy`) 


## References
<a id="1">[1]</a> 
Changyong Oh et al. (2019). 
[Radial and Directional Posteriors for Bayesian Neural Networks](https://arxiv.org/pdf/1902.02603.pdf).
AAAI Conference on Artificial Intelligence (AAAI-20).

<a id="2">[2]</a> 
Diego Ruiz-Antolin et al. (2016). 
[A new type of sharp bounds for ratios of modified Bessel functions](https://arxiv.org/pdf/1606.02008.pdf).
Journal of Mathematical Analysis and Applications, Volume 443, Issue 2, 15 November 2016, Pages 1232-1246.

## Meta

Jean-Rémy Conti – jeanremy.conti@gmail.com

Distributed under the GNU license. See LICENSE for more information.


## Contributing

1. Fork it (https://github.com/JRConti/Bessel-functions-Pytorch/fork)
2. Create your feature branch: `git checkout -b feature/fooBar`
3. Commit your changes: `git commit -am 'Add some fooBar'`
4. Push to the branch: `git push origin feature/fooBar`
5. Create a new Pull Request
