### A Pluto.jl notebook ###
# v0.18.2

using Markdown
using InteractiveUtils

# ╔═╡ 946d09e8-983f-11ec-071f-51a9e513da79
begin
	using Pkg
	Pkg.add(url="https://github.com/mcabbott/AtomicPriors.jl"; io=devnull)
	Pkg.add("Plots")
end

# ╔═╡ c1ba5b7d-afc9-4595-a0a5-cb5b21170028
using AtomicPriors, Plots

# ╔═╡ e087b505-7607-4e31-b196-297324ddad74
md"""
# [AtomicPriors.jl](https://github.com/mcabbott/AtomicPriors.jl)

This notebook demonstrates some basic use of this package.

You must install Julia to use it (at least 1.6, from [julialang.org](https://julialang.org/downloads/)). You don't have to install [Pluto.jl](https://github.com/fonsp/Pluto.jl), but this notebook is an easy way to show plots and code together.

This package isn't registered, so must be added using its address not just its name:
"""

# ╔═╡ ff3a92ad-ae33-4a25-a2e8-fffc80ed705b
md"""
### Unfair coins

Here's the optimal prior for a coin being flipped 10 times:
"""

# ╔═╡ dd903702-5aff-48bd-8617-54c60392bf58
nlopt!(p -> mutual(coin(p,10)), sobol(1,30)) |> plot

# ╔═╡ f8d0f4b8-ca05-4229-8ef4-0f982018acb8
md"""
To explain what's goin on, first note that all priors are going to be represented using a `Weighted` matrix type. With ``d=1`` parameter this has one row, and each column is the location ``\theta_a`` of an atom:
``
p(\theta) = \sum_a \lambda_a \delta(\theta-\theta_a)
``.

The struct stores alongside this matrix a vector of weights ``\lambda_a``, and some auxilliary information like the fact that ``0 \leq \theta_a \leq 1``. The initial condition could be random, but is in fact a sub-random Sobol sequence:
"""

# ╔═╡ d0d13ffa-f144-48fd-a764-425049711e10
sobol(1,4)

# ╔═╡ 1c01bdb9-9117-42a5-9afd-27ca3ccc5412
md"""
Next, `coin` is the Bernoulli problem. It returns a likelihood matrix ``p(x|\theta_a)`` for each outcome `x` and each atom ``\theta_a``. If we flip twice, this looks like:
"""

# ╔═╡ 65bd0afe-f882-41f5-97fc-094d4adf0b0c
coin(sobol(1,4), 2)

# ╔═╡ 9c83367b-7390-4706-a838-4f6e827e034e
md"""
It returns another `Weighted` struct because it's convenient to keep the ``\lambda_a`` attached. Together, these are enough to compute mutual information ``I(X;\Theta)``:
"""

# ╔═╡ d5324d1c-e802-4b32-9927-01d8ff7edbb4
mutual(coin(sobol(1,4), 2))

# ╔═╡ ef06284b-200b-4443-ab76-9648b6d9aa5d
md"""
To find the optimum, the function `nlopt!(f, p)` uses LBFGS to maximise `f(p)` by adjusting `p`. This is done respecting the constraints ``0 \leq \theta_a \leq 1`` and ``\lambda_a \geq 0``, and always normalising ``\sum_a \lambda_a = 1``. 

Atoms with no weight at the end are deleted, and those which co-incide are combined, so that the object returned has just five columns.

The first argument `p -> mutual(coin(p, 10))` is an anonymous function, which applies the steps above:
"""

# ╔═╡ 54785e32-40fd-4825-a999-f34c526df2f0
nlopt!(p -> mutual(coin(p,10)), sobol(1,30))

# ╔═╡ 5f17dbdf-71dd-4561-9412-93cb46dcfa0d
md"""
Finally, there is a plot recipe for `Weighted`, which for ``d=1`` prints what's shown above. For ``d=2``, below, the axes are components ``\theta_1`` and ``\theta_2``, with the area of the points ``\lambda``, for each atom.
"""

# ╔═╡ 558e049d-b347-465f-92f1-2a759cf75797
md"""
### Gaussian noise

The other one-dimensional model has Gaussian noise. One way to handle this is to discretise the space of outcomes ``X``, which looks like this:
"""

# ╔═╡ 246cc427-a996-4e29-8060-9b1515069957
gauss(xgrid(1, -0.1:0.1:1.1), sobol(1,4), 0.1)

# ╔═╡ 7d445a60-a6cb-4dfa-8e3a-bab7adaba372
nlopt!(p -> entropy(gauss(xgrid(1, -0.1:0.1:1.1), p, 0.1)), sobol(1, 30)) |> plot

# ╔═╡ 95f703cd-5b24-4752-9d29-b3472f668b7d
md"""
Since S(X|Θ) is constant here, it shouldn't matter whether we maximise `mutual` or `entropy` which is ``S(X)``.

But a grid isn't going to work well in higher dimensions. What does work well is a KDE approximation:"""

# ╔═╡ a11e642a-d5b8-47fe-8ca5-5f8aad9de47c
nlopt!(p -> ekern(p, 0.1 * √2), sobol(1, 50)) |> plot

# ╔═╡ df71a3fa-4f06-4146-bdc1-a4ab7edd118b
md"""
### Radioactive decay model

The simplest interesting multi-parameter model is for exponential decay. We observe 
``
y_t(\theta) = \sum_{\mu=1}^d \frac{e^{-k_\mu t}}{d}
``
at some times ``t``, with some Gaussian noise.

The function `yexp(prior, times)` implements this. With ``d=2`` parameters the prior should have 2 rows, and with ``t = 1, 3`` the output does too:
"""

# ╔═╡ a60f52c6-88c9-441e-aa96-debf5f46ea01
yexp(sobol(2, 1000), [1,3]) |> plot

# ╔═╡ 1dbce085-5b88-4c1e-9ad2-18c0dd6e2f27
md"""
We can optimise the prior using this:
"""

# ╔═╡ b7c21355-3398-4a52-97e6-21d7aea0ba04
begin
	pstar = nlopt!(p -> ekern(yexp(p, [1,3]), 0.03 * √2), sobol(2,100))
	yexp(pstar, [1,3]) |> unique |> plot
end

# ╔═╡ ba96aff9-b16e-4e68-8336-10caa1f972c5
md"""Here's how to sample from Jeffreys prior for this model:"""

# ╔═╡ 094dbb75-e3d2-4c95-a587-5ddd54df159a
begin 
	pjeff = emcee(jlog_exp([1,3]), soboln(2,10), 10^4)
	plot(yexp(pjeff, [1,3]), c=:red)
end

# ╔═╡ 56e037fa-fcd5-4de5-b63d-f74cea94d91a
md"""
Note that `yexp` understands two parameterisations. `sobol(2,10)` as before are points in ``0\leq\theta\leq 1``, like `rand`, while `soboln(2,10)` has unconstrained values, initially normally distributed, like `randn`. For the first, the decay rates are ``k_\mu = -\log \theta_\mu``, while for the second, ``k_\mu = \exp \theta_\mu``. Optimising with `nlopt!(p -> ekern(yexp(p, [1,3]), 0.05), soboln(2,100))` will work about equally well here, although larger cases work better with the first, compact, parameterisation.

But `jlog_exp` understands only the second parameterisation. In fact `jlog_exp([1,3])` is a function which expects a vector, not a `Weighted` matrix, so it can't warn you about this.
"""

# ╔═╡ c4ec280a-eae9-4432-adf6-c5369b4f51c1
let
	pstar_noncompact = nlopt!(p -> ekern(yexp(p, [1,3]), 0.05), soboln(2,100))
	plot(pstar_noncompact)
	plot!(pjeff, c=:red, size=(300, 300))
end

# ╔═╡ 0aa87e09-b848-47db-88eb-b7ba159bb30a
md"""
The 3rd prior considered in the review comes from projecting each point ``x`` in ``p_\mathrm{NML}(x)`` back to its maximum-likelihood point ``\hat\theta_x``. Like the optimal prior, this depends on ``\sigma``. It can be sampled as follows:
"""

# ╔═╡ 5648ba5d-fd00-4fc5-903a-ce1792e1549b
begin
	pproj = transtrum(p -> yexp(p, [1,3]), soboln(2, 20), 0.03, 10^4)
	plot(yexp(pproj, [1,3]), c=:green)
end

# ╔═╡ 8968c945-2e61-462c-bc1b-5b712ab35736
md"""
While we use KDE entropy (or rather, its gradient) to find `pstar`, this does not give an accurate value for the mutual information. For this, Monte Carlo sampling gives an unbiased estimate:
"""

# ╔═╡ 99a9877f-2d7b-4a25-b9c0-68bd0fea00bc
mcarlo(yexp(pstar, [1,3]), 0.03, 10^5) / log(2)  # I(X;Θ) in bits

# ╔═╡ 971d5057-9534-4ffa-89aa-d17279c121cd
mcarlo(yexp(pjeff, [1,3]), 0.03, 10^5) / log(2)

# ╔═╡ 0b62d09a-290e-4d5d-b244-64f1435c3608
mcarlo(yexp(pproj, [1,3]), 0.03, 10^5) / log(2)

# ╔═╡ f057120e-e24c-4c96-9f4b-bb4dc8f06f69
md"""
### Minimax condition

Instead of finding ``p_\star(\theta)`` from ``\max_{p(\theta)} I(X;\Theta)``, we can also find it from ``\min_{p(\theta)} \max_\theta f_\mathrm{KL}(\theta)``. That's harder for the computer to solve, but easy to plot afterwards, and perhaps helpful for intuition.

Back in one dimension, flipping a coin 10 times:
"""

# ╔═╡ b26e1e3a-e633-4981-9fb3-7a32081b9313
begin
	p10 = nlopt!(p -> mutual(coin(p, 10)), sobol(1, 30))
	plot(p10, lab="optimal prior")

	mi10 = mutual(coin(p10, 10))
	@info "I [bits]" mi10 / log(2)

	ts = wgrid(1, 0:0.01:1)
	fs = fbayes(coin(ts, 10), coin(p10, 10))
	plot!(vec(ts.array), fs ./ 5, lab="f_KL, scaled to fit on plot")
	plot!(vec(ts.array), fill(mi10, 101) ./ 5, lab="I(X;Θ), scaled")
end

# ╔═╡ 4f706555-344c-4c00-9b04-9b0f7bb3e4c2
p10 |> sortcols

# ╔═╡ a2754301-4466-4c42-aa90-a35a8d57913b
md"""
The green line has exactly the same height at every atom, and is lower elsewhere. So the game is balanced, every move by your Opponent (who chooses ``\theta``) costs you the same amount.

If we perturb the prior, this is no longer true:
"""

# ╔═╡ 6ef68f48-660d-4e11-a64b-7cf64bf682e8
let
	p10copy = copy(p10)
	p10copy.array[end] += 0.1	
	plot(p10copy, lab="broken optimal prior")

	mi = mutual(coin(p10copy, 10))
	@info "I [bits]" mi / log(2)

	fs = fbayes(coin(ts, 10), coin(p10copy, 10))
	plot!(vec(ts.array), fs ./ 5, lab="f_KL, scaled")
	plot!(vec(ts.array), fill(mi, 101) ./ 5, lab="I(X;Θ), scaled")
end

# ╔═╡ a0087f2e-7676-4591-9c92-2a0f295eff36
md"""
And if we approach a uniform prior, it's even less true. Clearly He will pick one of the ends, where you lose heavily:
"""

# ╔═╡ 5a509adb-1fa2-496d-9a09-fc0fb1bc9198
let
	p1uniform = wgrid(1, 0:0.05:1)
	plot(p1uniform, lab="evenly spaced prior", c=:black)

	mi = mutual(coin(p1uniform, 10))
	@info "I [bits]" mi/log(2)

	fs = fbayes(coin(ts, 10), coin(p1uniform, 10))
	plot!(vec(ts.array), fs ./ 5, lab="f_KL, scaled", ylim=[0, 0.5])
	plot!(vec(ts.array), fill(mi, 101) ./ 5, lab="I(X;Θ), scaled")
end

# ╔═╡ fba5c503-943d-43c4-9e53-61d885a0efae
md"""
For this model, the projected maximum likelihood prior (from 2021 review) is also discrete, with exactly 11 points of weight (for 10 coin flips). But these aren't uniform; like the optimal prior it places extra weight on the ends. Thus ``f_\mathrm{KL}(\theta)`` is closer to flat:
"""

# ╔═╡ bfade7c8-dc3d-4d3d-bbc5-bf3506fc67c0
p10proj = transtrum(p -> coin(p, 10), sobol(1, 30)) |> sortcols

# ╔═╡ 8cc59b72-1322-401b-a3f7-8aba4a35dae4
let
	plot(p10proj, lab="projected maximum likelihood prior", c=:darkgreen)

	mi = mutual(coin(p10proj, 10))
	@info "I [bits]" mi / log(2)

	fs = fbayes(coin(ts, 10), coin(p10proj, 10))
	plot!(vec(ts.array), fs ./ 5, lab="f_KL, scaled", ylim=[0, 0.4])
	plot!(vec(ts.array), fill(mi, 101) ./ 5, lab="I(X;Θ), scaled")
end

# ╔═╡ Cell order:
# ╟─e087b505-7607-4e31-b196-297324ddad74
# ╠═946d09e8-983f-11ec-071f-51a9e513da79
# ╠═c1ba5b7d-afc9-4595-a0a5-cb5b21170028
# ╟─ff3a92ad-ae33-4a25-a2e8-fffc80ed705b
# ╠═dd903702-5aff-48bd-8617-54c60392bf58
# ╟─f8d0f4b8-ca05-4229-8ef4-0f982018acb8
# ╠═d0d13ffa-f144-48fd-a764-425049711e10
# ╟─1c01bdb9-9117-42a5-9afd-27ca3ccc5412
# ╠═65bd0afe-f882-41f5-97fc-094d4adf0b0c
# ╟─9c83367b-7390-4706-a838-4f6e827e034e
# ╠═d5324d1c-e802-4b32-9927-01d8ff7edbb4
# ╟─ef06284b-200b-4443-ab76-9648b6d9aa5d
# ╠═54785e32-40fd-4825-a999-f34c526df2f0
# ╟─5f17dbdf-71dd-4561-9412-93cb46dcfa0d
# ╟─558e049d-b347-465f-92f1-2a759cf75797
# ╠═246cc427-a996-4e29-8060-9b1515069957
# ╠═7d445a60-a6cb-4dfa-8e3a-bab7adaba372
# ╟─95f703cd-5b24-4752-9d29-b3472f668b7d
# ╠═a11e642a-d5b8-47fe-8ca5-5f8aad9de47c
# ╟─df71a3fa-4f06-4146-bdc1-a4ab7edd118b
# ╠═a60f52c6-88c9-441e-aa96-debf5f46ea01
# ╟─1dbce085-5b88-4c1e-9ad2-18c0dd6e2f27
# ╠═b7c21355-3398-4a52-97e6-21d7aea0ba04
# ╟─ba96aff9-b16e-4e68-8336-10caa1f972c5
# ╠═094dbb75-e3d2-4c95-a587-5ddd54df159a
# ╟─56e037fa-fcd5-4de5-b63d-f74cea94d91a
# ╠═c4ec280a-eae9-4432-adf6-c5369b4f51c1
# ╟─0aa87e09-b848-47db-88eb-b7ba159bb30a
# ╠═5648ba5d-fd00-4fc5-903a-ce1792e1549b
# ╟─8968c945-2e61-462c-bc1b-5b712ab35736
# ╠═99a9877f-2d7b-4a25-b9c0-68bd0fea00bc
# ╠═971d5057-9534-4ffa-89aa-d17279c121cd
# ╠═0b62d09a-290e-4d5d-b244-64f1435c3608
# ╟─f057120e-e24c-4c96-9f4b-bb4dc8f06f69
# ╠═b26e1e3a-e633-4981-9fb3-7a32081b9313
# ╠═4f706555-344c-4c00-9b04-9b0f7bb3e4c2
# ╟─a2754301-4466-4c42-aa90-a35a8d57913b
# ╠═6ef68f48-660d-4e11-a64b-7cf64bf682e8
# ╟─a0087f2e-7676-4591-9c92-2a0f295eff36
# ╠═5a509adb-1fa2-496d-9a09-fc0fb1bc9198
# ╟─fba5c503-943d-43c4-9e53-61d885a0efae
# ╠═bfade7c8-dc3d-4d3d-bbc5-bf3506fc67c0
# ╠═8cc59b72-1322-401b-a3f7-8aba4a35dae4
