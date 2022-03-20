# AtomicPriors.jl

This package contains software related to the following papers:

<!--
1. *Far from Asymptopia* <br/>
Michael Abbott & Benjamin Machta <br/>
[arXiv:2204.XXXXX]()
-->

1. *Information geometry for multiparameter models: New perspectives on the origin of simplicity* <br/>
Katherine Quinn, Michael Abbott, Mark Transtrum, Benjamin Machta, James Sethna <br/>
[arXiv:2111.07176](http://arxiv.org/abs/2111.07176)

2. *A Scaling Law From Discrete to Continuous Solutions of
Channel Capacity Problems in the Low-Noise Limit* (a.k.a. *An information scaling law: ζ = 3/4*) <br/>
Michael Abbott & Benjamin Machta <br/>
[J. Stat. Phys. **176** (2019) 214–227]() <!-- [arXiv:1710.09351](https://arxiv.org/abs/1710.09351) -->

3. *Maximizing the information learned from finite data selects a simple model* (a.k.a. *Rational Ignorance*) <br/>
Henry Mattingly, Mark Transtrum, Michael Abbott, Benjamin Machta <br/>
[PNAS **115** (2018) 1760-1765](https://doi.org/10.1073/pnas.1715306115) <!-- ≈ [arXiv:1705.01166](https://arxiv.org/abs/1705.01166) -->

(All of this code post-dates the 2018 PNAS paper.)

### Installation

You will need at Julia 1.6 or later, freely available from [julialang.org](https://julialang.org/downloads/).
These commands will install the package, and all of its dependencies:

```julia
using Pkg  # Julia's built-in package manager
Pkg.add(url="https://github.com/mcabbott/AtomicPriors.jl")
Pkg.add("Plots")
using AtomicPriors, Plots
```

The basic use is shown in some noebooks in the `/docs/` folder,
which can be viewed nicely online at [...github.io...basic.html](https://mcabbott.github.io/AtomicPriors.jl/docs/basic.html).

In case this prompts anyone to learn Julia, [these lectures](https://julia.quantecon.org/intro.html) were helpful (the first few),
and [this page](https://docs.julialang.org/en/v1/manual/noteworthy-differences/index.html) lists differences from Matlab (and Python, R).

### Author

Michael Abbott, uploaded March 2022
