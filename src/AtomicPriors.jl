module AtomicPriors

using Underscores
export @_

#===== Matrices =====#

using WeightedArrays
export Weighted, WeightedMatrix, array, weights, trace, trim, sortcols, AbsMat, AbsVec
export sobol, soboln, sobolnp, wrand, wrandn, wrandnp, wgrid, xgrid, near, normalise, unclamp

#===== Models =====#

include("mod/simple.jl")
export coin, gauss

include("mod/ymap.jl")
export yexp, loglog, jlog_exp, jlogpost_exp, jlog_vdm, jlogpost_vdm
export flogpost_exp, flog_exp, nlog, nlogpost_exp

#===== Measures =====#

include("info/mutual.jl")
export entropy, mutual, fbayes, mjoint, mcond, mbottle, mpred, fcond

include("info/repeat.jl")

include("info/posterior.jl")
export post, gpost, predict, findcol, fpost, evidence, gevidence

include("info/kernel.jl")
export ekern, mkern, ejoint, econd, ebottle, fkern

include("info/monte.jl")
export mcarlo, fcarlo

#===== Maximisers =====#

include("max/optim.jl")
export optim!, woptim!

include("max/adam.jl")
export adam!, wadam!

include("max/nlopt.jl")
export nlopt!, wnlopt!

include("max/sample.jl")
export metropolis, emcee, mitchell, transtrum, transtrum_exp

#===== Making pictures? =====#

using Requires
@init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
    using .Plots
    include("plots.jl")
end
export pplot, pplot!, yplot, yplot!

end # module AtomicPriors
