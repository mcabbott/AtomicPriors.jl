
using TensorCast, Einsum

using Statistics

"""
    y(Π) = yexp(Π, [1,2])

Map the prior `Π` to data space `y` of sum-of-exponentials model, with equal
amplitudes, for given list of times. Default `times = [1,ℯ]` as in the paper.

* If the prior has `θ_μ ∈ [0..1]`, like `Π = wrand(2,20)`,
  then decay constants are `k_μ = -log(θ_μ) ∈ [0,∞]`.
* If the prior has unbounded `θ_μ`, like `Π = wrandn(2,20)`,
  then decay constants are `k_μ = exp(θ_μ)`.
"""
function yexp(theta::Weighted, times::AbsVec = [1,ℯ])
    if theta.opt.clamp
        theta.opt.lo == 0 && theta.opt.hi == 1 || error("yexp doesn't understand these limits")
        array = yexp_compact(theta.array, times)
    else
        array = yexp_noncompact(theta.array, times)
    end
    Weighted(array, theta.weights, clamp(aname(theta.opt, "y"))) # result y always in [0,1]
end

yexp_compact(theta::AbsMat, times::AbsVec) =
    @reduce yexp[t,a] := mean(i) theta[i,a] ^ times[t]

yexp_noncompact(theta::AbsMat, times::AbsVec) =
    @reduce yexp[t,a] := mean(i) exp(-exp(theta[i,a]) * times[t])

"""
    yexp(θ, [1,2])
    θ |> yexp([1,2])

Map one `θ::Vector` to data space. By default expects unbounded θ;
for `θ ∈ [0,1]` use `compact=true`.
"""
function yexp(theta::AbsVec, times::AbsVec; wrap=false, compact=false)
    wrap==false || error("option wrap=true removed as it was type-unstable!")
    v = compact ?
        [ mean(theta .^ t) for t in times ] :
        [ mean(@. exp( -exp(theta) * t) ) for t in times ]

    # Returns a vector, unless `wrap=true` makes a one-col WeightedMatrix.
    # but that wasn't type-stable!
    # !wrap ?
    #     v :
    #     Weighted(reshape(v,:,1), [1.0], WeightOpt(clamp=true, aname="y"))
end
yexp(times::AbsVec; kw...) = θ -> yexp(θ, times; kw...)

function yexp!(out::AbsVec, theta::AbsVec{T}, times::AbsVec) where {T}
    id = 1/T(length(theta))
    @fastmath @einsimd out[t] = exp(-exp(theta[μ]) * times[t]) * id
end

"""
    Π_R = loglog(Π_01)

This converts a prior with `θ_μ ∈ [0..1]` to one with the parameterisation `θ_μ ∈ ℝ`,
cut off at `|θ_μ|<bound`, keyword `bound=10` by default.
Both should give the same result under `yexp(Π, times)`.
"""
function loglog(prior::Weighted; bound=10)
    prior.opt.clamp && prior.opt.lo == 0 && prior.opt.hi == 1 || error("loglog expected a prior in 0...1 box")
    theta = @. clamp(log(-log(prior.array)), -bound, bound)
    Weighted(theta, prior.weights, unclamp(prior.opt))
end

#===== Jeffreys prior =====#

"""
    θ |> jlog_exp(times)
    θ |> jlog_exp(T, times)

Log Jeffreys density `log(sqrt(det(g_μν)))` for sum of exponentials model
described by `yexp(Π, times)`,
with unbounded `θ = log.(k) ∈𝐑ᵈ` such as a column of `Π = wrandn(d,k)`.

`jlog(big, 1:3)` uses `BigFloat` for the calculation.

Curried form is only for convenience, as no work can be re-used.

See also `jlog_vdm`.
"""
jlog_exp(times::AbsVec) = jlog_exp(Float64, times)

jlog_exp(T::F, times::AbsVec) where {F} = theta -> begin #  where {F} forces specialisation
    jac = jacobi_exp(convertarray(T,theta), convertarray(T,times))
    fish = jac' * jac
    if T isa typeof(big) || (T isa DataType && T <: BigFloat) # using logabsdet bigfloats give a crash?
        detg = LinearAlgebra.det(fish)
        detg>0 ? Float64(log(detg)/2) : -Inf
    else
        logdetg, s = LinearAlgebra.logabsdet(fish)
        ifelse(s>0, Float64(logdetg)/2, -Inf)
    end
end

# convertarray(T, x::AbstractArray) = map(T, x) # map preserves ranges, gives problems
convertarray(T, x::AbstractArray) = T.(x) # broadcast does not
convertarray(::Type{T}, x::AbstractArray{T}) where {T} = x

"""
    jacobi_exp(θ, times)

This is the matrix `dy_t / dθ_μ`, for unbounded θ parameter vector of `yexp`.
"""
function jacobi_exp(theta::AbsVec{T}, times::AbsVec) where {T}
    exptheta = exp.(theta) # d + d^2 * m calls to exp(), vs. 2 * d^2 * m
    id = 1/T(length(theta))
    @einsum jacobi[t,μ] := -times[t] * exp(theta[μ] - exptheta[μ] * times[t]) * id
    # @vielsum is a bad idea since KissMCMC has its own threaded loop.
end
function jacobi_exp!(jacobi::AbsMat, theta::AbsVec{T}, times::AbsVec; cache=similar(theta)) where {T}
    exptheta = cache .= exp.(theta)
    id = 1/T(length(theta))
    @fastmath @einsimd jacobi[t,μ] = -times[t] * exp(theta[μ] - exptheta[μ] * times[t]) * id
end

"""
    θ |> jlogpost_exp(x, σ, times)
    θ |> jlogpost_exp(T, x, σ, times)

Use this to sample from posterior `p_J(θ|x)` arising from Jeffreys prior `p_J(θ)`,
for the model `yexp(Π, times)` + gaussian noise `σ`.
Note that `x` is in data space, `length(x)==length(times)==size(yexp(Π, times),1)`.

Thus `emcee(jlogpost_exp(x,σ,times), Π0, N)` would be like `gpost(x, yexp(times), ΠJ, σ)`
if you had sufficient sampling of `ΠJ = emcee(jlog_exp(times), Π0, N)` near to `x`.

Version with `T` calls `jlog_exp` with this type.

Keyword `alpha=1` (default) gives the posterior density. Smaller values change
the power, down to `alpha=0` which gives the prior instead.

See `flogpost_exp` for flat-prior version.
"""
jlogpost_exp(x::AbsVec, σ::Real, times::AbsVec; kw...) = jlogpost_exp(Float64, x, σ, times; kw...)

function jlogpost_exp(T::F, x::AbsVec, σ::Real, times::AbsVec; alpha::Real=true) where {F}
    @assert length(x)==length(times) "data point x must be in same number of dimensions as y(θ)"
    T == Float64 && @warn "jlogpost_exp gave me some awful answers, because of accuracy... try with Double64 or BigFloat to be sure!" maxlog=3
    θ -> jlog_exp(T, times)(θ) - alpha * (0.5/σ^2) * sum(abs2, x .- yexp(θ, times))
end

"""
    θ |> jlog_vdm(times)

Log Jeffreys density for sum of exponentials model, but unlike `jlog_exp`
this needs as many times as parameters `m==d`,
evenly spaced `times::AbstractRange` with `first(times)==1`.

Works out `abs(det(J))` instead of `sqrt(det(g))`, and `J` is close enough to Vandermonde.

Gives good results with Float64 where `jlog_exp` needs to be using BigFloat,
and is thus much faster. Extending to `m>d` would involve some
messing with Schur polynomials.
"""
jlog_vdm(times) = jlog_vdm(Float64, times)
jlog_vdm(T::F, times::AbstractRange) where {F} = θ -> begin # where {F} forces specialisation
    first(times) == 1 || error("times must start at 1")
    d, m = length(θ), length(times)
    m == d || error("number of parameters must match number of times")
    tstep = T(step(times))
    θpow = @. exp(-tstep * exp(T(θ))) # does 2d calls to exp, and d later, could combine.
    # negs = 0
    out = 0.0
    @inbounds for i in 1:d
        for j in i+1:m
            delta = θpow[j] - θpow[i]
            # negs += delta<0 # det(J) gets squared, so negative is fine
            out += log(abs(delta)) |> Float64 # Vandermonde term
        end
    end
    for j in 1:m
        out += log(T(times[j])/d) |> Float64 # diagonal factor
    end
    for i in 1:d
        out += θ[i] - exp(T(θ[i])) |> Float64 # change of variables
    end
    out
end

"""
    θ |> jlogpost_vdm(T, x, σ, times)

Just like `jlogpost_exp`, but using `jlog_vdm` obviously.
"""
jlogpost_vdm(x::AbsVec, σ::Real, times::AbsVec; kw...) = jlogpost_vdm(Float64, x, σ, times; kw...)
function jlogpost_vdm(T::F, x::AbsVec, σ::Real, times::AbsVec; alpha::Real=true) where {F}
    @assert length(x)==length(times) "data point x must be in same number of dimensions as y(θ)"
    θ -> jlog_vdm(T, times)(θ) - alpha * (0.5/σ^2) * sum(abs2, x .- yexp(θ, times))
end

#===== Flat & log-normal priors =====#

"""
    θ |> flogpost_exp(x, σ, times)

Use this to sample from posterior `p_F(θ|x)` arising from a flat prior `p(θ) ∝ 1`,
for the model `yexp(Π, times)` + gaussian noise `σ`. Compare `jlogpost_exp` for Jeffreys prior.

Flat meaning flat on the noncompact `θ=log(k)` parameters by default.
`flogpost_exp(x, σ, times, :k)` is flat in decay rate instead,
and with `:T` flat in lifetime.
"""
function flogpost_exp(x::AbsVec, σ::Real, times::AbsVec, var::Symbol=:θ)
    @assert length(x)==length(times) "data point x must be in same number of dimensions as y(θ)"
    mi2σ = -0.5/σ^2
    if var === :θ
        θ ->  mi2σ * sum(abs2, x .- yexp(θ, times))
    elseif var === :k
        θ -> +sum(θ) + mi2σ * sum(abs2, x .- yexp(θ, times))
    elseif var === :T
        θ -> -sum(θ) + mi2σ * sum(abs2, x .- yexp(θ, times))
    else
        error("expected symbol in [:θ, :k, :T]")
    end
end

"""
    θ |> flog_exp(:k)

Log density for sampling from a flat prior, for model `yexp(Π, times)`.

Works in terms of the noncompact `θ=log(k)`, which should be restricted by e.g. using `clamp(wrandn(5,10),-5,5)` as the initial points for `emcee`.

With symbol `:k` it is flat in the decay rate, symbol `:T` flat in lifetime instead, default is `:θ`.

Compare `jlog_exp(times)` for Jeffreys prior, and `flogpost_exp(x, σ, times)` for posterior.
"""
function flog_exp(var::Symbol=:θ)
    if var === :θ
        θ -> 1.0
    elseif var === :k
        θ -> +sum(θ)
    elseif var === :T
        θ -> -sum(θ)
    else
        error("expected symbol in [:θ, :k, :T]")
    end
end

"""
    θ |> nlog(μ=0, σ=1)

Log density for sampling from normal distribution, `θ ~ 𝒩(μ, σ²)`.
Used with `yexp(Π, times)`, this amounts to a log-normal distribution
for each decay rate `k = exp(θ)`, using `μ=0.69` moves the centre up to `k=2`.
NB this `σ` is the width of the prior, unrelated to the model's noise!

Compare `jlog_exp(times)` for Jeffreys prior, and `flog_exp(:k)` for flat in `k`.
"""
nlog(μ=false, σ::Number=true) = θ -> sum(abs2.(θ .- μ) .* (-1/(2σ^2)))

# """
#     θ |> nlogpost(x, σ_mod, μ=0, σ_prior=1)

# Use this to sample from posterior `p_N(θ|x)` arising from
# a normal prior `p_N(θ)` a.k.a. `nlog(μ, σ_prior)`.
# NB `σ_mod` is the noise of the model, unrelated to the width of the prior.
# """
# nlogpost(x::AbsVec, σ_model, μ_prior=false, σ_prior::Number=true) =
#     θ -> sum(abs2.(θ .- x).*(-1/(2σ_model^2)) .+ abs2.(θ .- μ_prior).*(-1/(2σ_prior^2)))

"""
    θ |> nlogpost_exp(x, σ_mod, times, μ=0, σ_prior=1)

Use this to sample from posterior `p_N(θ|x)` arising from
a normal prior `p_N(θ)` a.k.a. `nlog(μ, σ_prior)`.
NB `σ_mod` is the noise of the model, unrelated to the width of the prior.

Keyword `alpha=1` (default) gives the posterior density. Smaller values change
the power, down to `alpha=0` which gives the prior instead.
"""
nlogpost_exp(x::AbsVec, σ_model, times::AbsVec, μ_prior=false, σ_prior::Number=true; alpha=true) =
    θ -> begin
        y = yexp(θ, times)
        alpha * sum(abs2.(y .- x).*(-1/(2σ_model^2))) + sum(abs2.(θ .- μ_prior) .* (-1/(2σ_prior^2)))
    end


#===== Faster =====#

using Einsum

using ArrayAllez

using Tracker: Tracker, TrackedMatrix, track, @grad

yexp_compact(theta::TrackedMatrix, times::AbsVec) = track(yexp_compact, theta, times)

@grad function yexp_compact(theta::TrackedMatrix{T}, times::AbsVec) where T # combined fwd+back
    d,k = size(theta)
    nt = length(times)
    id = 1/d
    θdata = theta.data

    powersave = Array_{T}(:powersave, nt,d,k)
    @vielsum powersave[t,i,a] = θdata[i,a] ^ (times[t]-1) # do the expensive bit once

    @einsum yexp[t,a] := id * powersave[t,i,a] * θdata[i,a] # Σ_i

    function back(Δ) # I think this is not hitting closure bug
        Δdata = Tracker.data(Δ)
        @einsum ∇[i,a] := id * Δdata[t,a] * times[t] * powersave[t,i,a] # Σ_t
        (∇, nothing)
    end

    yexp, back
end

