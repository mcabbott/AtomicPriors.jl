
using TensorCast

"""
    ekern(Π,σ)

KDE approximation to gaussian mixture entropy: `S₁(X) = -∑ᵢ λᵢ log(∑ⱼ λⱼ e^-(θᵢ-θⱼ)²/2σ² ) + κ(σ)`,
where `Π::Weighted` contains positions `θᵢ` & weights `λᵢ`, and now `κ(σ) = d/2 log(2πσ²) + d/2`.
The last `d/2` there is a correction to treat the `i==j` terms exactly, no effect on optimisation
obviously.

Since `S(X|Θ)` is a constant, maximising this is equivalent to maximising `I(Χ,Θ)`.

You probably want to use this (and its descendents `econd`, `ejoint`, `ebottle`, and its
cousin `fkern`) with scaled `σ₊=σ*√2`, an approximation which may be called ELK.
"""
ekern(prior::Weighted, sigma::Real) = ekern(array(prior), weights(prior), sigma)

function ekern(cols::AbsMat, lambdas::AbsVec{T}, sigma::Real) where T
    d, k = size(cols)

    mi2v = -0.5/sigma^2 |> T
    kappa = (d/2)*log(2π*sigma^2) + d/2 |> T

    dist2 = pairwise2(cols)

    @reduce innersum[a] := sum(b) lambdas[b] * exp(dist2[b,a] * mi2v)

    kappa - @reduce sum(a) finite(lambdas[a] * log(innersum[a]))
end

"""
    mkern(Π, σ) = ekern(Π,σ) - d/2 (1+log(2πσ²))

KDE approximation to gaussian mixture mutual information, simply `S₁(X) - S(X|Θ)`.
Not a great approximation!
"""
function mkern(prior::Weighted, sigma::Real)
    d,k = size(prior)
    off = (d/2)*(1 + log(2π*sigma^2))
    ekern(prior, sigma) - off
end

"""
    ejoint(Π, σX, σY=σX)
    ejoint(f(Π), σX, fY(Π), σY)

This is `S(XY)`, where `X` and `Y` are independent gaussians about points `θ∈Π`,
and entropy is calculated using KDE. Since they are gaussian, `S(XY|Θ)=const`,
hence optimising this is equivalent to optimising `I(XY;Θ) = mjoint(...)`
modulo the issue of KDE and `σ*√2`.
"""
ejoint(ΠX::Weighted, σX::Real, ΠY::Weighted, σY::Real) =
    ekern(vcat(array(ΠX) ./ σX, array(ΠY) ./ σY), weights(ΠX), sqrt(2))

ejoint(Π::Weighted, σX::Real, σY::Real=σX) = ejoint(Π,σX, Π,σY)

"""
    econd(Π, σX, σY=σX)
    econd(f(Π), σX, fY(Π), σY)

This is `S(XY)-S(X)` calculated using KDE, assuming X,Y are independent gaussians...
and hence maximising this is equivalent to maximising conditional mutual information
`I(Y;Θ|X) = I(XY;Θ) - I(X;Θ) = mcond(...)`, modulo the issue of KDE and `σ*√2`.
"""
econd(ΠX::Weighted, σX::Real, ΠY::Weighted, σY::Real) =
    ekern(vcat(array(ΠX) ./ σX, array(ΠY) ./ σY), weights(ΠX), 1) - ekern(ΠX, σX)

econd(Π::Weighted, σX::Real, σY::Real=σX) = econd(Π,σX, Π,σY)


"""
    ebottle(β, Π, σX, σY=σX)
    ebottle(β, f(Π), σX, fY(Π), σY)

This is `β S(Y) - S(X)`, where X and Y are independent gaussians about points θ∈Π,
and entropy is calculated using KDE. Since they are gaussian, maximising this is
equivalent to maximising the bottleneck measure `β I(Y;Θ) - I(X;Θ) = mbottle(...)`
modulo the issue of KDE and `σ*√2`.
"""
ebottle(β::Real, ΠX::Weighted, σX::Real, ΠY::Weighted, σY::Real) =
    β * ekern(ΠY, σY) - ekern(ΠX, σX)

ebottle(β::Real, Π::Weighted, σX::Real, σY::Real=σX) = ebottle(β, Π,σX, Π,σY)


## d/2 log(2πσ²)`.

"""
    fkern(φ, Π, σ) -> real
    fkern(Φ, Π, σ) -> vector

KDE approximation to Bayes risk function `fbayes()` for a gaussian model:
`f₁(φ) = - log(∑ⱼ λⱼ e^-(φ-θⱼ)²/2σ² ) + κ(σ)`,
where `φ::Vector`, and `Π::Weighted` contains positions `θᵢ` & weights `λᵢ`,
and `κ(σ) = 0` for now.

You probably want to use this (its cousins `ekern`, `econd` et. al.)
with scaled `σ₊=σ*√2`, an approximation which may be called ELK.

Given a list of points `φ`, as columns of `Φ::Matrix`, it returns a vector of f-values.

    fkern(f, Φ, Π, σ) = fkern(f(Φ), f(Π), σ)
    fkern(Π, σ) = fkern(Π, Π, σ)

Or give a function! Or omit `Φ` & it'll use `Π` for that too.
"""
function fkern(phi::AbsMat, prior::Weighted, sigma::Real)
    theta = prior.array
    lambdas = normalise(prior.weights)

    d, k = size(theta)
    dϕ, kϕ = size(phi)
    @assert dϕ==d "dimension of point φ must match that of prior θ"

    mi2v = -0.5/sigma^2
    kappa = 0 # (d/2)*log(2π*sigma^2) # should this have d/2 too?

    dist2 = pairwise2(theta, phi)

    @reduce innersum[a] := sum(b) lambdas[b] * exp(dist2[b,a] * mi2v)

    @. kappa - log(innersum)
end

# get phi into shape
fkern(phi::AbsVec, prior::Weighted, sigma::Real) = only(fkern(reshape(phi,:,1), prior, sigma))
fkern(phi::Weighted, prior::Weighted, sigma::Real) = fkern(phi.array, prior, sigma)

# with a function
fkern(f, phi::Weighted, prior::Weighted, sigma::Real) = fkern(f(phi).array, f(prior), sigma)
fkern(f, phi::AbsVec, prior::Weighted, sigma::Real) = only(fkern(f, WeightedMatrix(phi), prior, sigma))

# without phi
fkern(prior::Weighted, sigma::Real) = fkern(prior.array, prior, sigma)
fkern(f, prior::Weighted, sigma::Real) = fkern(f(prior), sigma)

# function fkern(phi::Weighted, prior::Weighted, sigma::Real)
#     fvec = fkern(phi.array, prior, sigma)
#     fopt = wname(phi.opt, "f_KL("*phi.opt.aname*")") |> unnormalise
#     Weighted(phi.array, fvec, fopt)
# end


#========== Faster ==========#

using LinearAlgebra
using Einsum
using ArrayAllez

function ekern(cols::Matrix{T1}, lambdas::Vector{T}, sigma::Real) where {T1<:AbstractFloat, T<:AbstractFloat}
    d, k = size(cols)
    @assert length(lambdas) == k
    mi2v = -0.5/sigma^2 |> T
    kappa = (d/2)*log(2π*sigma^2) + d/2 |> T

    dist2 = Array_{T}(:dist2, k,k) # pairwise! doesn't like ForwardDiff?
    innersum = sum( scale!(exp!(scale!(pairwise2!(dist2, cols), mi2v)), lambdas), dims=1) |> vec

    -dot(lambdas, finite!(log!(innersum))) + kappa
end

function ekern∇prob(cols::Matrix, lambdas::Vector{T}, sigma::Real, Δ=1) where {T}
    d, k = size(cols)
    mi2v = -0.5/sigma^2 |> T

    dist2 = Array_{T}(:dist2, k,k)
    exptop = exp!(scale!(pairwise2!(dist2, cols), mi2v))

    ilogin = inv!(exptop * lambdas)

    @einsum ∇lambdas[j] := lambdas[i] * exptop[i,j] * ilogin[i]
    @. ∇lambdas =  finite(Δ * (log(ilogin) - ∇lambdas))

    return ∇lambdas
end

function ekern∇both(cols::Matrix, lambdas::Vector{T}, sigma::Real, Δ=1) where {T}
    d, k = size(cols)
    mi2v = -0.5/sigma^2 |> T

    dist2 = Array_{T}(:dist2, k,k)
    exptop = exp!(scale!(pairwise2!(dist2, cols), mi2v))

    ilogin = inv!(exptop * lambdas)

    @einsum ∇lambdas[j] := lambdas[i] * exptop[i,j] * ilogin[i]
    @. ∇lambdas =  Δ * (log(ilogin) - ∇lambdas)

    ∇cols = Array_{T}(:∇cols, d,k)
    @einsum ∇cols[a,i] = -lambdas[i] * (ilogin[i] + ilogin[j]) * lambdas[j] * exptop[i,j] *
        (2*mi2v) * ( cols[a,i] - cols[a,j] ) # @reduce sum(j) lazy is much slower here

    return ∇cols, ∇lambdas
end

using Tracker: TrackedVector, TrackedMatrix, track, @grad, data

ekern(cols::Matrix, prob::TrackedVector, σ::Real) = track(ekern, cols, prob, σ)
ekern(cols::TrackedMatrix, prob::TrackedVector, σ::Real) = track(ekern, cols, prob, σ)

@grad ekern(cols::Matrix, prob::TrackedVector, σ::Real) =
    ekern(cols, data(prob), σ), Δ -> (nothing, ekern∇prob(cols, data(prob), σ, data(Δ)), nothing)

@grad function ekern(cols::TrackedMatrix, prob::TrackedVector, σ::Real)
    S = ekern(cols.data, prob.data, σ) # this takes about 1/8 the time of ∇both
    ∇cols, ∇prob = ekern∇both(cols.data, prob.data, σ)
    S, Δ -> (data(Δ) .* ∇cols, data(Δ) .* ∇prob, nothing)
end


