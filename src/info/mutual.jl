
using WeightedArrays

using TensorCast

"""
    entropy(L)  # S(X)   = ∑_x - p(x) log(p(x))
    mutual(L)   # I(X;Θ) = ∑_x ∑_θ p(x|θ)p(θ) log(p(x|θ)/p(x))

These act on `Weighted(likelihood, prior)` for a model with discrete outcomes,
as produceed for instance by `coin(Π,m)` or `gauss(X,Π,σ)`.

    mjoint(LX, LY)      # I(XY;Θ)
    mcond(LX, LY)       # I(Y;Θ|X) = I(XY;Θ) - I(X;Θ)
    mbottle(LX, LY, β)  # β * I(Y;Θ) - I(X;Θ)

These expect two such, with `LY=LX` be default.
Coming from the same prior, they are assumed to have the same `L.weights`.
"""
function entropy(L::Weighted)
    like, prob = L.array, normalise(L.weights)
    px = like * prob
    @reduce sum(x) -px[x] * log(px[x])
end

@doc @doc(entropy)
mutual(L::Weighted) = mutual(L.array, normalise(L.weights))
function mutual(like::AbsMat, prob::AbsVec) # this method has gradients & faster versions
    px = like * prob
    @reduce sum(x,t) finite(like[x,t] * prob[t] * log(like[x,t] / px[x]))
end

@doc @doc(entropy)
mjoint(LX::Weighted, LY::Weighted=LX) = mutual(vkron(LX, LY))

@doc @doc(entropy)
mcond(LX::Weighted, LY::Weighted=LX) = mjoint(LX, LY) - mutual(LX)

@doc @doc(entropy)
mbottle(β::Real, LX::Weighted, LY::Weighted=LX) = β * mutual(LY) - mutual(LX)

function vkron(LA::Weighted, LB::Weighted)
    @cast LC[μ⊗ν, a] := LA.array[μ, a] * LB.array[ν, a]
    Weighted(LC, LA.weights, LA.opt)
end

"""
    fbayes(Lφ, LΘ) -> real
    fbayes(LΦ, LΘ) -> vector

Bayes risk function `f_KL(φ)` for a model with likelihood matrix `LΘ`.
For the evaluation point `φ` it needs one column of the likelihood matrix, `Lφ`.

To apply to many points `φ1,φ2,...` at once, give a many-colum `LΦ`, and get a vector.

    fbayes(f, φ, Π) = fbayes(f(φ), f(Π))
    fbayes(LΘ) = fbayes(LΘ.array, LΘ)

Or give a function! This calls `f(WeightedMatrix(φ))` to make `LΦ`.
Or omit `LΦ` & it'll use `LΘ` for both.
"""
function fbayes(likephi::AbsMat, LX::Weighted)
    @assert size(likephi,1) == size(LX,1) "Lφ and LΘ must have same number of rows"
    px = LX.array * normalise(LX.weights)
    @reduce fkl[φ] := sum(x) likephi[x,φ] * log(likephi[x,φ] / px[x])
end

# get phi into shape
fbayes(likephi::AbsVec, LX::Weighted) = only(fbayes(reshape(likephi, :,1), LX))
fbayes(likephi::Weighted, LX::Weighted) = fbayes(likephi.array, LX)

# with a function
fbayes(f, phi::Weighted, prior::Weighted) = fbayes(f(phi).array, f(prior))
fbayes(f, phi::AbsVec, prior::Weighted) = fbayes(f, WeightedMatrix(phi), prior)

# without phi
fbayes(LX::Weighted) = fbayes(LX.array, LX)
fbayes(f, prior::Weighted) = fbayes(f(prior))


# """
#     fcond(fX, fY, phi, prior)

# Predictive risk `f_KL(φ) = < D_KL[p(y|φ)||p(y|x)] >_x~p(x|φ)`.
# Perhaps useful for testing minimax things?
# """
# function fcond(fX, fY, phi::Weighted, prior::Weighted)
#     LXΦ = fX(phi).array
#     LX = fX(prior).array
#     LYΦ = fY(phi).array
#     LY = fY(prior).array
#     error("not yet")
# end

# fcond(fX, fY, phi::Array, prior::Weighted) = fcond(fX, fY, WeightedMatrix(phi), prior)

"""
    mpred(LX, LY)

Computes `I(X;Y)` using Bayes predictive density p(y|x) from `predict(LX, LY)`.

Keyword `pipx=false` means `p(x)=1/N` instead of default `like * prior`.
Keyword `pipy=true` means that `p(y) = like * prior`, instead of propagating `p(x)`.
These are not well thought-through!
"""
function mpred(LX::Weighted, LY::Weighted=LX; pipx=true, pipy=false)
    prob = normalise(LX.weights)
    if pipx
        px = LX.array * prob # default
    else
        nx = size(LX,1)
        px = fill(1/nx, nx)
    end
    pyx = predict(LX, LY) # p(y|x)
    if pipy
        py = LY.array * prob
    else
        py = pyx * px  # default
    end
    # sum(finite, pyx .* px' .* log.( pyx ./ py ) )
    @reduce sum(y,x) finite(pyx[y,x] * px[x] * log(pyx[y,x] / py[y]))
end


# """
#     fcond(Lφ, LX, LY) -> real
#     fcond(LΦ, LX, LY) -> vector

# Predictive bayes risk function `f_C(φ)` for a model with likelihood matrix `LX`.
# For the evaluation point `φ` it needs one column of the likelihood matrix, `Lφ`.

# To apply to many points `φ1,φ2,...` at once, give a many-colum `LΦ`, and get a vector.

# """
# function fcond(likephi::AbsMat, LX::Weighted, LY::Weighted=LX)
#     pyx = predict(LX, LY)
#     py = LY.array * normalise(LX.weights)
#     @reduce fkl[φ] := sum(y,x) fuck * likephi[y,φ] * log(likephi[y,φ] / pyx[y,x])
#     error("not yet")
# end


"""
    fcond(fX, fY, φ, Π) -> real
    fcond(fX, fY, Φ, Π) -> vector

Predictive Bayes risk function `f_C(φ) = ∑_x p(x|φ) D_KL[ p(y|φ) || p(y|x) ]`.

The model has likelihood matrix `LX[j,a] = p(xⱼ|θₐ)` where `LX = fX(Π)`,
and predictions `LY[j,a] = p(yⱼ|θₐ)` where `LX = fY(Π)`.
The posterior is encoded `Π.weights[a] = p(θₐ|z)`.

The evaluation point `φ` is likewise mapped to likelihood `fX(WeightedMatrix(φ))`.
To apply to many points `φ1,φ2,...` at once, give a many-colum `Φ::Weighted`,
and get a vector.
"""
fcond(fX, fY, phi::Weighted, prior::Weighted) =
    fcond(fX(phi).array, fY(phi).array, fX(prior), fY(prior))

function fcond(LphiX::AbsMat, LphiY::AbsMat, LX::Weighted, LY::Weighted)
    pyx = predict(LX, LY)
    @reduce fC[φ] := sum(x,y) LphiX[x,φ] * (LphiY[y,φ] * log(LphiY[y,φ] / pyx[y,x]))
end

# get lone phi into shape, return a number
fcond(fX, fY, phi::AbsVec, prior::Weighted) =
    only(fcond(fX, fY, WeightedMatrix(phi), prior))
fcond(f, phi::AbsVec, prior::Weighted) =
    only(fcond(f, WeightedMatrix(phi), prior))

# case X=Y, you can omit one function
fcond(f, phi::Weighted, prior::Weighted) = fcond(f(phi).array, f(prior))
fcond(Lphi::AbsMat, LX::Weighted) = fcond(Lphi, Lphi, LX, LX)

# case Φ=Π, don't run f twice
fcond(fX, fY, prior::Weighted) = fcond(fX(prior), fY(prior))
fcond(f, prior::Weighted) = fcond(f(prior))
fcond(LX::Weighted, LY::Weighted=LX) = fcond(LX.array, LY.array, LX, LY)


finite(z) = ifelse(isfinite(z), z, zero(z))
finite!(zs) = zs .= finite.(zs)

#========== Faster ==========#

using ArrayAllez # vectorised log! etc, with IntelVML / AppleAccelerate

using LinearAlgebra

function mutual∇prob(like::Matrix, prob::Vector)
    px = like * prob
    inner = log!(iscale_(like, px))

    ∇prob = vec(sum(scale!(inner, like), dims=1)) |> finite!
end # ∇prob is f_KL, real gradient ∇prob - MI, done later

function mutual∇both(like::Matrix, prob::Vector)
    px = like * prob
    inner = log!(iscale_(like, px))  # next-best: @vielsum inner[x,θ] := log(like[x,θ] * ipx[x])

    ∇prob = vec(sum(like .* inner , dims=1)) |> finite!
    ∇like = scale!(inner, prob') |> finite!
    return ∇like, ∇prob # ∇prob is f_KL, real gradient ∇prob - MI, done later
end

using Tracker: Tracker, TrackedVector, TrackedMatrix, track, @grad, data

mutual(like::Matrix, prob::TrackedVector) = track(mutual, like, prob)
mutual(like::TrackedMatrix, prob::TrackedVector) = track(mutual, like, prob)

@grad function mutual(theta::Matrix, prob::TrackedVector)
    ∇prob = mutual∇prob(theta, prob.data) # ∇prob is f_KL
    MI = dot(∇prob, prob.data)
    ∇prob .= ∇prob .- MI
    MI, Δ -> (nothing, scale!(∇prob,data(Δ)))
end

@grad function mutual(like::TrackedMatrix, prob::TrackedVector)
    ∇like, ∇prob = mutual∇both(like.data, prob.data)
    MI = dot(∇prob, prob.data) # ∇prob is f_KL
    ∇prob .= ∇prob .- MI
    MI, Δ -> (scale!(∇like,data(Δ)), scale!(∇prob,data(Δ)))
end

WeightedArrays.normalise(x::TrackedVector) = track(normalise, x)

@grad function WeightedArrays.normalise(x)
    itot = inv(sum(data(x)))
    back(dy) = (data(dy) .* itot .- dot(data(dy), data(x)) * itot^2,)
    scale_(data(x), itot), back
end
