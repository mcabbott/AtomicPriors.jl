
using TensorCast

"""
    post(j, L, Π)
    post([j1,j2,...], L, Π)

Posterior `p(θ|xⱼ)`, given likelihood matrix `L[i,θ] = p(xᵢ|θ) ::Matrix`,
and prior `Π::Weighted`. For models with a discrete set of outcomes.

Instead of a matrix, `L::Weighted` is fine (weights ignored),
as is `L::Function` for which it finds `L(Π)` first.

With `m` observations `[j1, j2, ...]` it'll take the product of such.

    post(findcol(x,X), gauss(X,Π,σ), Π)

This used to have a method `post(x,X,Π,σ)`, but that was confusing.
"""
function post(j::Int, LX::Matrix, Π::Weighted)
    @cast un[θ] := LX[$j,θ] * Π.weights[θ]
    Weighted(Π.array, normalise(un), wname(Π.opt, "p(θ|x)"))
end

post(j::Int, LX::Weighted, Π::Weighted) = post(j, LX.array, Π)
post(j::Union{Int, AbsVec{Int}}, L::Function, Π::Weighted) = post(j, L(Π), Π)

function post(js::AbsVec{Int}, LX::Weighted, Π::Weighted)
    @assert size(LX,2) == size(Π,2) "likelihood & prior must have equal number of columns"
    un = copy(LX.weights)
    for j in js
        @cast un[θ] = un[θ] * LX.array[$j,θ]
    end
    Weighted(Π.array, normalise!(un), wname(Π.opt, "p(θ|xs)"))
end

"""
    gpost(x, Π, σ)
    gpost(x, f, Π, σ) = gpost(x, f(Π), σ)

Posterior `p(θ|x)` for models with Gaussian noise,
i.e. assuming that `p(x|θ) ∼ 𝒩(f(θ),σ²)`, without grid `X`.
"""
function gpost(x::AbsVec, f::Function, Π::Weighted, σ::Real)
    ys = f(Π).array
    @assert length(x) == size(ys,1) "length of vector must match height of matrix"
    dist2 = vec(pairwise2(reshape(x,:,1), ys))
    un = Π.weights .* exp.(dist2 .* (-0.5/σ^2))
    Weighted(Π.array, normalise!(un), wname(Π.opt, "p(θ|x)"))
end

gpost(x::AbsVec, Π::Weighted, σ::Real) = gpost(x, identity, Π, σ)

gpost(x::Real, f::Function, Π::Weighted, σ::Real) = gpost([x], identity, Π, σ)
gpost(x::Real, Π::Weighted, σ::Real) = gpost([x], identity, Π, σ)

"""
    findcol(x, X)

Finds `j` such that `x ≈ X[:,j]`, or close.
"""
function findcol(x::Vector, X::Matrix)
    @assert length(x)==size(X,1) "length of vector must match height of matrix"
    dist = vec(pairwise2(reshape(x,:,1), X))
    findfirst(d -> d == minimum(dist), dist)
end
findcol(x::Real, X::Matrix) = findcol([x], X)

"""
    predict(LX, LY)

Bayes predictive density `p(yᵢ|xⱼ)` for all `yᵢ ∈ Y` and all `xⱼ ∈ X`,
returned as a `(Matrix)ᵢⱼ`: each column is one `xⱼ`.
`(LX::Weighted)ᵢₐ` is `p(yᵢ|θₐ)`, with weights `p(θₐ)` i.e. it contains the prior.

    predict(j, LX, LY=LX) == predict(LX,LY)[:,j]
    predict([j1,j2,...], LX, LY=LX)

Predictive density `p(y|xⱼ)` for particular `xⱼ`, or several.
"""
function predict(LX::Weighted, LY::Weighted=LX)
    @assert size(LX,2) == size(LY,2) "likelihoods must have same number of columns"
    px = LX.array * LX.weights  # normalisation of prior will cancel from result
    ipx = 1 ./ px
    @cast post[θ,x] := LX.array[x,θ] * LX.weights[θ] * ipx[x] # Bayes rule
    LY.array * post # ∑_θ p(y|θ) p(θ|x)
end

function predict(j::Int, LX::Weighted, LY::Weighted=LX)
    @assert size(LX,2) == size(LY,2) "likelihoods must have same number of columns"
    @cast postj[θ] := LX.array[$j,θ] * LX.weights[θ]
    LY.array * normalise(postj)
end

function predict(js::AbsVec{Int}, LX::Weighted, LY::Weighted=LX)
    @assert size(LX,2) == size(LY,2) "likelihoods must have same number of columns"
    out = ones(size(LY,1))
    for j in js
        @cast postj[θ] := LX.array[$j,θ] * LX.weights[θ]
        out .*= (LY.array * postj) .* inv(sum(postj))
    end
    out
end

#========== Model Selection ==========#

"""
    evidence(j, L)

This computes `p(xⱼ)` for the model with likelihood matrix `L.array[i,θ] = p(xᵢ|θ)`,
`L.weights[θ] = p(θ)`, at one point `xⱼ = X[:,j]`.
"""
evidence(j::Int, L::Weighted) = dot(normalise(L.weights), L.array[j,:])
evidence(j::Int, f, Π::Weighted) = evidence(j, f(Π))

"""
    gevidence(x, y(Π), σ)

This computes `p(x)` for models with Gaussian noise, i.e. assuming that
`p(x|θ) ∼ 𝒩(y(θ),σ²)`. Should be equivalent to `evidence(j, gauss(X,y(Π),σ))`
with a grid `X`, apart from normalisation (i.e. factors `dx^d`).

Also `gevidence(big, x, y(Π), σ)` works at higher precision.
"""
function gevidence(T, x::AbsVec, Π::Weighted, σ::Number)
    d,k = size(Π)
    wei = T.(normalise(Π.weights))
    iden = T(1/(sqrt(2*T(π))*σ)^d)
    dot(wei, exp.(T.((-0.5/σ^2) .* pairwise2(x, Π.array)))) * iden
end

gevidence(x::AbsVec, Π::Weighted, σ::Number) = gevidence(Float64, x, Π, σ)
