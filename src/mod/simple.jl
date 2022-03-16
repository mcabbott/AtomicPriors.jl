
const like_opt = WeightOpt(norm=false, clamp=true, lo=0, hi=Inf, aname="p(x|θ)", like=true)

"""
    coin(Π::Weighted, m, ϵ=1e-10) -> Weighted

Works out likelihood matrix for Bernoilli model, `m` flips.
Third parameter is a regulator, by which `θ` is kept away from 0 & 1.
"""
function coin(prior::Weighted, m::Int, ϵ=1e-10)
    size(prior,1) == 1 || error("only one coin at a time, please")
    xs = 0:m
    binx = float_binomial.(m, xs)
    theta = ϵ .+ (1-2ϵ) .* prior.array # this is a row!
    like = @. binx * theta^xs * (1-theta)^(m-xs)
    Weighted(like, prior.weights, like_opt)
end

using SpecialFunctions # factorial(n::Float64) = gamma(n+1) from v2

function float_binomial(n,k) # usually only defined for integers
    if k>n
        return zero(float(n+k))
    else
        num = gamma(float(n+1))
        den = gamma(float(n-k+1)) * gamma(float(k+1))
        num/den
    end
end


"""
    gauss(X, Π::Weighted, σ) -> Weighted

Taking columns of `X` to be `x` etc, this gives the likelihood matrix
`p(x|θ) ∝ e^-(x-θ)²/2σ²` normalised `∑ₓ p(x|θ) = 1`,
in the form of a `Weighted{Matrix}` whose weights are `p(θ)` from `Π`.
"""
function gauss(data::AbsMat, prior::Weighted, sigma::Number)
    size(data,1)==size(prior,1) || error("gauss needs X and Π of same height")
    raw = exp.( pairwise2(data, prior.array) .* (-0.5/sigma^2) )
    like = raw ./ sum(raw, dims=1)
    Weighted(like, prior.weights, like_opt)
end

using LinearAlgebra

"""
    pairwise2(x, y=x) = Distances.pairwise(SqEuclidean(), x, y)

Resulting `mat[i,j]` is distance sqared from `x[:,i]` to `y[:,j]`.
(If one of them is a vector, then it it returns all distances to `y[:,j]`.)
"""
pairwise2(x::AbsMat, y::AbsMat) = sum(x.*x; dims=1)' .+ sum(y .* y; dims=1) .- 2 .* x'*y
# pairwise2(x::AbsMat, y::AbsMat) = diag(x'*x) .+ diag(y'*y)' .- 2 .* x'*y

function pairwise2(x::AbsMat) # = diag(x'*x) .+ diag(x'*x)' .- 2 .* x'*x
    mat = x'*x
    vec = diag(mat)
    vec .+ vec' .- 2 .* mat
end

pairwise2(x::AbsVec, y::AbsMat) = vec(pairwise2(reshape(x,:,1), y))
pairwise2(x::AbsMat, y::AbsVec) = vec(pairwise2(x, reshape(y,:,1)))

using Distances

pairwise2(x::Matrix, y::Matrix) = pairwise(SqEuclidean(), x, y; dims=2)
pairwise2(x::Matrix) = pairwise(SqEuclidean(), x; dims=2)

pairwise2!(out::Matrix, x::Matrix, y::Matrix) = pairwise!(out, SqEuclidean(), x, y; dims=2)
pairwise2!(out::Matrix, x::Matrix) = pairwise!(out, SqEuclidean(), x; dims=2)
