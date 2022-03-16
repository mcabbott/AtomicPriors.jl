
using Random

"""
    mcarlo(Π, σ, [n])

Calculates mutual information `I(X;Θ)` for gaussian noise, by monte carlo sampling
roughly `n` times. Really `cld(n,k)` per atom, i.e. `cld(n,k)*k` in total.
(Default is `2k` points in total, and at least 1000.)

As usual positions of prior's delta functions are the `k` columns of `Π`
and data is `x ~ 𝒩(θ,σ²)` in `d` dimensions, `d,k=size(Π)`.

Keyword `threads=true` runs multi-threaded, default is `Threads.nthreads()>1`.

Compare to `mutual(X,Π,σ)` with explicit grid, and `mkern(Π, σ)` KDE very-approx.
"""
mcarlo(Π::Weighted, σ::Real, n::Int=max(2*size(Π,2),1000); kw...) = mcarlo(array(Π), weights(Π), σ, n; kw...)

function mcarlo(theta::Matrix{T1}, weights::Vector{T2}, sigma::Real, iter::Int; threads::Bool=Threads.nthreads()>1) where {T1,T2}
    T = promote_type(T1,T2)
    eltype(weights) == T || return mcarlo(theta, map(T,weights), sigma, iter)
    sigma isa T || return mcarlo(theta, weights, T(sigma), iter)

    d,k = size(theta)
    @assert length(weights) == k
    if k==0
        @warn "mcarlo run with zero points, returning 0"
        return 0.0
    elseif iter<k
        @warn "ignoring n = $iter as this is smaller than k = $k"
    elseif sigma>π && sigma isa Int
        @warn "mcarlo called with integer σ = $sigma, is that what you wanted? (And n = $iter iterations)"
    end

    threads && return mcarlo_threads(theta, weights, sigma, iter)

    sumlog = zero(T)      # accumulate almost the answer
    randx = zeros(T,d)    # pre-allocate for random x
    mi2v = 1/(-2*sigma^2) # quiker to work this out once

    iterper = cld(iter, k) # unlike div this gives 1 if iter<k
    for a=1:k             # iterate over cols with nonzero weight
        weights[a] < MINWEIGHT && continue
        @inbounds for loop = 1:iterper
            # first draw data point x
            for r=1:d
                randx[r] = theta[r,a] + sigma * randn(T)
            end
            # contribution λ_a log( p(x|θ_a) / p(x) )
            sumlog += weights[a] * log_pxθa_px(randx, a, theta, weights, d,k, mi2v)
        end
    end
    sumlog / iterper
end

@inline function log_pxθa_px(randx, a, theta, weights, d,k, mi2v)
    T = eltype(weights)
    px = zero(T)
    pxθa = zero(T)
    # build up p(x), and get p(x|θ_a) along the way.
    @inbounds for j=1:k
        dist2 = zero(T)
        for r=1:d
            dist2 += (randx[r] - theta[r,j])^2
        end
        Pj = exp(dist2 * mi2v) # prefactor of gaussian would cancel in pxθa/px
        if j == a
            pxθa = Pj
        end
        px += Pj * weights[j]
    end
    log(pxθa / px)
end

function mcarlo_threads(theta::Matrix, weights::Vector{T}, sigma::Real, iter::Int) where {T}
    d,k=size(theta)
    # @assert length(weights)==k # removing this cures type-instability WTF?
    mi2v = (1/(-2*sigma^2))

    caches = [zeros(T,d) for tid=1:Threads.nthreads()]   # working space per thread
    sumlog = zeros(T, Threads.nthreads())   # accumulate per thread, no locks, total at end
    # generators = [MersenneTwister() for tid=1:Threads.nthreads()]

    iterper = cld(iter, k)
    Threads.@threads for a=1:k
        weights[a] < MINWEIGHT && continue

        tid = Threads.threadid()
        # rng = generators[tid]
        randx = caches[tid]

        @inbounds for loop = 1:iterper
            for r=1:d
                randx[r] = theta[r,a] + sigma * randn(T) # (rng)
            end
            sumlog[tid] += weights[a] * log_pxθa_px(randx, a, theta, weights, d,k, mi2v)
        end
    end
    sum(sumlog) / iterper
end

# On Julia 1.3 the default rng is thread-safe.
# If it's a bottleneck you can look at faster ones:
# https://github.com/JuliaLang/julia/issues/27614
# https://sunoru.github.io/RandomNumbers.jl/dev/man/benchmark/#Speed-Test-1

# RI2 had gradient definitions for mcarlo(), but not so useful really.

"""
    fcarlo(φ, Π, σ, [n]) -> real
    fcarlo(Φ, Π, σ, [n]) -> vector

Calculates bayes risk `f_KL(φ)` for gaussian noise, by monte carlo sampling
roughly `n` times in total (default is `2k` points in total, and at least 1000).

To apply to many points `φ1,φ2,...` at once, give a many-colum `Φ::Weighted`, and get a vector.

Similar to `fbayes(Φ,X,Π,σ)` with explicit grid, and `fkern(Φ,Π,σ)` KDE approx.

    fcarlo(f, Φ, Π, σ) = fcarlo(f(Φ), f(Π), σ)
    fcarlo(Π, σ) = fcarlo(Π, Π, σ)

Given a function, it applies this to both `Φ` and `Π`.
If not given `Φ`, then it uses `Π` for this too.
"""
fcarlo(phi::AbsVec, prior::Weighted, sigma::Real, n::Int=max(1000, 2*size(prior,2))) =
    fcarlo(reshape(phi,:,1), array(prior), weights(prior), sigma, n) |> first

fcarlo(manyphi::Weighted, prior::Weighted, sigma::Real, n::Int=max(1000, 2*size(prior,2))) =
    fcarlo(array(manyphi), array(prior), weights(prior), sigma, n)

# with a function
fcarlo(f, phi::AbsVec, prior::Weighted, σ::Real, n::Int...) = fcarlo(f(WeightedMatrix(phi)), f(prior), σ, n...) |> first
fcarlo(f, phis::Weighted, prior::Weighted, σ::Real, n::Int...) = fcarlo(f(phis), f(prior), σ, n...)

# without phi
fcarlo(prior::Weighted, σ::Real, n::Int...) = fcarlo(prior, prior, σ, n...)
fcarlo(f, prior::Weighted, σ::Real, n::Int...) = fcarlo(f, prior, prior, σ, n...)

function fcarlo(phi::AbstractMatrix{T1}, theta::AbstractMatrix{T2}, weights::Vector{T3}, sigma::Real, iter::Int) where {T1,T2,T3}
    T = promote_type(T1, T2, T3, typeof(sigma))
    eltype(phi) == T || return fcarlo(map(T,phi), theta, weights, sigma, iter)
    eltype(theta) == T || return fcarlo(phi, map(T,theta), weights, sigma, iter)
    eltype(weights) == T || return fcarlo(phi, theta, map(T,weights), sigma, iter)
    sigma isa T || return fcarlo(phi, theta, weights, T(sigma), iter)

    d, k = size(theta)
    dϕ, kϕ = size(phi)
    iter<k && @warn "ignoring n = $iter as this is smaller than k = $k"
    @assert length(weights) == k
    dϕ==d || error("dimension of point ϕ must match that of prior θ")

    out = zeros(T,kϕ)

    randx = zeros(T,d)
    mi2v = 1/(-2*sigma^2)

    iterper = cld(iter, kϕ)
    for b=1:kϕ
        @inbounds for loop = 1:iterper
            # draw data point x
            for r=1:d
                randx[r] = phi[r,b] + sigma * randn(T)
            end
            # save log( p(x|ϕ_b) / p(x) ) -- different inner function this time
            out[b] += log_pxϕb_px(randx, b,phi, theta, weights, d,k, mi2v) / iterper
        end
    end
    out
end

@inline function log_pxϕb_px(randx, b,phi, theta, weights, d,k, mi2v)
    T = eltype(weights)
    #  build up p(x)
    px = zero(T)
    @inbounds for j=1:k
        dist2 = zero(T)
        for r=1:d
            dist2 += (randx[r] - theta[r,j])^2
        end
        Pj = exp(dist2 * mi2v) # prefactor of gaussian would cancel
        px += Pj * weights[j]
    end
    # need p(x|ϕb) separately as ϕb not among the list
    dist2′ = zero(T)
    for r=1:d
        dist2′ += (randx[r] - phi[r,b])^2
    end
    pxϕb = exp(dist2′ * mi2v)
    # log(pxϕb / px) # |> finite # seems dodgy
    safelogpoq(pxϕb, px)
end

function safelogpoq(p,q)
    x = log(p/q)
    ifelse(iszero(p), zero(x), x)
end
