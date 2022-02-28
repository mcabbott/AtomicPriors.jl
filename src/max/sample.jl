
#========== MCMC ==========#

using KissMCMC: KissMCMC

using Printf

"""
    metropolis(logpdf, θ0, n) -> matrix
    metropolis(logpdf, Π0, n) -> Weighted

Takes `n` samples according to the given `logpdf(::Vector)`.
If given a `Π0::Weighted` instead of an initial vector,
it will read bounds from this, and return another such.

Keywords `step=1.0` size of jump, `burn=n/3` number of burn-in steps,
`nthin=1` keeps every step.

```
metropolis(jlog_exp(1:2), wrandn(2,1), 1_000)
plot(yexp(ans, 1:2))
```
"""
function metropolis(logf::Function, theta0::Vector, n::Int;
        step=1.0, burn=div(n,3), verbose=true, kw...)
    res, acc = KissMCMC.metropolis(logf, θ->θ .+ step .* randn.(), theta0;
        niter=n+burn, nburnin=burn, kw...)
    @info string("metropolis accepted ", @sprintf("%.2f",100*acc), " percent")
    reduce(hcat, res)
end

function metropolis(logf::Function, x::Weighted, num::Int; kw...)
    newf = x.opt.clamp ? clamped_pdf(logf, x.opt.lo, x.opt.hi) : logf
    mat = metropolis(newf, x.array[:,1], num; kw...)
    n = size(mat,2)
    w1 = convert(eltype(mat), inv(n))
    clamp!(Weighted(mat, fill(w1, n), x.opt))
end

clamped_pdf(logf, lo, hi) = θ -> all(z -> lo<=z<=hi, θ) ? logf(θ) : convert(eltype(θ),-Inf)


"""
    emcee(logpdf, Π0, n) -> Weighted

Takes about `n` samples according to the given `logpdf(::Vector)`.
The initial `Π0::Weighted` provides number of walkers, their starting points,
and bounds on `θ` if any.

Sets `res.trace` to be the accepted logpdf values.

Keywords `burn=n/3` number of burn-in steps, `nthin=1` keeps every step.
```
emcee(jlog_exp(1:2), wrandn(2,10), 1_000)
plot(yexp(ans, 1:2))
```
"""
function emcee(logf::Function, vecs::AbsVec, n::Int; burn=div(n,3), verbose=true, kw...)
    isodd(length(vecs)) && push!(vecs, first(vecs))
    res_, acc_, den_, blobs_ = KissMCMC.emcee(logf, vecs; niter=n+burn, nburnin=burn, kw...)
    res, acc, den = KissMCMC.squash_walkers(res_, acc_, den_)
    blobs = if !isnothing(blobs_)
        blobs_2, _ = KissMCMC.squash_walkers(blobs_, acc_)
        reduce(hcat, blobs_2)
    end
    @info string("emcee accepted ", @sprintf("%.2f",100*acc), " percent")
    reduce(hcat, res), den, blobs
end

function emcee(logf::Function, x::Weighted, num::Int; kw...)
    d,k = size(x)
    if k < d+2 # then double the number of walkers & repeat
        xplus = clamp!(hcat(x, x.array .+ 0.01 .* randn.()))
        return emcee(logf, xplus, num; kw...)
    end
    newf = x.opt.clamp ? clamped_pdf(logf, x.opt.lo, x.opt.hi) : logf
    mat, den, _ = emcee(newf, collect.(eachcol(x.array)), num; kw...)
    n = size(mat,2)
    w1 = convert(eltype(mat), inv(n))
    opt = WeightedArrays.wname(_wtrace(x.opt, den), "emcee-"*x.opt.wname)
    clamp!(Weighted(mat, fill(w1, n), opt))
end

_wtrace(o::WeightOpt, v::AbstractVector) =
    WeightedArrays.Setfield.set(o, WeightedArrays.Setfield.@lens(_.trace), v)


#========== My algorithm ==========#
# This aims to find useful candidate points, for further optimisation.

using Distances

using Statistics

"""
    mitchell(y, Π0, n) -> Weighted

This finds `n` samples, trying to make `y(Πnew)[:,a]` far from every `y(Π0)[:,b]`,
and returns the combined `hcat(Πnew, Π0)`. Variant of Mitchell's best-candidate algorithm.

Keyword `cand=30` is the number of candidate points at each step, of which the
best `batch=5` get saved. `step = 0.5` is the size of each jump (in `θ` space).
`dist = Distances.SqEuclidean()` is the measure of far away (in `y` space).
"""
function mitchell(yfun, Π::Weighted, num::Int; cand=30, batch=5, step=0.5, dist=SqEuclidean())
    d,k = size(Π)
    empty!(Π.opt.trace)

    # run yfun once, and build a big array to store new points
    y0 = yfun(Π).array
    m,k = size(y0)
    ymat = similar(y0, m, k+num)
    ymat[:, 1:k] .= y0

    # similar big matrix for all positions
    θmat = similar(Π.array, d, k+num)
    θmat[:, 1:k] .= Π.array

    @views for col in k+1:batch:k+num
        # randomly select points to perturb, and map new versions to y
        is = rand(1:(col-1), cand)
        θ1 = θmat[:, is] .+ step .* randn.()
        Π1 = clamp!(Weighted(θ1, trues(cand), Π.opt))
        y1 = yfun(Π1).array

        # pick the candidates furthest from everybody, and save them
        d2 = pairwise(dist, y1, ymat[:, 1:col-1], dims=2) # size cand × col-1
        m2 = vec(minimum(d2, dims=2)) # for each candidate, dist to closest point

        b = min(batch, k+num+1-col) # how many to save, usually == batch
        cs = sortperm(m2)[end-b+1:end]

        θmat[:, col:col+b-1] .= θ1[:, cs]
        ymat[:, col:col+b-1] .= y1[:, cs]
        append!(Π.opt.trace, m2[cs])
    end
    s1 = @sprintf("%.3f",mean(Π.opt.trace[1:batch]))
    s2 = @sprintf("%.3f",mean(Π.opt.trace[end-batch+1:end]))
    @info "mitchell dist = $s1 -> $s2"

    Weighted(θmat, normalise(vcat(Π.weights .* (k÷2), trues(num))), Π.opt)
end

#========== Mark's algorithm ==========#

"""
    transtrum(y, Π0, σ, n) -> Weighted

This is samples from `p^NML(x)` using `emcee`, and returns the max-likelihood `θ̂`s,
for a model with gaussian noise `x ~ y(θ) + 𝒩(0,σ²)`.
These points sample a generalised slab-and-spikes prior, favouring the bondaries.

Initial `Π0::Weighted` sets the number of dimensions & the number of walkers.
Keyword `best=true` starts at the best column of `Π0`, else a random point (default).

Keywords `iter=30, ftol=0.0` controls LBFGS for finding each `θ̂` if `method=Optim`.
But default is now `method=LsqFit`, seems a bit better maybe?
Others keywords passed to `emcee`.
```
@_ transtrum(yexp(_, 1:2), soboln(2,5), 0.05, 10_000)
plot(unique(yexp(ans, 1:2), digits=3))
```
"""
function transtrum(yfun, Π::Weighted, σ::Real, n::Int; kw...)
    _, _, θs = _transtrum(yfun, Π, σ, n; kw...)
    n = size(θs,2)
    w1 = convert(eltype(θs), inv(n))
    Weighted(θs, fill(w1, n), Π.opt)
end

function _transtrum(yfun, Π::Weighted, σ::Real, n::Int;
        method=LsqFit, best=false, iter=30, ftol=0.0, eps=1e-2, kw...)
    Π.opt.clamp && error("expected an unclamped prior!")
    # θ = Π[:,1]
    x0s = map(yfun, eachcol(Π.array))
    # J0 = similar(θ, length(first(x0s)), length(θ)) # not thread-safe!
    Jcaches = [similar(Π.array, length(first(x0s)), size(Π,1)) for _ in 1:Threads.nthreads()]

    xs, dens, θs = emcee(x0s, n; hasblob=true, kw...) do x
        θ0 = best ? Π.array[:, find_closest_index(x, x0s)] : randn!(Π[:,1])
        if method == LsqFit
            lsqfit_gauss_mle(x, yfun, θ0, σ, Jcaches[Threads.threadid()])
        elseif method == Optim
            optim_gauss_mle(x, yfun, θ0, σ; iter=iter, ftol=ftol) # gives (logpdf, θ)
        elseif method == ForwardDiff
            grad_gauss_mle(x, yfun, θ0, σ; iter=iter, eps=eps)
        end
    end
end

using LsqFit: LsqFit # Mark's reccommendation, Levenberg-Marquardt alg:

function lsqfit_gauss_mle(z::AbsVec, yfun::Function, θ0::AbsVec, σ::Real,
        J=similar(θ0, length(yfun(θ0)), length(θ0)))
    lsq_fun(_,θ) = yfun(θ)
    lsq_jac(_,θ) = ForwardDiff.jacobian!(J, yfun, θ)
    fit = LsqFit.curve_fit(lsq_fun, lsq_jac, 1:0, z, θ0)
    sum(abs2, fit.resid)/(-2 * σ^2), fit.param
end

using Optim # my go-to LBFGS idea, for finding closest θ:

function optim_gauss_mle(z::AbsVec, yfun::Function, θ0::AbsVec, σ::Real; iter=30, ftol=0.0)
    od = OnceDifferentiable(θ0, autodiff=:forward) do θ
        sum(abs2.(yfun(θ) .- z))
    end
    opt = LBFGS(linesearch = Optim.BackTracking()) # default is linesearch = Optim.HagerZhang()
    res = Optim.optimize(od, θ0, opt, Optim.Options(iterations=iter, f_abstol=ftol))
    Optim.minimum(res)/(-2 * σ^2), Optim.minimizer(res)
end

using ForwardDiff # my super-crude gradient descent, a sanity check:

function grad_gauss_mle(z::AbsVec, yfun::Function, θ0::AbsVec, σ::Real; iter=10^4, eps=1e-2)
    for _ in 1:iter
        grad = ForwardDiff.gradient(θ0) do θ
            sum(abs2.(yfun(θ) .- z))
        end
        θ0 .-= eps .* grad
    end
    sum(abs2.(yfun(θ0) .- z))/(-2 * σ^2), θ0
end

function find_closest_index(x::AbsVec, x0s::AbsVec{<:AbsVec})
    @assert length(x) == length(first(x0s))
    ind, dist = 0, Inf
    @inbounds for j in 1:length(x0s)
        dj = 0.0
        x0j = x0s[j]
        for μ in 1:length(x)
            dj += (x[μ] - x0j[μ])^2
        end
        ind = ifelse(dj<dist, j, ind)
        dist = min(dj, dist)
    end
    ind
end

#=

best=true seems to lead this (from notes/sumexp03.jl) to run into singularities:
@time m3 = transtrum(yexp(t26), 4 .* soboln(3,50), 0.1, 200_000; nthin=20)

ERROR: TaskFailedException:
SingularException(3)
Stacktrace:
 [1] checknonsingular at /Applications/Julia-1.5.app/Contents/Resources/julia/share/julia/stdlib/v1.5/LinearAlgebra/src/factorization.jl:19 [inlined]
...
 [6] \(::Array{Float64,2}, ::Array{Float64,1}) at /Applications/Julia-1.5.app/Contents/Resources/julia/share/julia/stdlib/v1.5/LinearAlgebra/src/generic.jl:1116
 [7] levenberg_marquardt(::NLSolversBase.OnceDifferentiable{Array{Float64,1},Array{Float64,2},Array{Float64,1}}, ::Array{Float64,1}; x_tol::Float64, g_tol::Float64, maxIter::Int64, lambda::Float64, tau::Float64, lambda_increase::Float64, lambda_decrease::Float64, min_step_quality::Float64, good_step_quality::Float64, show_trace::Bool, lower::Array{Float64,1}, upper::Array{Float64,1}, avv!::Nothing) at /Users/me/.julia/packages/LsqFit/LeVh7/src/levenberg_marquardt.jl:126

 =#

"""
    transtrum(f, Π0) -> Weighted
    transtrum(f, Π0, n; repeat)

Version for discrete outcomes: `f(Π0)` should be the likelihood matrix.
This just scans all `x ∈ X` and stores the closest `θ` point to each,
weighted by `p^NML(x)`.

Nothing monte-carlo about it, and no need to specify a number of samples!
But you can sub-sample to keep just `n`, especially for large `repeat`.

Keyword `iter=100` controls LBFGS for finding each `θ̂`, this is
not as robust as I would like!
```
@_ transtrum(coin(_,10), sobol(1,5))        # 11 points for 11 outcomes
@_ transtrum(coin(_,1), sobol(1,5), repeat=10)
@_ nlopt!(mutual(coin(_,10)), sobol(1,20))  # 5 points, maximal I(X;Θ)
```
Keyword `repeat=5` uses a more efficient method than `repeat(f(_),5)`.
"""
function transtrum(f, Π::Weighted, n::Union{Nothing,Int}=nothing; iter=100, repeat=nothing)
    Π.opt.clamp && @warn "not sure this works well with clamped priors"

    isnothing(repeat) || return transtrum_repeat(f, Π, n, repeat; iter=iter)
    n isa Int && return transtrum_repeat(f, Π, n, 1; iter=iter)

    d,k = size(Π)
    like = f(Π).array
    m,_ = size(like)

    # outputs
    phi = similar(Π.array, d, m)
    pNML = similar(Π.weights, m)

    Threads.@threads for i in 1:m
        # start at the closest point, why not?
        ai = argmax(like[i,:])
        θ0 = Π.array[:,ai]

        # solve
        od = OnceDifferentiable(θ0, autodiff=:forward) do θ
            pie = WeightedMatrix(θ, [1.0], Π.opt) |> clamp!
            -log(f(pie).array[i,1]) # maximum likelihood, minimum -log(f())
            # -f(pie).array[i,1]
        end
        opt = LBFGS(linesearch = Optim.BackTracking())
        res = Optim.optimize(od, θ0, opt, Optim.Options(iterations=iter))

        # save
        phi[:,i] .= Optim.minimizer(res)
        pNML[i] = exp(-Optim.minimum(res)) # optimising in terms of log-prob a little better
        # pNML[i] = -Optim.minimum(res)
    end

    # sanitise
    ok = map(isfinite, pNML)
    Weighted(phi[:, ok], normalise!(pNML[ok]), Π.opt) |> clamp! |> unique
end

function transtrum_repeat(f, Π::Weighted, n, repeat::Int; iter=100)
    d,k = size(Π)
    like1 = f(Π)::Weighted
    m1,_ = size(like1) # m1 is number of outcomes of 1 repetition

    zz = part_pos(repeat, m1) # don't ask why it's called zz everywhere
    mul = multi_pos(repeat, m1)
    like2 = Base.repeat(like1, repeat).array
    if n isa Int
        n < length(zz) || @warn "you asked for $n samples, out of $(length(zz)) total ($m1 outcomes with $repeat repetitions)"
        list = rand(1:length(zz), n)
        zz = zz[list]
        mul = mul[list]
        like2 = like2[list, :]
    end
    m2 = length(zz) # m2 is the total, after all repetitions

    # outputs
    phi = similar(Π.array, d, m2)
    pNML = similar(Π.weights, m2)

    _transtrum_repeat(f,Π,repeat,iter,d,k,like2,zz,mul,m2,phi,pNML)
end # because zz is not type-stable
function _transtrum_repeat(f,Π,repeat,iter,d,k,like2,zz,mul,m2,phi,pNML)

    Threads.@threads for i in 1:m2
    # for i in 1:m2
        # start at the closest point, why not?
        ai = argmax(like2[i,:])
        θ0 = Π.array[:,ai]

        # solve
        od = OnceDifferentiable(θ0, autodiff=:forward) do θ
            pie = WeightedMatrix(θ, [1.0], Π.opt) |> clamp!

            # now, instead of calculating the whole repeated likelihood, only i-th row:
            loglikepie = f(pie).array
            -log(mul[i]) - sum(log, loglikepie[x,1] for x in zz[i])
        end
        opt = LBFGS(linesearch = Optim.BackTracking())
        res = Optim.optimize(od, θ0, opt, Optim.Options(iterations=iter))

        # save
        phi[:,i] .= Optim.minimizer(res)
        pNML[i] = exp(-Optim.minimum(res))
    end

    # sanitise
    ok = map(isfinite, pNML)
    Weighted(phi[:, ok], normalise!(pNML[ok]), Π.opt) |> clamp! |> unique
end

#= # This is copied from here:

@noinline function _repeat_pos(mat,m,zz,out,k,mul)
    for (r,z) in enumerate(zz) # row of the final likelihood
        out[r,:] .= mul[r]
        for x in z             # is a product of like of individual outcomes
            out[r,:] .*= mat[x,:]
        end
    end
    out
end

=#

"""
    transtrum_exp(Π0, times, σ, n)

This is like `transtrum(y, Π0, σ, n)` but uses the analytic `jacobi_exp`
instead of ForwardDiff, for LsqFit. Example:
```
ty1 = transtrum_exp(soboln(2,10), 1:2, 0.1, 10^4; nthin=10)
ty2 = @_ transtrum(yexp(_, 1:2), soboln(2,10), 0.1, 10^4; nthin=10)
plot(yexp(ty1, 1:2))
plot!(yexp(ty2, 1:2) .+ [0, 0.25])
```
"""
transtrum_exp(Π::Weighted, times::AbsVec, σ::Real, n::Int; kw...) = 
    first(transtrum_exp_pair(Π, times, σ, n; kw...))

function transtrum_exp_pair(Π::Weighted, times::AbsVec, σ::Real, n::Int; best::Bool=false, kw...)
    Π.opt.clamp && error("expected an unclamped prior!")
    x0s = collect.(eachcol(yexp(Π, times).array))

    Jcaches = [similar(Π.array, length(first(x0s)), size(Π,1)) for _ in 1:Threads.nthreads()]
    θcaches = [similar(Π.array, size(Π,1)) for _ in 1:Threads.nthreads()]
    ycaches = [similar(Π.array, length(times)) for _ in 1:Threads.nthreads()]

    xs, dens, θs = emcee(x0s, n; hasblob=true, kw...) do x # NB emcee is multi-threaded!
        θ0 = best ? Π.array[:, find_closest_index(x, x0s)] : randn!(Π[:,1])
        tid = Threads.threadid()
        lsqfit_exp_mle(x, θ0, times, σ, Jcaches[tid], θcaches[tid], ycaches[tid])
    end

    n = size(θs,2)
    w1 = convert(eltype(θs), inv(n))
    Weighted(θs, fill(w1, n), Π.opt), Weighted(xs) # return xs too, for p_NML scatter plots
end

function lsqfit_exp_mle(z::AbsVec, θ0::AbsVec, times::AbsVec, σ::Real,
        Jcache::AbsMat, θcache::AbsVec, ycache::AbsVec)
    lsq_fun(_,θ) = yexp!(ycache, θ, times)
    lsq_jac(_,θ) = jacobi_exp!(Jcache, θ, times; cache=θcache) # these two share a lot of work!
    fit = LsqFit.curve_fit(lsq_fun, lsq_jac, 1:0, z, θ0)
    sum(abs2, fit.resid)/(-2 * σ^2), fit.param
end
