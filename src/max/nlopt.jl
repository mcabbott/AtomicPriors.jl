
using NLopt

"""
    nlopt!(F, Π)

This uses NLopt's LBFGS to maximise `F(Π)`.
Mutates `Π`, returns `unique(trim(Π))`.

Keyword `iter=0` allows as many steps as it desires,
`xtol=0, ftol=0` use the package's default stopping,
else these are absolute not relative tolerances.

    wnlopt!(F, Π)
    wnlopt!(F, G, Π) = wnlopt!(F, G(Π), Π)

`wnlopt!` adjusts only the weights. The second form maximises `F(G(Π))`
evaluating `G` only once, `F` at every step.
"""
function nlopt!(fmax::Function, x::Weighted; kw...)
    sv = flatten(x)
    minfun(v) = -fmax(flatcopy(x, v))
    tr = nlopt_vec_tracker(sv, minfun, (x.opt.clamp, x.opt.lo, x.opt.hi, length(x.array)); kw...)
    append!(x.opt.trace, tr)
    flatcopy!(x, sv) |> normalise! |> clamp! |> trim |> unique
end


function wnlopt!(fmax::Function, x::Weighted; kw...)
    sv = x.weights
    minfun(v) = -fmax(wcopy(x, v))
    tr = nlopt_vec_tracker(sv, minfun; kw...)
    append!(x.opt.trace, tr)
    wcopy!(x, sv) |> normalise! |> clamp! |> trim |> unique
end

wnlopt!(fmax::Function, g::Function, x::Weighted; kw...) =
    wnlopt!(fmax, g(x), x; kw...)

function wnlopt!(fmax::Function, gofx::Weighted, x::Weighted; kw...)
    size(x,2)==size(gofx,2) || error("Π and G(Π) must have same number of columns")
    sv = x.weights
    minfun(v) = -fmax(wcopy(gofx, v))
    tr = nlopt_vec_tracker(sv, minfun; kw...)
    append!(x.opt.trace, tr)
    wcopy!(x, sv) |> normalise! |> clamp! |> trim |> unique
end

using Tracker: Tracker, data

function nlopt_vec_tracker(sv::Vector, minfun::Function,
        (limit, lo, hi, num) = (false,0,1,0); opt=:LD_LBFGS, iter=0, xtol=0, ftol=0)
    trace = Float64[]

    # Funcion in NLopt's format
    function nlfunc(vec, grad)
        val, back = Tracker.forward(minfun, vec)
        if length(grad) != 0
            grad .= data(back(true)[1])
        end
        sv .= vec # to return something if it fails
        push!(trace, -data(val))
        data(val)
    end

    # Now build up an objective for it
    nlname = NLopt.Opt(opt, length(sv))
    min_objective!(nlname, nlfunc)

    iter > 0 && maxeval!(nlname, iter)
    xtol>0 && xtol_abs!(nlname, xtol)
    ftol>0 && ftol_abs!(nlname, xtol)

    top = limit ? fill(hi, length(sv)) : fill(Inf, length(sv))
    bot = limit ? fill(lo, length(sv)) : fill(-Inf, length(sv))
    top[num+1:end] .= Inf # simple bounds in weight directions, as clamping gave errors
    bot[num+1:end] .= 0
    upper_bounds!(nlname, top)
    lower_bounds!(nlname, bot)

    try
        minf,minx,ret = NLopt.optimize(nlname, sv)
        if ret == :FORCED_STOP
            push!(trace, -nlfunc(sv, zero(sv))) # will give better error message
            # if no error, then probably ^C interrupted it:
            @warn "NLopt FORCED_STOP after $(length(trace)) steps"
        end
        sv .= minx
        push!(trace, -minf)
    catch err
        @error "NLopt failed after $(length(trace)) steps"
        rethrow(err)
    end
    trace
end

