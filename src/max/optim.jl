
using Optim

"""
    optim!(F, Π::Weighted)
    woptim!(F, Π)

Maximise function `F` by adjusting `Π`, using `Optim.jl` with LBFGS.
Mutates `Π` but returns `unique(trim(Π))`.
Version `woptim!` adjusts only the weights.

Keyword `grad=:forward` or `grad=:tracker` selects AD.
Keyword `iter=n` is maximum number of iterations, but seems to be ignored.
"""
function optim!(f, x::Weighted; grad=:forward, verbose=false, kw...)
    res = if grad == :forward
        optim_forward!(f, x; kw...)
    elseif grad == :tracker
        optim_tracker!(f, x; kw...)
    else
        error("don't understand grad = :$grad")
    end
    verbose && printstyled(" Optim -- ",res,"\n", color=:blue)
    append!(x.opt.trace, map(z -> -z.value, res.trace))
    x |> trim |> unique
end

function woptim!(f, x::Weighted; grad=:forward, verbose=false, kw...)
    res = if grad == :forward
        w_optim_forward!(f, x; kw...)
    elseif grad == :tracker
        w_optim_tracker!(f, x; kw...)
    else
        error("don't understand grad = :$grad")
    end
    verbose && printstyled(" Optim -- ",res,"\n", color=:blue)
    append!(x.opt.trace, map(z -> -z.value, res.trace))
    x |> trim |> unique
end


using ForwardDiff

function optim_forward!(fmax::Function, x::Weighted; iter::Int=typemax(Int))
    sv = flatten(x)
    od = OnceDifferentiable(sv, autodiff=:forward) do v
        y = clamp!(flatcopy(x, v))
        -fmax(y)
    end
    res = Optim.optimize(od, sv, LBFGS(), Optim.Options(iterations=iter, store_trace=true))
    flatcopy!(x, Optim.minimizer(res)) |> clamp! |> normalise!
    res
end
function w_optim_forward!(fmax::Function, x::Weighted; iter::Int=typemax(Int))
    sv = x.weights
    od = OnceDifferentiable(sv, autodiff=:forward) do v
        y = Weighted(x.array, clamp.(v,0,Inf), x.opt)
        -fmax(y)
    end
    res = Optim.optimize(od, sv, LBFGS(), Optim.Options(iterations=iter, store_trace=true))
    copy!(x.weights, Optim.minimizer(res))
    x |> clamp! |> normalise!
    res
end


using Tracker: Tracker, data

# Weird feature here is that the separate gradients grow to disagree from the whole-vector one.
# The latter is what nlopt! uses, and solves things better;
# but the former is what adam! uses.

function optim_tracker!(fmax::Function, x::Weighted; iter::Int=typemax(Int))
    sv = flatten(x)
    function fg!(F,G,v)
        clamp!(flatcopy!(x, v))
        value, back = Tracker.forward((a,w) -> -fmax(Weighted(a, w, x.opt)), x.array, x.weights)
        value2, back2 = Tracker.forward(vv -> -fmax(clamp(flatcopy(x, vv))), v)
        if G != nothing
            da, dw = back(one(eltype(v)))
            @show da[5:7]
            copyto!(G, vcat(vec(da.data), dw.data)) #, CatView(da, dw))
            dv = back2(one(eltype(v)))[1].data
            @show dv[5:7]
            G .= dv
        end
        if F != nothing
            return Tracker.data(value)
        end
    end
    res = Optim.optimize(Optim.only_fg!(fg!), sv, LBFGS(), Optim.Options(iterations=iter, store_trace=true))
    flatcopy!(x, Optim.minimizer(res)) |> clamp! |> normalise!
    res
end

function w_optim_tracker!(fmax::Function, x::Weighted; iter::Int=typemax(Int))
    sv = x.weights
    function fg!(F,G,v)
        value, back = Tracker.forward(w -> -fmax(Weighted(x.array, w, x.opt)), clamp.(v,0,Inf))
        if G != nothing
            dw = back(one(eltype(v)))[1]
            G .= Tracker.data(dw)
        end
        if F != nothing
            return Tracker.data(value)
        end
    end
    res = Optim.optimize(Optim.only_fg!(fg!), sv, LBFGS(), Optim.Options(iterations=iter, store_trace=true))
    copy!(x.weights, Optim.minimizer(res))
    x |> clamp! |> normalise!
    res
end
