
using Tracker: Tracker, data

"""
    adam!(F, Π; iter=10^3)

This uses the ADAM optimiser, taken from Flux, which copes better with noise
than does LBFGS. It has no stopping criterion, you just specify `iter`.
```
julia> adam!(x -> sum(x.array), sobol(1,3); iter=100)
Weighted 1×3 Matrix{Float64}, clamped 0.0 ≦ θ ≦ 1.0:
 0.6  0.85  0.35
with normalised weights p(θ), 3-element Vector{Float64}:
 0.333333  0.333333  0.333333
```
"""
function adam!(fmax::Function, x::Weighted; opt=ADAM(), kw...)
    opt_tracker!(fmax, x, opt; kw...)
    x |> trim |> unique
end

"""
    wadam!(F, Π; iter=100)
    wadam!(F, G, Π) = wadam!(F, G(Π), Π)

Adjusts only the weights. The second form maximises `F(G(Π))` evaluating `G`
only once, `F` at every step.
"""
function wadam!(fmax::Function, x::Weighted; opt=ADAM(), kw...)
    vec_opt_tracker!(v -> fmax(wcopy(x, v)), x, opt; kw...)
    x |> trim |> unique
end

wadam!(fmax::Function, g::Function, x::Weighted; kw...) =
    wadam!(fmax, g(x), x; kw...)

function wadam!(fmax::Function, gofx::Weighted, x::Weighted; opt=ADAM(), kw...)
    size(x,2)==size(gofx,2) || error("Π and G(Π) must have same number of columns")
    vec_opt_tracker!(v -> fmax(wcopy(gofx, v)), x, opt; kw...)
    x |> trim |> unique
end

function opt_tracker!(fmax, x, opt; iter::Int=1000)
    try
        for t in 1:iter
            value, back = Tracker.forward((a,w) -> -fmax(Weighted(a, w, x.opt)), x.array, x.weights)
            isfinite(value) || error("encountered value = $value at step $t")
            da, dw = back(one(eltype(x.weights)))
            update!(opt, x.array, data(da))
            update!(opt, x.weights, data(dw))
            x |> clamp! |> normalise!
            push!(x.opt.trace, -data(value))
        end
    catch err
        if err isa InterruptException
            @warn "Optimisation interrupted after $(length(x.opt.trace)) steps"
        else
            rethrow(err)
        end
    end
end

function vec_opt_tracker!(fmax, x, opt; iter::Int=1000)
    try
        for t in 1:iter
            value, back = Tracker.forward(w -> -fmax(w), x.weights)
            isfinite(value) || error("encountered value = $value at step $t")
            dw = back(one(eltype(x.weights)))[1]
            update!(opt, x.weights, data(dw))
            x |> clamp! |> normalise!
            push!(x.opt.trace, -data(value))
        end
    catch err
        if err isa InterruptException
            @warn "Optimisation interrupted after $(length(x.opt.trace)) steps"
        else
            rethrow(err)
        end
    end
end

#===== from Flux, verbatim =====#
# Note that this runs independently for each parameter,
# rather than looking at one big vector of them all.

const ϵFLUX = 1e-8

"""
    ADAM(η = 0.001, β::Tuple = (0.9, 0.999))

[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.
# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
# Examples
```julia
opt = ADAM()
opt = ADAM(0.001, (0.9, 0.8))
```
"""
mutable struct ADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, IdDict())

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵFLUX) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end


"""
    update!(opt, p, g)

Perform an update step of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).
As a result, the parameters are mutated and the optimizer's internal state may change.
"""
function update!(opt, x, Δ)
  x .-= apply!(opt, x, Δ)
end

# function update!(opt, xs::Params, gs)
#   for x in xs
#     gs[x] == nothing && continue
#     update!(opt, x, gs[x])
#   end
# end
