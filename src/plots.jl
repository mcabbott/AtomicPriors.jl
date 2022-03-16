
# Am not sure whether these can easily be made into recipies.
# import AtomicPriors: pplot, pplot!, yplot, yplot!

PPLOT_LIMITS = (x=[0.0, 1.0], y=[0.0, 1.0])

"""
    pplot(x::Weighted)
    pplot!(x)

The first calls `sPCA`, the second calls `rPCA` to plot on the same axes.
The axis ranges are now fixed to to what the first call decides.

    pplot(f, x)
    pplot(λ, x)

Ditto, but first applies function `f`, or scales by `λ`.
"""
function pplot(x::Weighted, f::Function=identity; digits=WeightedArrays.DIGITS, kw...)
    pca_fx = sPCA(unique(f(x), digits=digits))
    PPLOT_LIMITS.x .= extrema(pca_fx.array[1,:]) # would be nice to expand a little!
    PPLOT_LIMITS.y .= WeightedArrays.pcaylim(pca_fx)

    Plots.plot(pca_fx; kw...)
    Plots.plot!(xlim=PPLOT_LIMITS.x, ylim=PPLOT_LIMITS.y)
end
pplot(x::Weighted, λ::Number; kw...) = pplot(x, z->λ .* z)

function pplot!(x::Weighted, f::Function=identity; shift=0, digits=WeightedArrays.DIGITS, kw...)
    Plots.plot!(shift .+ rPCA(unique(f(x), digits=digits)); kw...)
    Plots.plot!(xlim=PPLOT_LIMITS.x, ylim=PPLOT_LIMITS.y)
end
pplot!(x::Weighted, λ::Number; kw...) = Plots.plot!(x, z->λ .* z; kw...)

pplot(f::Function, x::Weighted; kw...) = pplot(x, f; kw...) # standard order
pplot!(f::Function, x::Weighted; kw...) = pplot!(x, f; kw...)


"""
    yplot(y, x::Weighted, ts)

Time-series plotting. If `y(Π, 1:3)` is what we observe (up to noise)
then `yplot(y, Π, 1:3)` will plot a line for each column of `Π`,
for times `range(0, tmax, length=50)` points; `pts=50` and `tmax=3*1.1`
are keywords.

Line opacity represents weight (relative to heaviest) which can be scaled
by e.g. `alpha=1/20`. Keyword `tri=true` puts markers at each time point.
"""
function yplot(args...; tri=true, kw...)
    Plots.plot(xaxis = "time")
    yplot!(args...; tri=tri, kw...)
end

function yplot!(yfun::Function, prior::Weighted, times::AbstractVector;
        pts=50, c=:blue, m=:uptriangle, alpha=1, tmax=1.1*maximum(times), lab="", tri=false, kw...)
    alpha = alpha .* sqrt.(prior.weights ./ maximum(prior.weights)) |> transpose

    # run y once on all times:
    ptimes = range(zero(tmax), tmax, length=pts)
    res = yfun(prior, ptimes).array

    if tri == true
        Plots.scatter!(times, zero(times); c=:grey, m=m, lab="", opacity=0.5)
    end

    # case of y_react etc, when y(θ) should be a matrix
    if :syms in fieldnames(typeof(yfun))
        dout, k = size(res)
        tensor = reshape(res, length(yfun.syms), :,k)
        if length(yfun.syms) > 1
            for (i,s) in enumerate(yfun.syms)
                Plots.plot!(0:0, 0:0; lab=join([s,lab], " "), c=i, alpha=1, kw...)
                Plots.plot!(ptimes, tensor[i, :, :]; lab="", c=i, alpha=alpha, kw...)
            end
        else # the same, just without fixing the colour!
            Plots.plot!(0:0, 0:0, lab=lab, c=c, alpha=1, kw...)
            Plots.plot!(ptimes, tensor[1, :, :]; lab="", c=c, alpha=alpha, kw...)
        end

    # simple yexp etc.
    else
        Plots.plot!(0:0, 0:0; lab=lab, c=c, alpha=1, kw...)
        Plots.plot!(ptimes, res; lab="", c=c, alpha=alpha, kw...)
    end
    Plots.plot!()
end


