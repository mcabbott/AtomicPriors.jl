
using Test, Printf, Statistics, LinearAlgebra

t1 = time()
using AtomicPriors, WeightedArrays
t2 = time()
@info string("Loading AtomicPriors took ", @sprintf("%.1f", t2-t1), " seconds")

using DoubleFloats

@testset "AtomicPriors.jl" begin

mat = sobol(5,10).array
mat = mat ./ sum(mat, dims=1)
prob = normalise(rand(10))

_allequal(xs; atol=eps()) = all(isapprox.(first(xs), xs, atol=atol))

@testset "information" begin

    two = Weighted([1 0; 0 1])
    @test entropy(two) ≈ log(2)
    @test mutual(two) ≈ log(2)

    @test mjoint(two) ≈ log(2)
    @test mcond(two) ≈ 0
    @test mpred(two) ≈ log(2)
    @test mbottle(1, two) ≈ 0

    @test mcarlo(two, 0.1, threads=false) ≈ log(2)
    @test mcarlo(two, 0.1, threads=true) ≈ log(2)

    dbltwo = Weighted(Double64.([1 0; 0 1]))
    @test mcarlo(dbltwo, 0.1, threads=false) isa Double64
    @test mcarlo(dbltwo, 0.1, threads=true) isa Double64

    # Monte carlo MI, compare to grid
    xs = xgrid(1, -0.5:0.005:1.5)
    pr = sobol(1,5); pr.weights .*= 1:5
    mi = mutual(gauss(xs, pr, 0.1))
    @test mi ≈ mcarlo(pr, 0.1, 10^6) atol=0.01

    @test mi ≈ mkern(pr, 0.1 * sqrt(2)) atol=0.1 # not very accurate!

    # Check predict(j, L1, L2) method
    L1 = wrand(4,5); L1.array ./= sum(L1.array, dims=1); L1
    L2 = wrand(4,5); L2.array ./= sum(L2.array, dims=1); L2
    @test predict(L1, L2) ≈ reduce(hcat, [predict(j, L1, L2) for j in 1:4])

    # Risk f_kl
    @test fbayes(gauss(xs,wgrid(1,0:1),0.1)) ≈ [log(2), log(2)] atol=0.001
    @test fkern(wgrid(1,0:1), 0.1) ≈ [log(2), log(2)] atol=0.001
    @test fcarlo(wgrid(1,0:1), 0.1) ≈ [log(2), log(2)] atol=0.001

    @test fcond(gauss(xs,wgrid(1,0:1),0.1)) |> _allequal

    gl = gauss(xs,sobol(1,7),0.1);
    @test mutual(gl) ≈ mean(fbayes(gl))
    @test [fbayes(v, gl) for v in eachcol(gl.array)] ≈ fbayes(gl)
    ys = soboln(4,13)
    @test mkern(ys,0.1) ≈ mean(fkern(ys,0.1))
    @test [fkern(v, ys, 0.1) for v in eachcol(ys.array)] ≈ fkern(ys, 0.1)
    @test mcarlo(ys,0.1) ≈ mean(fcarlo(ys,0.1)) atol=0.02
    @test [fcarlo(v, ys, 0.1) for v in eachcol(ys.array)] ≈ fcarlo(ys, 0.1) atol=0.02

    # Repeated
    @test mutual(Weighted(mat,prob),2) ≈ mutual(repeat(Weighted(mat,prob),2))
    @test mutual(Weighted(mat,prob),5) ≈ mutual(repeat(Weighted(mat,prob),5))
    @test mutual(Weighted(mat,prob),12) ≈ mutual(repeat(Weighted(mat,prob),3),4)

end
@testset "fast versions" begin
    # Check that fast versions (which accept Matrix not AbstractMatrix)
    # agree with simple ones, by making a trivial wrapper:
    using OffsetArrays
    mat1 = OffsetArray(mat, (0,0))
    prob1 = OffsetArray(prob, (0,))
    col = mat[:,end]
    col1 = mat1[:,end]

    @test AtomicPriors.pairwise2(mat, mat) ≈ AtomicPriors.pairwise2(mat1, mat1)
    @test AtomicPriors.pairwise2(col, mat) ≈ AtomicPriors.pairwise2(col1, mat1)
    @test AtomicPriors.pairwise2(mat) ≈ AtomicPriors.pairwise2(mat1)

    @test mutual(mat, prob) ≈ mutual(mat1, prob1)

    @test ekern(mat, prob, 0.1) ≈ ekern(mat1, prob1, 0.1)

    @test AtomicPriors.yexp_compact(mat, 1:4) ≈ AtomicPriors.yexp_compact(mat1, 1:4)

end
@testset "gradients" begin
    # Check that custom gradient definitions match automatic:
    using Tracker, ForwardDiff

    dm, dp = @_ Tracker.gradient(mutual(_1, normalise(_2)), mat, prob) # mutual∇both
    @test dm ≈ @_ ForwardDiff.gradient(mutual(_, normalise(prob)), mat)
    @test dp ≈ @_ ForwardDiff.gradient(mutual(mat, normalise(_)), prob)
    @test dp ≈ (@_ Tracker.gradient(mutual(mat, normalise(_)), prob) |> __[1]) # mutual∇prob

    dm, dp = @_ Tracker.gradient(ekern(_1,_2,0.1), mat, prob) # ekern∇both
    @test dm ≈ @_ ForwardDiff.gradient(ekern(_, prob, 0.1), mat)
    @test dp ≈ @_ ForwardDiff.gradient(ekern(mat, _, 0.1), prob)
    @test dp ≈ (@_ Tracker.gradient(ekern(mat, _, 0.1), prob) |> __[1]) # ekern∇prob

    dm = Tracker.gradient(mat -> sum(sin, AtomicPriors.yexp_compact(mat, 1:4)), mat)[1]
    @test dm ≈ ForwardDiff.gradient(mat -> sum(sin, AtomicPriors.yexp_compact(mat, 1:4)), mat)

    r = rand(10)
    dr = (@_ Tracker.gradient(sum(sin, normalise(_)), r) |> __[1])
    @test dr ≈ @_ ForwardDiff.gradient(sum(sin, normalise(_)), r)

    # Repeated MI
    dm, dp = @_ Tracker.gradient(AtomicPriors.mutual_pos(_1, normalise(_2), 5), mat, prob) # ∇both
    @test dm ≈ @_ ForwardDiff.gradient(mutual(AtomicPriors.repeat_pos(_, 5), normalise(prob)), mat)
    @test dp ≈ @_ ForwardDiff.gradient(AtomicPriors.mutual_pos(mat, normalise(_), 5), prob)
    @test dp ≈ (@_ Tracker.gradient(AtomicPriors.mutual_pos(mat, normalise(_), 5), prob) |> __[1]) # ∇prob

end
@testset "optimisers" begin

    two = Weighted([0 1], 0,1)

    # Basic discrete outcome solving
    @test two ≈ optim!(pr -> mutual(coin(pr,1)), sobol(1,20), grad=:forward) |> sortcols
    @test_skip two ≈ optim!(pr -> mutual(coin(pr,1)), sobol(1,20), grad=:tracker) |> sortcols
    @test two ≈ adam!(pr -> mutual(coin(pr,1)), sobol(1,20), iter=1000) |> sortcols
    @test two ≈ nlopt!(pr -> mutual(coin(pr,1)), sobol(1,20)) |> sortcols

    # Weights only
    @test_broken two ≈ woptim!(pr -> mutual(coin(pr,1)), wgrid(1,0:0.1:1), grad=:forward) # doesn't converge?
    @test_broken two ≈ woptim!(pr -> mutual(coin(pr,1)), wgrid(1,0:0.1:1), grad=:tracker) # doesn't update!
    @test two ≈ wadam!(pr -> mutual(coin(pr,1)), wgrid(1,0:0.1:1), iter=1000) |> sortcols
    @test two ≈ wnlopt!(pr -> mutual(coin(pr,1)), wgrid(1,0:0.1:1)) |> sortcols

    # Using ekern() and variants
    optim!(pr -> ekern(pr, 0.5), sobol(1,10))
    @test two ≈ adam!(pr -> ekern(pr, 0.5), sobol(1,10); iter=10^3) |> sortcols
    @test two ≈ nlopt!(pr -> ekern(pr, 0.5), sobol(1,10)) |> sortcols
    @test two ≈ nlopt!(pr -> econd(pr, 0.9), sobol(1,10)) |> sortcols

    # Weights only
    @test two ≈ wadam!(pr -> ekern(pr, 0.5), wgrid(1,0:0.1:1); iter=10^3)
    @test two ≈ wnlopt!(pr -> ekern(pr, 0.5), wgrid(1,0:0.1:1)) |> sortcols
    @test two ≈ wnlopt!(pr -> econd(pr, 0.9), wgrid(1,0:0.1:1)) |> sortcols

end
@testset "posteriors" begin

    # Discrete
    two = Weighted([0 1], 0,1)
    @test (@_ post(1, coin(_,3), two)).weights ≈ [1,0]
    @test (@_ post([2,2,2], coin(_,2), two)) ≈ two  atol=0.01

    # Gaussian, trivial test!
    gp = gpost([0], wgrid(1, -2:0.01:2), 0.3).weights
    @test _allequal(exp.(.-(-2:0.01:2).^2 ./ (2*0.3^2)) ./ gp, atol=0.01)

end
@testset "models" begin

    @test coin(sobol(1,5),1).array ≈ vcat(1 .- sobol(1,5).array, sobol(1,5).array)

    @test coin(sobol(1,7),5).array ≈ repeat(coin(sobol(1,7),1),5).array

    # Check that bounded and unbounded parameters for yexp end up agreeing:
    p1 = adam!(pr -> ekern(yexp(pr,1:2),1), sobol(2,20), iter=1000)
    p2 = optim!(pr -> ekern(yexp(pr,1:2),1), 2 .* soboln(2,20))
    y1 = yexp(p1, 1:2) |> sortcols
    y2 = yexp(p2, 1:2) |> WeightedArrays.trim |> unique |> sortcols
    @test y1 ≈ y2 atol=0.00001

    @test _allequal(@_ fkern(yexp(_,1:2), p1, 1); atol=0.001)
    @test _allequal(@_ fkern(yexp(_,1:2), p2, 1); atol=0.001)

    # Check that one-vector and bulk yexp agree
    y4 = mapslices(θ->yexp(θ, 1:4, compact=true), sobol(3,10), dims=1)
    @test y4 ≈ yexp(sobol(3,10), 1:4)

    y5 = mapslices(θ->yexp(θ, 1:4, compact=false), soboln(3,10), dims=1)
    @test y5 ≈ yexp(soboln(3,10), 1:4)

    y6 = mapslices(θ->AtomicPriors.yexp!(zeros(4), θ, 1:4), soboln(3,10), dims=1)
    @test y6 ≈ yexp(soboln(3,10), 1:4)

    # Check that Vandermonde formula is working
    jl10 = map(jlog_exp(1:2), eachcol(soboln(2,10).array))[2:end] # 1st one -21 not -Inf on 1.4?
    @test jl10 ≈ map(jlog_vdm(1:2), eachcol(soboln(2,10).array))[2:end]
    @test jlog_exp(1:0.5:2)([0.1, -0.2, 0.3]) ≈ jlog_vdm(1:0.5:2)([0.1, -0.2, 0.3])
    @test_throws Exception jlog_vdm(1:3)(rand(2)) # only for d=m

    # Check that Jacobian is right, and in-place version matches:
    θ3 = [0.1, -2, 1/pi]
    J4 = ForwardDiff.jacobian(yexp(1:4), θ3)
    @test J4 ≈ AtomicPriors.jacobi_exp(θ3, 1:4)
    @test J4 ≈ AtomicPriors.jacobi_exp!(similar(J4), θ3, 1:4)
    @test jlog_exp(1:4)(θ3) ≈ logdet(J4' * J4)/2
end

@testset "sampling" begin

    # Gaussian samples with metropolis/emcee:
    flog_gauss(θ::AbsVec, σ::Real=1) = sum(abs2, θ) / (-2σ^2)
    @test mcarlo(metropolis(flog_gauss, wrandn(2,1), 10_000), 1) ≈ log(2) atol=0.1
    @test mcarlo(emcee(flog_gauss, wrandn(2,10), 10_000), 1) ≈ log(2) atol=0.1

    emcee(jlog_exp(1:2), wrandn(2,10), 10_000)
    # plot(yexp(ans, 1:2))

    # Mitchell sampling: this should be close to uniform
    f1 = mitchell(identity, sobol(1,5), 3002)
    f0 = wgrid(1, range(0,1,length=1000))
    x0 = xgrid(1, -0.2:0.01:1.2)
    @test entropy(gauss(x0, f1, 0.1)) ≈ entropy(gauss(x0, f0, 0.1)) atol=0.05

    # Mark's algorithm
    me1 = @_ transtrum(yexp(_, 1:2), soboln(2,5), 0.1, 10_000, best=false)
    me2 = @_ transtrum(yexp(_, 1:2), soboln(2,5), 0.1, 10_000, best=true)
    me3 = transtrum_exp(soboln(2,5), 1:2, 0.1, 10_000, best=false) # different implementation
    me4 = transtrum_exp(soboln(2,5), 1:2, 0.1, 10_000, best=true)
    # plot(unique(yexp(ans, 1:2), digits=3))
    memi = mcarlo(yexp(me1, 1:2), 0.1)
    @test memi ≈ mcarlo(yexp(me2, 1:2), 0.1) atol=0.1
    @test memi ≈ mcarlo(yexp(me3, 1:2), 0.1) atol=0.1
    @test memi ≈ mcarlo(yexp(me4, 1:2), 0.1) atol=0.1

    mt = @_ transtrum(tanh.(_), soboln(1,5), 0.1, 10_000) # not uniform, has weight on edges
    @test mcarlo(tanh(mt), 0.1) > mcarlo(wgrid(1, -1:0.01:1), 0.1)

    mc = @_ transtrum(coin(_,20), sobol(1,5)) # not sampling, deterministic
    @test length(mc.weights) == 21
    oc = @_ nlopt!(mutual(coin(_,20)), sobol(1,30))
    oi = mutual(coin(oc,20))
    @test  oi-0.1 < mutual(coin(mc,20)) < oi

    mc2 = @_ transtrum(coin(_,1), sobol(1,5), repeat=20) # different implementation
    @test sort(vec(mc2.array)) ≈ sort(vec(mc.array))

end

t3 = time()
@info string("Main tests took ", @sprintf("%.1f", t3-t2), " seconds")

@testset "notebook: $file" for file in filter(endswith(".jl"), readdir("../docs"))
    include("../docs/$file") 
end

@info string("Notebook tests took ", @sprintf("%.1f", time()-t3), " seconds")

end # @testset "AtomicPriors.jl"
