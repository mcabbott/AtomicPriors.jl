
"""
    repeat(L::Weighted, m)

Works out the likelihood matrix corresponding to `m` repetitions of the given one.
"""
Base.repeat(x::Weighted, m::Int) = Weighted(repeat_pos(x.array, m), x.weights, x.opt)

# Base.repeat(m::Int) = x -> repeat(x,m) # Base.repeat(3) is an error by default

function repeat_pos(mat::AbsMat, m::Int)  # correct but not quick
    m==1 && return mat
    d,k = size(mat)

    zz = part_pos(m, d)
    mul = multi_pos(m, d)

    mul .* reduce(vcat, [prod(mat[z,:], dims=1) for z in zz])
end

function repeat_pos(mat::Matrix{T}, m::Int) where T
    m==1 && return mat
    d,k = size(mat)

    zz = part_pos(m, d)   # each z∈zz is an overall outcome, made up of [x1, x2, x3] individual outcomes
    mul = multi_pos(m, d) # each entry here is a multinomial factor corresp to a z
    out = similar(mat, length(zz), k)

    _repeat_pos(mat,m,zz,out,k,mul)
end # because zz is not type-stable
@noinline function _repeat_pos(mat,m,zz,out,k,mul)
    for (r,z) in enumerate(zz) # row of the final likelihood
        out[r,:] .= mul[r]
        for x in z             # is a product of like of individual outcomes
            out[r,:] .*= mat[x,:]
        end
    end
    out
end


#===== multinomials etc. =====#

using StaticArrays
using Base.Cartesian

"""
    part_pos(m, n)

For m repetitions of a process with n outcomes, this returns a list of combined outcomes,
not counting order. Each is a list of m numbers, sorted `1 ≦ j₁≦j₂≦…≦j_m ≦ n`,
"positions" of what happened.

Earlier there was a `part_occ` saving occupation numbers, but this worked out much slower.

1st call is to @nloops version, 2nd call is memoized by hand now:
```
@btime AtomicPriors.part_pos(5, 5)            # 13.947 ns -- from dict
@code_warntype AtomicPriors.part_pos(5, 5)    # ::Any
@btime AtomicPriors.part_pos(Val(5), Val(5))  # 1.550 μs -- calculated, Vector{SVector}
```
"""
function part_pos(m,n)
    get!(part_pos_dict, (m,n)) do
        part_pos(Val(m), Val(n))
    end
end

const part_pos_dict = Dict{NTuple{2,Int},Vector}()

@generated function part_pos(::Val{m}, ::Val{len}) where {m,len}
    quote
        out = SVector{$m,Int}[]
        # syntax is @nloops(how_many_loops, index_name_base, range_making_function, loop_body)
        @nloops $m  i  k->(k==$m ? (1:$len) : (1:i_{k+1}))  begin
            push!(out, SVector(@ntuple($m,i)))
        end
        out
    end
end

function multinomial_pos(m::Int, z::SVector{M}) where M
    @assert m==M
    multinomial_pos(z)
end

function multinomial_pos(m::Int, v::AbsVec)
    z = sort(SVector{m}(v))
    multinomial_pos(z)
end

function multinomial_pos(z::SVector{m}) where m
    cnt = zero(MVector{m,Int})
    c = 1
    cnt[c] = 1
    @inbounds for i=2:m
        # z[i] != z[i-1] && (c += 1) # exploit that z is sorted
        c += ifelse(z[i] == z[i-1], 0,1)
        cnt[c] += 1
    end

    if m<=20
        factorial(m) / prod(factorial, cnt)
    else
        gamma(m + 1.0) / prod(n -> gamma(n + 1.0), cnt)
    end
end

"""
    multi_pos(m, n)

Gives the full list of multinomial coefficients for the outcomes `part_pos(m, n)`.
"""
function multi_pos(m,n)::Vector{Float64}
    get!(multi_pos_dict, (m,n)) do
        zz = part_pos(m, n)
        multinomial_pos.(zz)
    end
end

const multi_pos_dict =  Dict{NTuple{2,Int},Vector{Float64}}()


#===== repeated mutual information =====#

"""
    mutual(L, m) = mutual(repeat(L,m))
    L |> mutual(m)

Mutual information for repeated measurements. This avoids explicitly working out
the full likelihood matrix `repeat(L,m)`.
"""
mutual(x::Weighted, m::Int) = mutual_pos(x.array, normalise(x.weights), m)
mutual(m::Int) = x::Weighted -> mutual(x,m)

using Base.Threads
using LinearAlgebra

function mutual_pos(like::AbsMat, prob::AbsVec, m::Int)
    m==1 && return mutual(like, prob)
    d,k = size(like)
    zz = part_pos(m, d)

    _mutual_pos(d,k,like,prob,m,zz)
end # barrier because part_pos not type-stable
@noinline function _mutual_pos(d,k,like,prob,m,zz)

    # Unthreaded
    # sum(finite, multinomial_pos(z) * posn_inner(like,z,prob,k) for z in zz)

    # Threaded without locks
    multi = multi_pos(m, d)
    out = zeros(eltype(prob), nthreads())
    pre = [ones(k) for t=1:nthreads()]

    @threads for i=1:length(zz)
        out[threadid()] += multi[i] * posn_inner!(pre[threadid()],like,zz[i],prob,k) |> finite
    end
    sum(out)
end

# function posn_inner(like::AbsMat,z,prob,k)
#     pzx = reduce( (v,w)->v.*w , like[r,:] for r in z)
#     ipx = 1/dot(pzx, prob)
#     sum( pzx .* prob .* log.(pzx .* ipx) )
# end

function posn_inner!(pzx, like::Matrix, z::SVector{m}, prob, k) where m
    pzx .= 1
    @inbounds for c=1:k
        for r in z
            pzx[c] *= like[r,c]
        end
    end
    ipx = 1/dot(pzx, prob)

    # sum( pzx .* prob .* log.(pzx .* ipx) )
    out = zero(eltype(like))
    @inbounds @simd for c=1:k
        out += pzx[c] * prob[c] * log(pzx[c] * ipx)
    end
    out
end


#===== gradient for mutual information =====#

function mutual_pos∇prob(like::Matrix, prob::Vector, m::Int)
    m==1 && return mutual∇prob(like, prob)
    d,k = size(like)
    zz = part_pos(m, d)
    sum(multinomial_pos(z) * pos_prob_inner(like,z,prob,k) for z in zz)
end # gradient should really be ∇prob - MI, but I correct later

function pos_prob_inner(like,z,prob,k)
    # pzx = reduce( (v,w)->v.*w , like[r,:] for r in z)
    pzx = like[z[1],:]
    for r in z[2:end], i=1:k
        pzx[i] *= like[r,i]
    end
    ipx = 1/dot(pzx, prob)
    pzx .* log.(pzx .* ipx) |> vec
end


function mutual_pos∇both(like::Matrix, prob::Vector, m::Int, Δ=1)
    m==1 && return mutual∇both(like, prob)
    d,k = size(like)
    @assert k==length(prob)
    zz = part_pos(m, d)
    barr_pos∇both3(d,k,like,prob,zz,Δ) # version 3 has better threading, and 30x less allocation
end # barrier because zz type-unstable

# @noinline function barr_pos∇both1(d,k,like,prob,zz::Vector{<:SVector{m}},Δ) where m
#     ∇prob = zeros(k)                 # k
#     ∇like = zeros(d,k)               # d,k

#     for z in zz
#         mm = multinomial_pos(z) # scalar

#         pzx = reduce( (v,w)->v.*w , like[r,:] for r in z)
#         # pzx = like[z[1],:]
#         # for r in z[2:end], i=1:k
#         #     pzx[i] *= like[r,i]
#         # end                          # k

#         ipx = 1/dot(pzx, prob)    # scalar
#         lograt = log.(pzx .* ipx)    # k

#         ∇prob .+= mm .* pzx .* lograt

#         @inbounds for i=1:m
#             ∇like[z[i],:] .+= mm .* ∇like_row_del(like,z,i) .* lograt
#         end
#     end
#     ∇like .*= prob' .* Δ
#     scale!(∇prob, Δ)

#     return ∇like, ∇prob # gradient should really be ∇prob - MI, but I correct later
# end

# using Base.Threads

# Version 3 of barrier function -- threaded without locks, and careful about allocation
@noinline function barr_pos∇both3(d,k,like,prob,zz::Vector{<:SVector{m}},Δ) where m

    ∇prob = [zeros(k) for t=1:nthreads()]                 # k
    ∇like = [zeros(d,k) for t=1:nthreads()]               # d,k

    row = [ones(k) for t=1:nthreads()] # for ∇like_row_del!
    pre = [ones(k) for t=1:nthreads()] # for pzx
    rat = [ones(k) for t=1:nthreads()] # for log(ratio)

    multi = multi_pos(m, d)

    @threads for iz=1:length(zz)
        þ = threadid()
        emm = multi[iz]
        zed = zz[iz]

        pzx = pre[þ]
        @inbounds for i=1:k
            pzx[i] = 1
            for r in zed
                pzx[i] *= like[r,i]
            end
        end

        ipx = 1/dot(pzx, prob)    # scalar

        @inbounds for j=1:k
            rat[þ][j] = log(pzx[j] * ipx)

            ∇prob[þ][j] += Δ * emm * pzx[j] * rat[þ][j]
        end

        @inbounds for i=1:m
            ∇like_row_del!(row[þ],k,like,zed,i)  # mutates row[þ]
            for j=1:k
                ∇like[þ][zed[i],j] += ( emm * row[þ][j] * rat[þ][j] * prob[j] * Δ )
            end
        end
    end

    return sum_first!(∇like), sum_first!(∇prob) # .* prob' .* Δ  etc put in above instead
end

∇like_row_del(like,z,i) = # multiply all rows like[z[j],:] except the z[i]-th
    reduce( (v,w)->v.*w , like[z[j],:] for j=1:length(z) if j!=i )

function ∇like_row_del!(out,k,like,z,i)
    out .= 1
    @inbounds for j = 1:length(z)
        if i!=j
            out .*= @view like[z[j],:]
        end
    end
    out
end

function sum_first!(vec::Vector{<:Array})
    @inbounds for i=2:length(vec)
        vec[1] .+= vec[i]
    end
    vec[1]
end

using Tracker: TrackedVector, TrackedMatrix, track, @grad, data

mutual_pos(like::Matrix, prob::TrackedVector, m::Int) = track(mutual_pos, like, prob, m)
mutual_pos(like::TrackedMatrix, prob::TrackedVector, m::Int) = track(mutual_pos, like, prob, m)
mutual_pos(like::TrackedMatrix, prob::Vector, m::Int) = track(mutual_pos, like, param(prob), m) # added for tests?

@grad function mutual_pos(like::Matrix, prob::TrackedVector, m::Int)
    ∇prob = mutual_pos∇prob(like, prob.data, m)
    MI = dot(∇prob, prob.data)
    ∇prob .-= MI
    MI, Δ -> (nothing, lmul!(data(Δ),∇prob), nothing)
end

@grad function mutual_pos(like::TrackedMatrix, prob::TrackedVector, m::Int)
    ∇like, ∇prob = mutual_pos∇both(like.data, prob.data, m) # if you do this on forward pass you don't have Δ
    MI = dot(∇prob, prob.data)
    ∇prob .-= MI
    MI, Δ -> (lmul!(data(Δ),∇like), lmul!(data(Δ),∇prob), nothing)
end

