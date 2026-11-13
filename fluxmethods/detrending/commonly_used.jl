# Block-average detrending, in Julia.
#
# A translation of `block_average` in `commonly_used.py`, the file beside this
# one: the mean of the averaging period is subtracted from the series, a missing
# value being left out of the mean and kept in the series.
#
# The mean is the whole of it, and languages add a series up differently: what
# Python computes is `numpy.nanmean`, pairwise in blocks of 128, and Julia's
# `sum` is pairwise with other blocks. So `pairwise_sum` below is numpy's
# summation in numpy's order, as it is in `axis_rotation/wilczak_et_al_2001.jl`.
# It is written out in each file rather than shared, so that a file is all of
# the code a call of it runs, and its digest says so.
#
# Base Julia only. `linear_detrend` is not translated: its fit is LAPACK's,
# which no file here can follow statement by statement.

# numpy's `pairwise_sum` (numpy/_core/src/umath/loops_utils.h.src): fewer than
# eight values are added in turn; up to 128, eight running sums are kept, added
# as ((0+1)+(2+3))+((4+5)+(6+7)), and the remainder added in turn; a longer run
# is cut in two at a multiple of eight and each half summed the same way.
function pairwise_sum(a::AbstractVector{Float64}, start::Int = 1, n::Int = length(a))
    if n < 8
        res = 0.0
        for i in 1:n
            res += a[start + i - 1]
        end
        return res
    end
    if n <= 128
        r = a[start:start + 7]
        i = 8
        while i < n - (n % 8)
            for j in 1:8
                r[j] += a[start + i + j - 1]
            end
            i += 8
        end
        res = ((r[1] + r[2]) + (r[3] + r[4])) + ((r[5] + r[6]) + (r[7] + r[8]))
        while i < n
            res += a[start + i]
            i += 1
        end
        return res
    end
    n2 = n ÷ 2
    n2 -= n2 % 8
    return pairwise_sum(a, start, n2) + pairwise_sum(a, start + n2, n - n2)
end

# `numpy.nanmean`: the missing values are replaced by zero and left out of the
# count. A series with no value at all is 0 / 0, which is NaN there and here.
function nanmean(a::AbstractVector{Float64})
    mask = isnan.(a)
    arr = copy(a)
    arr[mask] .= 0.0
    cnt = count(!, mask)
    tot = pairwise_sum(arr)
    return tot / cnt
end

"""
    block_average(x)

Remove the block mean of `x`. Returns the series, in the shape it came in.
"""
function block_average(x)
    return x .- nanmean(convert(Vector{Float64}, vec(x)))
end
