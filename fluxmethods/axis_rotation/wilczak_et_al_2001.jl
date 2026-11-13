# The double rotation of Wilczak et al. (2001), in Julia.
#
# A translation of `double_rotation` in `wilczak_et_al_2001.py`, the file beside
# this one, statement for statement: the same operations in the same order, so
# that the two give the same numbers and not merely close ones. `tests/` holds
# them to that, and `oneflux_preproc` runs both through the same pipeline.
#
# One thing is written out here that Python has in a library. `numpy.nanmean`
# adds a series up pairwise, in blocks of 128 with eight running sums, and
# Julia's `sum` is pairwise too but cuts the series elsewhere; the two differ in
# the last bit, and the rotation angles are made of these means. So
# `pairwise_sum` below is numpy's summation, in numpy's order.
#
# And one thing is taken from where Python and R take it. `numpy` and R compute
# an arctangent, a cosine and a sine with the C library's `atan2`, `cos` and
# `sin`; Julia has its own, written in Julia, which differ from the C library's
# in the last bit for one argument in ten (measured 2026-11-13 on 50 000 random
# arguments against glibc 2.39: `atan` 18 %, `cos` and `sin` 3 %). One bit in
# the angle is one part in 1e16 of the rotated wind, and it is still a
# difference. So the three functions below call the C library where there is
# one to call (Linux, macOS) and are Julia's own where there is not (Windows),
# and `LIBM` says which: the library's name, or "" for Julia's own.
#
# Julia and its standard library `Libdl` only: no package is needed. The file
# defines functions and no module, so it can be `include`d into any.

using Libdl

const LIBM = Libdl.find_library(["libm.so.6", "libm"])

@static if LIBM != ""
    c_atan2(y::Float64, x::Float64) = ccall((:atan2, LIBM), Float64, (Float64, Float64), y, x)
    c_cos(x::Float64) = ccall((:cos, LIBM), Float64, (Float64,), x)
    c_sin(x::Float64) = ccall((:sin, LIBM), Float64, (Float64,), x)
else
    c_atan2(y::Float64, x::Float64) = atan(y, x)
    c_cos(x::Float64) = cos(x)
    c_sin(x::Float64) = sin(x)
end

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
    double_rotation(u, v, w; verbosity=0)

Perform a double rotation correction on the input wind components u, v, w.
The first rotation aligns the mean wind vector with the x-axis, and the second
aligns the mean vertical wind component with the z-axis.

Returns a named tuple: the corrected `u`, `v`, `w` and the angles of the first
(`theta`) and second (`phi`) rotations.
"""
function double_rotation(u, v, w; verbosity = 0)
    # Ensure inputs are vectors of Float64
    u = convert(Vector{Float64}, vec(u))
    v = convert(Vector{Float64}, vec(v))
    w = convert(Vector{Float64}, vec(w))

    # first rotation: align the x-axis with the mean horizontal wind. atan2 (not
    # atan) is required so u aligns to +|wind| for every wind direction; atan
    # collapses the quadrant and flips u to -|wind| whenever mean(u) < 0.
    theta = c_atan2(nanmean(v), nanmean(u))
    u1 = u .* c_cos(theta) .+ v .* c_sin(theta)
    v1 = -u .* c_sin(theta) .+ v .* c_cos(theta)
    w1 = w

    # second rotation: tilt into the mean streamline so mean(w) -> 0.
    phi = c_atan2(nanmean(w1), nanmean(u1))
    u2 = u1 .* c_cos(phi) .+ w1 .* c_sin(phi)
    v2 = v1
    w2 = -u1 .* c_sin(phi) .+ w1 .* c_cos(phi)

    if verbosity > 0
        println("Double rotation angles: theta=$theta, phi=$phi")
    end

    # Return the corrected components and angles, named as the Python object
    # names them.
    return (u = u2, v = v2, w = w2, theta = theta, phi = phi)
end
