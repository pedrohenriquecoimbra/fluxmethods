# The double rotation of Wilczak et al. (2001), in R.
#
# A translation of `double_rotation` in `wilczak_et_al_2001.py`, the file beside
# this one, statement for statement: the same operations in the same order, so
# that the two give the same numbers and not merely close ones. `tests/` holds
# them to that, and `oneflux_preproc` runs both through the same pipeline.
#
# One thing is written out here that Python has in a library. `numpy.nanmean`
# adds a series up pairwise, in blocks, and R's `mean` adds it up in extended
# precision and corrects it with a second pass; the two differ in the last bit,
# and the rotation angles are made of these means. So `pairwise_sum` below is
# numpy's summation, in numpy's order.
#
# Base R only: nothing is attached and no package is needed.

# numpy's `pairwise_sum` (numpy/_core/src/umath/loops_utils.h.src): fewer than
# eight values are added in turn; up to 128, eight running sums are kept, added
# as ((0+1)+(2+3))+((4+5)+(6+7)), and the remainder added in turn; a longer run
# is cut in two at a multiple of eight and each half summed the same way.
pairwise_sum <- function(a, start = 1L, n = length(a)) {
  if (n < 8L) {
    res <- 0
    for (i in seq_len(n)) res <- res + a[[start + i - 1L]]
    return(res)
  }
  if (n <= 128L) {
    r <- a[start:(start + 7L)]
    i <- 8L
    while (i < n - (n %% 8L)) {
      r <- r + a[(start + i):(start + i + 7L)]
      i <- i + 8L
    }
    res <- ((r[[1L]] + r[[2L]]) + (r[[3L]] + r[[4L]])) +
      ((r[[5L]] + r[[6L]]) + (r[[7L]] + r[[8L]]))
    while (i < n) {
      res <- res + a[[start + i]]
      i <- i + 1L
    }
    return(res)
  }
  n2 <- n %/% 2L
  n2 <- n2 - (n2 %% 8L)
  pairwise_sum(a, start, n2) + pairwise_sum(a, start + n2, n - n2)
}

# `numpy.nanmean`: the missing values are replaced by zero and left out of the
# count. A series with no value at all is 0 / 0, which is NaN there and here.
# `is.na` is true of NaN and of NA, R's two missing values, which are one
# missing value in Python.
nanmean <- function(a) {
  mask <- is.na(a)
  arr <- a
  arr[mask] <- 0
  cnt <- sum(!mask)
  tot <- pairwise_sum(arr)
  tot / cnt
}

# Perform a double rotation correction on the input wind components u, v, w.
# The first rotation aligns the mean wind vector with the x-axis, and the second
# aligns the mean vertical wind component with the z-axis.
#
# Returns a list: the corrected `u`, `v`, `w` and the angles of the first
# (`theta`) and second (`phi`) rotations.
double_rotation <- function(u, v, w, verbosity = 0) {
  # Ensure inputs are numeric vectors
  u <- as.double(u)
  v <- as.double(v)
  w <- as.double(w)

  # first rotation: align the x-axis with the mean horizontal wind. atan2 (not
  # atan) is required so u aligns to +|wind| for every wind direction; atan
  # collapses the quadrant and flips u to -|wind| whenever mean(u) < 0.
  theta <- atan2(nanmean(v), nanmean(u))
  u1 <- u * cos(theta) + v * sin(theta)
  v1 <- -u * sin(theta) + v * cos(theta)
  w1 <- w

  # second rotation: tilt into the mean streamline so mean(w) -> 0.
  phi <- atan2(nanmean(w1), nanmean(u1))
  u2 <- u1 * cos(phi) + w1 * sin(phi)
  v2 <- v1
  w2 <- -u1 * sin(phi) + w1 * cos(phi)

  if (verbosity > 0) {
    cat(sprintf("Double rotation angles: theta=%s, phi=%s\n", theta, phi))
  }

  # Return the corrected components and angles, named as the Python object
  # names them.
  list(u = u2, v = v2, w = w2, theta = theta, phi = phi)
}
