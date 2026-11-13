# Block-average detrending, in R.
#
# A translation of `block_average` in `commonly_used.py`, the file beside this
# one: the mean of the averaging period is subtracted from the series, a missing
# value being left out of the mean and kept in the series.
#
# The mean is the whole of it, and languages add a series up differently: what
# Python computes is `numpy.nanmean`, pairwise in blocks, and R's `mean` sums in
# extended precision with a second pass. So `pairwise_sum` below is numpy's
# summation in numpy's order, as it is in `axis_rotation/wilczak_et_al_2001.R`.
# It is written out in each file rather than shared, so that a file is all of
# the code a call of it runs, and its digest says so.
#
# Base R only. `linear_detrend` is not translated: its fit is LAPACK's, which no
# file here can follow statement by statement.

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

# Remove the block mean of `x`. Returns the series, in the shape it came in.
block_average <- function(x) {
  x - nanmean(as.double(x))
}
