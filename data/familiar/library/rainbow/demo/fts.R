## This demo considers creating a functional time series object

library(rainbow)

par(mfrow = c(1, 2))
fts(x = 15:49, y = Australiafertility$y, xname = "Age", yname = "Fertility rate")
fts(x = 15:49, y = Australiasmoothfertility$y, xname = "Age", yname = "Smoothed fertility rate")

