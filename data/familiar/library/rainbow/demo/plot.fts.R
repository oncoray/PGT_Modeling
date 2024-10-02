## This demo considers functional data visualization using fts().

library(rainbow)

par(mfrow = c(1, 2))
plot(fts(x = 15:49, y = Australiafertility$y, xname = "Age", yname = "Fertility rate"))
plot(fts(x = 15:49, y = Australiasmoothfertility$y, xname = "Age", yname = "Smoothed fertility rate"))

