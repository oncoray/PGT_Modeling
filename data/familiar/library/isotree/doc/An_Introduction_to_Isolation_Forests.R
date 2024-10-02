## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
    collapse = TRUE,
    comment = "#>"
)
options(rmarkdown.html_vignette.check_title = FALSE)

## ---- include = FALSE---------------------------------------------------------
### Don't overload CRAN servers
### https://stackoverflow.com/questions/28961431/computationally-heavy-r-vignettes
is_check <- ("CheckExEnv" %in% search()) || any(c("_R_CHECK_TIMINGS_",
             "_R_CHECK_LICENSE_") %in% names(Sys.getenv()))

## ---- fig.width=3.5, fig.height=2.2-------------------------------------------
set.seed(123)
random_numbers <- matrix(rnorm(1000))
par(oma = c(0,0,0,0), mar = c(4,4,3,2))
hist(random_numbers, breaks=50, col="navy",
     main="Randomly-generated numbers\nfrom normal distribution",
     xlab="value")

## ---- fig.width=5, fig.height=3-----------------------------------------------
library(isotree)

model <- isolation.forest(random_numbers, ndim=1, ntrees=10, nthreads=1)
scores <- predict(model, random_numbers, type="avg_depth")
par(mar = c(4,5,3,2))
plot(random_numbers, scores, type="p", col="darkred",
     main="Average isolation depth\nfor normally-distributed numbers",
     xlab="value", ylab="Average isolation depth")

## ---- fig.width=4, fig.height=3-----------------------------------------------
### Randomly-generated data from different distributions
set.seed(1)
cluster1 <- data.frame(
    x = rnorm(1000, -1, .4),
    y = rnorm(1000, -1, .2)
)
cluster2 <- data.frame(
    x = rnorm(1000, +1, .2),
    y = rnorm(1000, +1, .4)
)
outlier <- data.frame(
    x = -1,
    y =  1
)

### Putting them together
X <- rbind(cluster1, cluster2, outlier)

### Function to produce a heatmap of the scores
pts = seq(-3, 3, .1)
space_d <- expand.grid(x = pts, y = pts)
plot.space <- function(Z, ttl, cex.main = 1.4) {
    image(pts, pts, matrix(Z, nrow = length(pts)),
          col = rev(heat.colors(50)),
          main = ttl, cex.main = cex.main,
          xlim = c(-3, 3), ylim = c(-3, 3),
          xlab = "", ylab = "")
    par(new = TRUE)
    plot(X, type = "p", xlim = c(-3, 3), ylim = c(-3, 3),
         col = "#0000801A",
         axes = FALSE, main = "",
         xlab = "", ylab = "")
}

model <- isolation.forest(X, ndim=1, ntrees=100, nthreads=1)
scores <- predict(model, space_d)
par(mar = c(2.5,2.2,2,2.5))
plot.space(scores, "Outlier Scores\n(clustered data with an outlier on top)", 1.0)

## ---- eval=FALSE--------------------------------------------------------------
#  par(mfrow = c(3, 2), mar = c(2.5,2.2,2,2.5))
#  
#  iforest <- isolation.forest(
#      X, ndim=1, ntrees=100,
#      missing_action="fail"
#  )
#  plot.space(
#      predict(iforest, space_d),
#      "Isolation Forest"
#  )
#  ext_iforest <- isolation.forest(
#      X, ndim=2, ntrees=100,
#      missing_action="fail"
#  )
#  plot.space(
#      predict(ext_iforest, space_d),
#      "Extended Isolation Forest"
#  )
#  sciforest <- isolation.forest(
#      X, ndim=2, ntrees=100,
#      missing_action="fail",
#      coefs="normal",
#      prob_pick_avg_gain=1
#  )
#  plot.space(
#      predict(sciforest, space_d),
#      "SCiForest"
#  )
#  fcf <- isolation.forest(
#      X, ndim=2, ntrees=100,
#      missing_action="fail",
#      prob_pick_pooled_gain=1
#  )
#  plot.space(
#      predict(fcf, space_d),
#      "Fair-Cut Forest"
#  )
#  dens_iforest <- isolation.forest(
#      X, ndim=2, ntrees=100,
#      missing_action="fail",
#      scoring_metric="density"
#  )
#  plot.space(
#      predict(dens_iforest, space_d),
#      "Density Isolation Forest"
#  )
#  bdens_iforest <- isolation.forest(
#      X, ndim=1, ntrees=100,
#      missing_action="fail",
#      scoring_metric="boxed_ratio"
#  )
#  plot.space(
#      predict(bdens_iforest, space_d),
#      "Boxed Isolation Forest"
#  )

## ---- echo=FALSE, fig.width=5, fig.height=6-----------------------------------
par(mfrow = c(3, 2), mar = c(2.5,2.2,2,2.5))

if (!is_check) {
    iforest <- isolation.forest(
        X, ndim=1, ntrees=100,
        missing_action="fail"
    )
    ext_iforest <- isolation.forest(
        X, ndim=2, ntrees=100,
        missing_action="fail"
    )
    sciforest <- isolation.forest(
        X, ndim=2, ntrees=100,
        missing_action="fail",
        coefs="normal",
        prob_pick_avg_gain=1
    )
    fcf <- isolation.forest(
        X, ndim=2, ntrees=100,
        missing_action="fail",
        prob_pick_pooled_gain=1
    )
    dens_iforest <- isolation.forest(
        X, ndim=2, ntrees=100,
        missing_action="fail",
        scoring_metric="density"
    )
    bdens_iforest <- isolation.forest(
        X, ndim=1, ntrees=100,
        missing_action="fail",
        scoring_metric="boxed_ratio"
    )
} else {
    iforest <- isolation.forest(
        X, ndim=1, ntrees=10,
        sample_size=32, nthreads=1,
        missing_action="fail"
    )
    ext_iforest <- isolation.forest(
        X, ndim=2, ntrees=10,
        sample_size=32, nthreads=1,
        missing_action="fail"
    )
    sciforest <- isolation.forest(
        X, ndim=2, ntrees=10,
        sample_size=32, nthreads=1,
        missing_action="fail",
        coefs="normal",
        prob_pick_avg_gain=1
    )
    fcf <- isolation.forest(
        X, ndim=2, ntrees=10,
        sample_size=32, nthreads=1,
        missing_action="fail",
        prob_pick_pooled_gain=1
    )
    dens_iforest <- isolation.forest(
        X, ndim=2, ntrees=10,
        sample_size=32, nthreads=1,
        missing_action="fail",
        scoring_metric="density"
    )
    bdens_iforest <- isolation.forest(
        X, ndim=1, ntrees=10,
        sample_size=32, nthreads=1,
        missing_action="fail",
        scoring_metric="boxed_ratio"
    )
}
plot.space(
    predict(iforest, space_d),
    "Isolation Forest"
)
plot.space(
    predict(ext_iforest, space_d),
    "Extended Isolation Forest"
)
plot.space(
    predict(sciforest, space_d),
    "SCiForest"
)
plot.space(
    predict(fcf, space_d),
    "Fair-Cut Forest"
)
plot.space(
    predict(dens_iforest, space_d),
    "Density Isolation Forest"
)
plot.space(
    predict(bdens_iforest, space_d),
    "Boxed Isolation Forest"
)

## -----------------------------------------------------------------------------
library(mlbench)

data("Satellite")
is_outlier <- Satellite$classes %in% c("damp grey soil", "cotton crop", "vegetation stubble")
sat_without_class <- Satellite[, names(Satellite)[names(Satellite) != "classes"]]
dim(sat_without_class)

## -----------------------------------------------------------------------------
summary(is_outlier)

## ---- eval=FALSE--------------------------------------------------------------
#  library(MLmetrics)
#  library(kableExtra)
#  
#  model_orig <- isolation.forest(
#      sat_without_class,
#      ndim=1, sample_size=256,
#      ntrees=100,
#      missing_action="fail"
#  )
#  pred_orig <- predict(model_orig, sat_without_class)
#  
#  model_dens <- isolation.forest(
#      sat_without_class,
#      ndim=1, sample_size=256,
#      ntrees=100,
#      missing_action="fail",
#      scoring_metric="density"
#  )
#  pred_dens <- predict(model_dens, sat_without_class)
#  
#  model_fcf <- isolation.forest(
#      sat_without_class,
#      ndim=1, sample_size=32,
#      prob_pick_pooled_gain=1,
#      ntrees=100,
#      missing_action="fail"
#  )
#  pred_fcf <- predict(model_fcf, sat_without_class)
#  
#  results_df <- data.frame(
#      Model = c(
#          "Isolation Forest",
#          "Density Isolation Forest",
#          "Fair-Cut Forest"
#      ),
#      AUROC = c(
#          AUC(pred_orig, is_outlier),
#          AUC(pred_dens, is_outlier),
#          AUC(pred_fcf, is_outlier)
#      )
#  )
#  results_df %>%
#      kable() %>%
#      kable_styling()

## ---- echo=FALSE, message=FALSE-----------------------------------------------
library(MLmetrics)
library(kableExtra)
if (!is_check) {
    model_orig <- isolation.forest(
        sat_without_class,
        ndim=1, sample_size=256,
        ntrees=100,
        missing_action="fail"
    )
    model_dens <- isolation.forest(
        sat_without_class,
        ndim=1, sample_size=256,
        ntrees=100,
        missing_action="fail",
        scoring_metric="density"
    )
    model_fcf <- isolation.forest(
        sat_without_class,
        ndim=1, sample_size=32,
        prob_pick_pooled_gain=1,
        ntrees=100,
        missing_action="fail"
    )
} else {
    model_orig <- isolation.forest(
        sat_without_class,
        ndim=1, sample_size=32, nthreads=1,
        ntrees=10,
        missing_action="fail"
    )
    model_dens <- isolation.forest(
        sat_without_class,
        ndim=1, sample_size=32, nthreads=1,
        ntrees=10,
        missing_action="fail",
        scoring_metric="density"
    )
    model_fcf <- isolation.forest(
        sat_without_class,
        ndim=1, sample_size=32, nthreads=1,
        prob_pick_pooled_gain=1,
        ntrees=10,
        missing_action="fail"
    )
}
pred_orig <- predict(model_orig, sat_without_class)
pred_dens <- predict(model_dens, sat_without_class)
pred_fcf <- predict(model_fcf, sat_without_class)

results_df <- data.frame(
    Model = c(
        "Isolation Forest",
        "Density Isolation Forest",
        "Fair-Cut Forest"
    ),
    AUROC = c(
        AUC(pred_orig, is_outlier),
        AUC(pred_dens, is_outlier),
        AUC(pred_fcf, is_outlier)
    )
)
results_df %>%
    kable() %>%
    kable_styling()

## ---- eval=!is_check----------------------------------------------------------
library(kernlab)

model_svm <- ksvm(
    as.matrix(sat_without_class),
    type="one-svc",
    nu = 0.5
)
pred_svm <- predict(model_svm, as.matrix(sat_without_class), type="decision")
results_svm <- data.frame(
    Model = "One-Class SVM",
    AUROC = AUC(-pred_svm, is_outlier)
)
results_svm %>%
    kable() %>%
    kable_styling()

