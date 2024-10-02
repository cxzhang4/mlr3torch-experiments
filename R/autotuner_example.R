library(mlr3torch)
library(mlr3tuning)

learner = lrn("classif.mlp",
              batch_size = to_tune(c(16, 32, 64)),
              device = "cpu",
              epochs = to_tune(upper = 1000L, internal = TRUE),
              p = to_tune(0.1, 0.9)
              )

at = auto_tuner(
  learner = learner,
  tuner = tnr("grid_search"),
  resampling = rsmp("holdout"),
  measure = msr("classif.acc"),
  term_evals = 10
  )

# the autotuner is just another learner