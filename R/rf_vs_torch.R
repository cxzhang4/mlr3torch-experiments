library(mlr3torch)
library(mlr3tuning)
library(mlr3learners)

library(mlr3oml)
library(tidytable)

binary_class_tasks = list_oml_data(number_classes = 2,
              number_missing_values = 0)

simple_tasks = binary_class_tasks |>
       filter(NumberOfSymbolicFeatures == 1) |>
       select(data_id, name, NumberOfFeatures, NumberOfInstances)

View(simple_tasks)

# Bischl CC-18 benchmarking suite
library(mlr3oml)
library(tidytable)

cc18_collection = ocl(99)

cc18_simple = list_oml_data(data_id = cc18_collection$data_ids, 
              number_classes = 2,
              number_missing_values = 0)

cc18_simple |>
  filter(NumberOfSymbolicFeatures == 1) |>
  select(data_id, name, NumberOfFeatures, NumberOfInstances) |>
  filter(name %in% c("qsar-biodeg", "madelon", "kc1", "blood-transfusion-service-center", "climate-model-simulation-crashes",
                     "Bioresponse"))
  

# diabetes = otsk(37)
# diabetes$data$data

oml_task = otsk(268)
task = as_task(oml_task)
task$col_info


# phoneme 1489
phoneme_data = odt(1489)
phoneme_data$data
# phoneme = otsk(9952)
phoneme$data
# NASA code 1063
software_defects = otsk(1063)

# click prediction 1219: needs multiple files
click_prediction = otsk(1219)
# tumor_endometrium_colon
tumor_endo_colon = otsk(1133)


tumor_endo_colon

learner = lrn("classif.mlp",
              # define the tuning space via the to_tune() tokens
              
              # use either 16, 32, or 64
              batch_size = to_tune(c(16, 32, 64)),
              # tune the dropout probability in the interval [0.1, 0.9]
              p = to_tune(0.1, 0.9),
              # tune the epochs using early stopping (internal = TRUE)
              epochs = to_tune(upper = 1000L, internal = TRUE),
              # configure the early-stopping / validation
              validate = 0.3,
              measures_valid = msr("classif.acc"),
              patience = 10,
              
              device = "cpu"
)

at = auto_tuner(
  learner = learner,
  tuner = tnr("grid_search"),
  resampling = rsmp("cv"),
  measure = msr("classif.acc"),
  term_evals = 10
)

task = tsk("iris")

at$train(task)

future::plan("multisesssion")

design = benchmark_grid(
  tsk("iris"),
  learners = list(at, lrn("classif.ranger")),
  resampling = rsmp("cv", folds = 10)
)
benchmark(design)

# parallelize the outer resampling, not the inner resampling

# 1. apply learner at to fold 1 of iris (outer)
# 2. apply learner at to fold 2 of iris (outer)
#  the autotuner itself also can parallelize execution (inner)
# ...
# 10. apply learner at to fold 10 of iris (outer)
# 11. apply learner ranger to fold 1 of iris (outer)
# ..
# 20. apply learner ranger to fold 10 of iris (outer)