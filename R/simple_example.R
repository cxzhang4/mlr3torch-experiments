library(mlr3torch)
library(tfevents)
library(fs)
library(here)

source(here("R", "tfevents-custom_callback-helper.R"))
# source("CallbackTFEvents.R")

task = tsk("iris")

# custom_tf_logger = CallbackTFEvents$new()
mlp = lrn("classif.mlp", 
          callbacks = custom_tf_logger,
          epochs = 10, batch_size = 64, neurons = 200,
          validate = 0.2, 
          measures_valid = msrs(c("classif.acc", "classif.ce")), 
          measures_train = msrs(c("classif.acc", "classif.ce"))
)

mlp$train(task)
# print(mlp$model$callbacks$custom_tf_logger)

fs::dir_tree("logs")

tensorflow::tensorboard(normalizePath("logs"), port = 6060)

# need to restart R session to "start from scratch"
