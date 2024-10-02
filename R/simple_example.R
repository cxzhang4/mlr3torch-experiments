library(mlr3torch)
library(tfevents)
library(here)

source(here("R", "tfevents-custom_callback-helper.R"))

task = tsk("iris")

# output_dir = "logs2"
set_default_logdir("logs")

# custom_tf_logger = CallbackTFEvents$new()
mlp = lrn("classif.mlp", 
          callbacks = custom_tf_logger,
          epochs = 2, batch_size = 50, neurons = 200,
          # validate = 0.2, 
          # measures_valid = msrs(c("classif.acc", "classif.ce")), 
          measures_train = msrs(c("classif.acc", "classif.ce"))
)

mlp$train(task)
# print(mlp$model$callbacks$custom_tf_logger)

tensorflow::tensorboard(normalizePath("logs"), port = 6060)

# need to restart R session to "start from scratch"
events = collect_events()
events$summary[[2]] |> unlist()
events$summary[[2]] |> value()

events_list = events$summary %>%
  mlr3misc::map(unlist)

n_last_loss <- events_list %>%
  mlr3misc::map(\(x) x["tag"] == "last_loss") %>%
  unlist() %>%
  sum()
