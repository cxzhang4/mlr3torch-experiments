library(tfevents)
library(here)
library(mlr3torch)

source(here("R", "CallbackTFEvents.R"))

# devtools::load_all("~/mlr3_hiwi/mlr3torch")

task = tsk("iris")

custom_tf_logger = CallbackSetTFLog$new()
pth0 = tempfile()
mlp = lrn("classif.mlp", 
          callbacks = t_clbk("tb"),
          epochs = 2, batch_size = 50, neurons = 200,
          validate = 0.2,
          measures_valid = msrs(c("classif.acc", "classif.ce")),
          measures_train = msrs(c("classif.acc", "classif.ce"))
)
mlp$param_set$set_values(cb.tb.path = pth0)

mlp$train(task)
# print(mlp$model$callbacks$custom_tf_logger)


# tensorflow::tensorboard(normalizePath("logs"), port = 6060)

# need to restart R session to "start from scratch"
events = collect_events(pth0)$summary %>%
  mlr3misc::map(unlist)
events$summary[[2]] |> unlist()
events$summary[[2]] |> value()

events_list = events$summary %>%
  mlr3misc::map(unlist)

n_last_loss <- events_list %>%
  mlr3misc::map(\(x) x["tag"] == "last_loss") %>%
  unlist() %>%
  sum()
