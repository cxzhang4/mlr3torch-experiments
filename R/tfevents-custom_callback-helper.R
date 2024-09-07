library(mlr3torch)
library(tfevents)
library(fs)

# custom_tf_logger = torch_callback("custom_tf_logger",
#   initialize = function() {
#     self$last_train_loss = NULL
#   },
#   on_batch_end = function() {
#     self$last_train_loss = self$ctx$last_loss
#     log_event(train_loss = self$last_train_loss)
#   },
#   state_dict = function() {
#     self$last_train_loss
#   },
#   load_state_dict = function(state_dict) {
#     self$last_train_loss = state_dict
#   }
# )

# TODO: extend to multiple measures
custom_tf_logger_valid = torch_callback("custom_tf_logger",
  on_batch_end = function() {
    if (length(self$ctx$last_scores_train) > 0) {
      log_event(train_measure = self$ctx$last_scores_train[[names(self$ctx$measures_train)]])
    }
    print(self$ctx$last_scores_train)
  },
  on_batch_valid_end = function() {
    if (length(self$ctx$last_scores_valid) > 0) {
      log_event(valid_measure = self$ctx$last_scores_valid[[names(self$ctx$measures_valid)]])
    }
    print(self$ctx$last_scores_valid)
  }
)

task = tsk("iris")

mlp = lrn("classif.mlp", 
          callbacks = custom_tf_logger_valid,
          epochs = 5, batch_size = 64, neurons = 20,
          validate = 0.2, measures_valid = msrs(c("classif.acc")), measures_train = msrs(c("classif.acc"))
      )

mlp$train(task)
# print(mlp$model$callbacks$custom_tf_logger)

fs::dir_tree("logs")

tensorflow::tensorboard(normalizePath("logs"), port = 6060)

# need to restart R session to "start from scratch"
