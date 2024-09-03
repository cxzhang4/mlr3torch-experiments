library(mlr3torch)
library(tfevents)

custom_tf_logger = torch_callback("custom_tf_logger",
  initialize = function() {
    self$last_train_loss = NULL
    # self$last_valid_loss = NULL
  },
  on_epoch_end = function() {
    log_event(train_loss = list(loss = self$ctx$last_loss))
  },
  state_dict = function() {
    self$last_train_loss
  },
  load_state_dict = function(state_dict) {
    self$last_train_loss = state_dict
  }
)

task = tsk("iris")
task$divide(0.3)

mlp = lrn("classif.mlp", callbacks = custom_tf_logger, epochs = 5, batch_size = 64, neurons = 20)

mlp$train(task)
print(mlp$model$callbacks$custom_logger)


