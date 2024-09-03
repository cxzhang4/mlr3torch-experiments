library(mlr3torch)
library(tfevents)

custom_tf_logger = torch_callback("custom_tf_logger",
  initialize = function() {
    self$last_train_loss = NULL
    self$last_valid_loss = NULL
  },
  on_before_valid = function() {
    # logging wrong value
    log_event(train_loss = list(loss = self$ctx$last_loss))
  },
  on_valid_end = function() {
    # logging wrong value
    log_event(valid_loss = list(loss = self$ctx$last_loss))
  }
  # state_dict = function() {
  #   list(train_loss = self$moving_train_loss,
  #        valid_loss = self$moving_valid_loss)
  # },
  # load_state_dict = function(state_dict) {
  #   self$moving_train_loss = state_dict$moving_train_loss
  #   self$moving_valid_loss = state_dict$moving_valid_loss
  # }
)

task = tsk("iris")
task$divide(0.3)

mlp = lrn("classif.mlp", callbacks = custom_tf_logger, epochs = 5, batch_size = 64, neurons = 20)

mlp$train(task)
print(mlp$model$callbacks$custom_logger)


