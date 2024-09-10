library(mlr3torch)
library(tfevents)
library(fs)
library(mlr3misc)
library(checkmate)

custom_tf_logger = torch_callback("custom_tf_logger",
  # TODO: checkmate assertions
  initialize = function(path = get_default_logdir()) {
    self$path = assert_path_for_output(path)
    set_default_logdir(path)
  },
  # training measure logged after every batch 
  on_batch_end = function() {
    # TODO: change to do.call() or mlr3misc::invoke()
    log_train_score = function(measure_name) {
      train_score = list(self$ctx$last_scores_train[[measure_name]])
      names(train_score) = paste0("train.", measure_name)
      do.call(log_event, train_score)
    }
    
    if (length(self$ctx$last_scores_train)) {
      map(names(self$ctx$measures_train), log_train_score)
    }
  },
  # different model only at the end of an epoch
  # because there are no gradient steps between batch validations
  # TODO: display the x axis
  on_epoch_end = function() {
    log_valid_score = function(measure_name) {
      valid_score = list(self$ctx$last_scores_valid[[measure_name]])
      names(valid_score) = paste0("valid.", measure_name)
      do.call(log_event, valid_score)
    }

    if (length(self$ctx$last_scores_valid)) {
      map(names(self$ctx$measures_valid), log_valid_score)
    }
  }
)

# need to restart R session to "start from scratch"
