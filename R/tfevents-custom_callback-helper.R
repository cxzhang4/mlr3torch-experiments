library(mlr3torch)
library(tfevents)
library(fs)
library(mlr3misc)
library(checkmate)

custom_tf_logger = torch_callback("custom_tf_logger",
  # TODO: checkmate assertions
  # initialize = function() {
    # self$path = assert_path_for_output(path)
    # set_default_logdir(path)
  # },
  # training measure logged after every batch 
  on_batch_end = function() {
    # TODO: determine whether you can refactor this and the 
    # validation one into a single function
    # need to be able to access self$ctx
    log_event(last_loss = self$ctx$last_loss)
  },
  #' @description
  #' Logs the validation measures as TensorFlow events.
  #' Meaningful changes happen at the end of each epoch.
  #' Notably NOT on_batch_valid_end, since there are no gradient steps between validation batches,
  #' and therefore differences are due to randomness
  # TODO: log last_scores_train here
  # TODO: display the appropraite x axis with its label in TensorBoard
  # relevant when we log different scores at different times
  on_epoch_end = function() {
    log_valid_score = function(measure_name) {
      valid_score = list(self$ctx$last_scores_valid[[measure_name]])
      names(valid_score) = paste0("valid.", measure_name)
      do.call(log_event, valid_score)
    }
    
    log_train_score = function(measure_name) {
      # TODO: change this to use last_loss
      train_score = list(self$ctx$last_scores_train[[measure_name]])
      names(train_score) = paste0("train.", measure_name)
      do.call(log_event, train_score)
    }
    
    if (length(self$ctx$last_scores_train)) {
      map(names(self$ctx$measures_train), log_train_score)
    }
    
    if (length(self$ctx$last_scores_valid)) {
      map(names(self$ctx$measure_valid), log_valid_score)
    }
  })

# need to restart R session to "start from scratch"
