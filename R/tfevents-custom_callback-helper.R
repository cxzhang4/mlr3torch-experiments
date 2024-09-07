library(mlr3torch)
library(tfevents)
library(fs)
library(mlr3misc)

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

# for each name in the vector measure_names := names(self$ctx$measures_train)
  # log_event(measure_names[i] = self$ctx$last_scores_train[[measure_names[i]]])

log_helper_train = function(measure_name) {
  expr <- paste0("log_event(", measure_name, " = self$ctx$last_scores_train[[", measure_name, "]])")
  expr
}

log_helper_valid = function(measure_name) {
  expr <- paste0("log_event(", measure_name, " = self$ctx$last_scores_valid[[", measure_name, "]])")
  expr
}

# implementation for one measure
# custom_tf_logger_valid = torch_callback("custom_tf_logger",
#   on_batch_end = function() {
#     if (length(self$ctx$last_scores_train) > 0) {
#       log_event(train_measure = self$ctx$last_scores_train[[names(self$ctx$measures_train)]])
#     }
#   },
#   on_batch_valid_end = function() {
#     if (length(self$ctx$last_scores_valid) > 0) {
#       log_event(valid_measure = self$ctx$last_scores_valid[[names(self$ctx$measures_valid)]])
#     }
#   }
# )

# TODO: extend to multiple measures
custom_tf_logger_valid = torch_callback("custom_tf_logger",
  on_batch_end = function() {
    # if (length(self$ctx$last_scores_train) > 0) {
    #   log_event(train_measure = self$ctx$last_scores_train[[names(self$ctx$measures_train)]])
    # }
    if (length(self$ctx$last_scores_train) > 0) {
      # map(names(self$ctx$measures_train), )
      eval(parse(text = log_helper_train("classif.acc")))
    }
  },
  on_batch_valid_end = function() {
    # if (length(self$ctx$last_scores_valid) > 0) {
    #   log_event(valid_measure = self$ctx$last_scores_valid[[names(self$ctx$measures_valid)]])
    # }
    if (length(self$ctx$last_scores_valid) > 0) {
      eval(parse(text = log_helper_valid("classif.acc")))
    }
  }
)

# need to restart R session to "start from scratch"
