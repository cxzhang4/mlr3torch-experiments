library(mlr3torch)
library(tfevents)
library(fs)
library(mlr3misc)

custom_tf_logger = torch_callback("custom_tf_logger",
  on_batch_end = function() {
    log_helper_train = function(measure_name) {
      expr <- paste0("log_event(train.", measure_name, " = self$ctx$last_scores_train[[\"", measure_name, "\"]])")
      eval(parse(text = expr))
    }
    
    if (length(self$ctx$last_scores_train) > 0) {
      map(names(self$ctx$measures_train), log_helper_train)
    }
  },
  on_batch_valid_end = function() {
    log_helper_valid = function(measure_name) {
      expr <- paste0("log_event(valid.", measure_name, " = self$ctx$last_scores_valid[[\"", measure_name, "\"]])")
      eval(parse(text = expr))
    }

    if (length(self$ctx$last_scores_valid) > 0) {
      map(names(self$ctx$measures_valid), log_helper_valid)
    }
  }
)

# need to restart R session to "start from scratch"


# OLD: implementation for one measure
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

