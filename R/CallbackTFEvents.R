CallbackSetTFLog = R6Class("CallbackSetTFLog",
                           inherit = CallbackSet,
                           lock_objects = TRUE,
                           public = list(
                             on_batch_end = function() {
                               log_helper_train = function(measure_name) {
                                 expr <- paste0("log_event(train.", measure_name, " = self$ctx$last_scores_train[[\"", measure_name, "\"]])")
                                 eval(parse(text = expr))
                               }
                               
                               if (length(self$ctx$last_scores_train) > 0) {
                                 map(names(self$ctx$measures_train), log_helper_train)
                               }
                             },
                             on_epoch_end = function() {
                               log_helper_valid = function(measure_name) {
                                 expr <- paste0("log_event(valid.", measure_name, " = self$ctx$last_scores_valid[[\"", measure_name, "\"]])")
                                 eval(parse(text = expr))
                               }
                               
                               if (length(self$ctx$last_scores_valid) > 0) {
                                 map(names(self$ctx$measures_valid), log_helper_valid)
                               }
                             }
                           )
                          )

# 
# mlr3torch_callbacks$add()