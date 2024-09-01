library(mlr3torch)

tc_hist = t_clbk("history")

# access the CallbackSet
cbs_hist = tc_hist$generator$new()

# a custom logger that stores an exponential moving average loss and prints it at the end of every epoch
custom_logger = torch_callback("custom_logger",
    initialize = function(alpha = 0.1) {
        self$alpha = alpha
        self$moving_loss = NULL
    },
    on_batch_end = function() {
        if (is.null(self$moving_training_loss)) {
            self$moving_loss = self$ctx_last_loss
        } else {
            self$moving_loss = self$alpha * self$last_loss + (1 - self$alpha) * self$moving_loss
        }
    },
    on_before_valid = function() {
        cat(sprintf("Epoch %s: %.2f\n", self$ctx$epoch,))
    },
    state_dict = function() {
        self$moving_loss
    },
    load_state_dict = function(state_dict) {
        self$moving_loss = state_dict
    }
)

task = tsk("iris")
mlp = lrn("classif.mlp", callbacks = custom_logger, cb.custom_logger.alpha = 0.05, epochs = 5, batch_size = 64, neurons = 20)

mlp$train(task)

# view the info in the state_dict, i.e. the final moving loss
mlp$model$callbacks$custom_logger