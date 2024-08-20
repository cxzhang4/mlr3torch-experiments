library(here)
library(torch)
library(bench)
library(readr)
library(magrittr)
library(tibble)
library(dplyr)

source(here("R", "get_data.R"))

source(here("R", "mlr3torch", "data_setup.R"))
source(here("R", "mlr3torch", "learner_creation.R"))
source(here("R", "mlr3torch", "learner_training.R"))

source(here("R", "time_training.R"))

source(here("R", "read_bench_results.R"))
source(here("R", "output_dir_name.R"))

config = config::get()

data_dir = here("data", "correlation")
should_download = list.files(data_dir) == 0
get_data(data_dir, should_download)

# TODO: create a validation dataset
# TODO: create a test dataset

train_mlr3torch_ds = create_mlr3torch_dataset(data_dir, config$architecture_id, trn_idx)
train_responses = fread(here(data_dir, "guess-the-correlation", "train_responses.csv"))
response_col_name = "corr"
tsk_gtcorr = create_task_from_ds(train_mlr3torch_ds, train_responses, response_col_name, config$architecture_id)

mlr3torch_learner = create_mlr3torch_learner(tsk_gtcorr, torch_learner,
                                             config$architecture_id, config$batch_size, config$n_epochs,
                                             config$learning_rate, config$accelerator)
mlr3torch_learner$train(tsk_gtcorr)

print("mlr3torch learner:")
if ("GraphLearner" %in% class(mlr3torch_learner)) {
  print(mlr3torch_learner$base_learner()$model$network)
  print(paste("mlr3torch number of epochs:", mlr3torch_learner$base_learner()$param_set$get_values()$epochs))
  print(paste("mlr3torch batch size:", mlr3torch_learner$base_learner()$param_set$get_values()$batch_size, sep = " "))
} else {
  print(mlr3torch_learner$model$network)
  print(paste("mlr3torch number of epochs:", mlr3torch_learner$param_set$get_values()$epochs))
  print(paste("mlr3torch batch size:", mlr3torch_learner$param_set$get_values()$batch_size, sep = " "))
}
