library(mlr3torch)
task = tsk("iris")

iris = task$data()

# advice: comment out everything after the layer you are interested in
# x %>% 
#   self$conv1() %>% 
#   nnf_relu() %>%
#   nnf_max_pool2d(2) %>%


library(torchvision)
set.seed(777)
torch_manual_seed(777)

dir <- here::here("data")

train_ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  transform = function(x) {
    x %>%
      transform_to_tensor() 
  }
)

valid_ds <- tiny_imagenet_dataset(
  dir,
  split = "val",
  transform = function(x) {
    x %>%
      transform_to_tensor()
  }
)

train_dl <- dataloader(train_ds,
                       batch_size = 128,
                       shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 128)

batch <- train_dl %>%
  dataloader_make_iter() %>%
  dataloader_next()

dim(batch$x)

y_pred = model(batch$x)

# begin copy
convnet <- nn_module(
  "convnet",
  initialize = function() {
    self$features <- nn_sequential(
      nn_conv2d(3, 64, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_adaptive_avg_pool3d(c(16, 32)),
      # nn_conv2d(64, 128, kernel_size = 3, padding = 1),
      # nn_relu(),
      # nn_max_pool2d(kernel_size = 2),
      # nn_conv2d(128, 256, kernel_size = 3, padding = 1),
      # nn_relu(),
      # nn_max_pool2d(kernel_size = 2),
      # nn_conv2d(256, 512, kernel_size = 3, padding = 1),
      # nn_relu(),
      # nn_max_pool2d(kernel_size = 2),
      # nn_conv2d(512, 1024, kernel_size = 3, padding = 1),
      # nn_relu(),
      # nn_adaptive_avg_pool2d(c(1, 1))
    )
    self$classifier <- nn_sequential(
      nn_linear(1024, 1024),
      nn_relu(),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_linear(1024, 200)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    # x <- self$features(x)$squeeze()
    # x <- self$classifier(x)
    x
  }
)
model <- convnet()
y_pred = model(batch$x)
dim(batch$x)
dim(y_pred)



opt <- optim_adam(model$parameters)

### training loop --------------------------------------

for (t in 1:3) {
  
  ### -------- Forward pass --------
  y_pred <- model(x)
  
  ### -------- Compute loss -------- 
  loss <- nnf_mse_loss(y_pred, y)
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation --------
  opt$zero_grad()
  loss$backward()
  
  ### -------- Update weights -------- 
  opt$step()
  
}
