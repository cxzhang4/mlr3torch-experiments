devtools::load_all("../mlr3torch")

po_test = po("nn_adaptive_avg_pool1d", output_size = 10)
task = tsk("iris")
graph = po("torch_ingress_num") %>>%
  po("nn_unsqueeze", dim = 2) %>>%
  po_test
expect_pipeop_torch(graph, "nn_adaptive_avg_pool1d", task)

nn_sequential(
  nn_linear(4),
  torch_unqueeze()
)

m <- nn_adaptive_avg_pool1d(5)
input <- torch_randn(12, 64, 8)
output <- m(input)


  po_test = po("nn_avg_pool1d", kernel_size = 2)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_avg_pool1d", task)

input_image <- torch_randn(1, 1, 3, 3)

# Define the adaptive average pooling layer
adaptive_pool <- nn_adaptive_avg_pool2d(c(2, 2))

# Apply adaptive average pooling
output_image <- adaptive_pool(input_image)

# Print the shapes of input and output
cat("Input shape:", paste(dim(input_image), collapse = "x"), "\n")
cat("Output shape:", paste(dim(output_image), collapse = "x"), "\n")

# Print the input and output tensors
print("Input image:")
print(input_image)
print("Output image:")
print(output_image)



"In deep learning, we work with batches; thus, there is an additional dimension – the very first one – that refers to batch number.

Let’s look at an example image that may be used with the above template:

img <- torch_randn(1, 1, 64, 64)"

library(mlr3torch)
task = tsk("iris")
task
