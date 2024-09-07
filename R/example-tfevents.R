library(tfevents)
library(torch)
library(fs)

epochs <- 10
for (i in seq_len(epochs)) {
  log_event(
    train = list(loss = runif(1), acc = runif(1)),
    valid = list(loss = runif(1), acc = runif(1))
  )
}

fs::dir_tree("logs")

tensorflow::tensorboard(normalizePath("logs"), port = 6060)

