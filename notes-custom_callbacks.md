The torch callback mechanism allows one to customize the training loop of a neural network. We describe how to write a custom callback for a network trained using `mlr3torch`.

There are three `R6` classes that are "at the heart of the callback mechanism":

- `CallbackSet`: contains the methods that are executed at different stages of the training loop

- `TorchCallback` wraps a `CallbackSet` and annotates it with meta information, including a `ParamSet`

- `ContextTorch` defines the information in the training process that is accessible by `CallbackSet`

Generally, when using predefined callbacks, one usually only interacts with the `TorchCallback` class. You can construct an instance using `t_clbk(<id>)`.

A callback can execute code at the following stages of the training process:

- "on_begin"

- "on_epoch_begin"

- "on_batch_begin"

- "on_after_backward"

- "on_batch_end"

- "on_before_valid"

- "on_batch_valid_begin"

- "on_batch_valid_end"

- "on_valid_end"

- "on_epoch_end"

- "on_end"

- "on_exit"

The `torch_callback()` helper function can be used to define a custom callback.