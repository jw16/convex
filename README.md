# convex
Convolutional Neural Net Example

The data.h5 file contains 10000 200-sample signals.   The example.py script loads the data, creates an 80/20 train/test split, builds a simple CNN model in Keras, then fits and evaluates the model.

```
python3 example.py
```

The data is randomly shuffled so output will differ between runs, but the model should obtain test accuracies in the 75-85% range

