## About

This is an implementation of the perceptron from scratch using python.

>In machine learning, the perceptron (or McCulloch-Pitts neuron) is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.[1] It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

https://en.wikipedia.org/wiki/Perceptron


## How to use

After installing, you can simply:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


X, y = make_classification(
    n_samples=1000, n_features=3, n_informative=3, n_redundant=0
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for activation in ("relu", "sigmoid"):
    for loss in ("mse", "log_loss"):
        print(f"{loss} {activation}: ")
        model = Perceptron(
            activation=activation, loss=loss, epochs=100, learning_rate=0.01
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("\tClassification Report:")
        print(f"\t{classification_report(y_test, y_pred)}")

        print("\tConfusion Matrix:")
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()

        print("\n\n\n\n------------------------------------------\n\n\n\n")
```