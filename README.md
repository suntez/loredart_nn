
### _LoreDart NN_
Simple library for creating and training Deep Neural Networks, written in pure Dart.

Conceptually, the library has educational and entertainment purposes. Hope you will find it fun to use DNN with loredart_nn.

## Getting started

Just import library into the project.
```
import 'package:loredart_nn/loredart_nn.dart';
```
## NeuralNetwork usage

Here is a small example of creating MNIST classification Deep Neural Network:

### Load the data
```dart
  // List of 784 pixels for each digit
  List<List<double>> xTrain = data[0].sublist(0,30000);
  // One-Hot encoded digits' label 
  List<List<double>> yTrain = data[1].sublist(0,30000);
  List<List<double>> xTest = data[0].sublist(30000);
  List<List<double>> yTest = data[1].sublist(30000);
```
I used the 'flatten' version of MNIST, where the digit's image flattened into the row of 784 pixels.
### Define the model
```dart
var model = NeuralNetwork(
    784, // input len = 784 pixels
    [
      Dense(64, activation: Activation.softplus()), //fully connected layer
      Normalization(), // min-max normalization of the data
      Dense(10, activation: Activation.softmax()) // output layers with softmax
    ],
    loss: Loss.crossEntropy(), // cross entropt for One-Hot encoded target values
    optimizer: SGD(learningRate: 0.01, momentum: 0.9), // optionally customize SGD optimizer with momentum > 0
    useAccuracyMetric: true // for classification task you can use 'accuracy' metric
);
```
### Train the model
```dart
  model.fit(xTrain, yTrain, epochs: 1, verbose: true);
```
With `verbose == true`, you will see how the model is updating after each step, and summary information for each epoch, like that:
```
epoch 1/1 |100.00%| -> mean secs per train step: 0.00347s, mean loss [cross_entropy]: 0.332344, accuracy: 90.12%
```
For now, batch size of training is 1, which means that model will calculate gradients for each row of data and update weights and biases of trainable layers after each training step.

### Test the model
```dart
  history = model.evaluate(xTest, yTest, verbose: true)
  print(history); // Output is a Map: {'mean cross_entropy': 0.237818, 'accuracy': 0.927833} 
```
Again, with `verbose == true`, you will see a little bit more information:
```
evaluating 12000/12000 |100.00%| -> mean secs per test step: 0.00025s, mean loss [cross_entropy]: 0.237819, accuracy: 92.78%
```
### Use the model
Prediction is performed for many data rows at once, and output is a List of predictions for each input.
```dart
  // data is some List of inputs
  final prediction = model.predict(data); // model.predict returns prediction for each row in data 
  print(prediction[0].flattenList()); 
  // prints something like [0.00, 0.00, 0.00, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
```
> The output of the single prediction is the `Matrix` instance

To extract values from Matrix you can use `Matrix().matrix` getter or `Matrix().flattenList()` method. Additionally you can read docs from `Matrix` class.
### Save and load weights of model
You can save weights and biases of trainable (aka Dense) layers into some directory.
```dart
  // save parametrs into the `mnist_classifier/model_weights.bin` file
  model.saveWeights('mnist_classifier')
```
> `saveWeights` method saves only weights and biases of Dense layers

Then you can load weights and biases into the model, but be sure you use appropriate architecture, or the results of the loaded model won't meet expectations.

```dart
 model.loadWeights('mnist_classifier')
```
If you want to load weights from Flutter assets, use the `loadWeightsFromBytes` method.
```dart
  var model = NeuralNetwork(...)
   rootBundle.load('assets/model/model_weights.bin').then((value) {
     model.loadWeightsFromBytes(value.buffer);
   });
```
## Flutter example with NeuralNetwork model
Simple example of using MNIST classifier within Flutter app.

Working Flutter project can be found under the `example` folder.

!['1' exmaple](assets/mnist1s.png) !['2' example](assets/mnist2s.png) !['3' example](assets/mnist3s.png) !['5' example](assets/mnist5s.png)

## Additional information
### Supported layers:
- `Dense` - a regular fully connected layer with weights, biases and activation function.
- `Normalization` - the data normalization layer; supports two normalizations: `min-max` and `z-score`.
- `Input` - a special layer that is generated by the NeuralNetwork.

### Supported activations:
- `Activation.sigmoid()` - Sigmoid activation function
- `Activation.swish()`  - Swish activation function
- `Activation.softplus()`  - Softplus activation function
- `Activation.softmax()`  - Softmax activation function
- `Activation.relu()`  - ReLU activation function
- `Activation.leakyReLU()`  - Leaky ReLU activation function
- `Activation.elu()`  - ELU activation function

### Supported losses:
- `Loss.mae()` - Mean Absolute Error loss
- `Loss.mse()` - Mean Square Error loss
- `Loss.crossEntropy()` - Cross Entropy loss
- `Loss.sparseCrossEntropy()` - Sparse Cross Entropy loss
### Supported optimizers:
For now, `loredart_nn` has only one optimizer: 
- `SGD with momentum`.