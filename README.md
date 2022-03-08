<div align="right">

**рускій воєний корабль - іди нах#й**
<img src="assets/GloryForUkraineMini.svg" width="25">

<div align="left">

> You can support the [Armed Forces of Ukraine](https://bank.gov.ua/en/news/all/natsionalniy-bank-vidkriv-spetsrahunok-dlya-zboru-koshtiv-na-potrebi-armiyi), or contribute to the [Humanitarian Assistance to Ukrainians](https://bank.gov.ua/en/news/all/natsionalniy-bank-vidkriv-rahunok-dlya-gumanitarnoyi-dopomogi-ukrayintsyam-postrajdalim-vid-rosiyskoyi-agresiyi) affected by russia’s aggression. Thank you.

## _LoreDart NN_
Simple library for creating and training Deep Neural Networks, written in pure Dart.

Conceptually, the library has educational and entertainment purposes. Hope you will find it fun to use DNN with loredart_nn.

## Getting started

Just import library into the project.
```
import 'package:loredart_nn/loredart_nn.dart';
```
## NeuralNetwork usage

In the `loredart_nn` data inside the NeuralNetwork is as a (mini)batch Matrix, where each **column** representes one data record and each row representes one feature. So NeuralNetwork uses column-based data representation.

But user's inputs and model predictions are `List<List<double>>`, where each row is a data record.

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
I used the 'flatten' version of MNIST, where the digit's image flattened into the record of 784 pixels.
### Define the model
```dart
var model = NeuralNetwork(
  784, // length of one input record = 784 pixels
  [
    Dense(128, activation: Activation.softplus()), // fully connected layer
    LayerNormalization(), // normalization for each data record in the batch
    Dense(10, activation: Activation.softmax()) // output layer with softmax for probabilities
  ],
  loss: Loss.crossEntropy(), // crossEntropy loss for One-Hot encoded target values
  optimizer: SGD(learningRate: 0.01, momentum: 0.9), // optionally customize SGD optimizer with momentum
  useAccuracyMetric: true // for classification task you can use 'accuracy' metric
);
```
### Train the model
```dart
var history = model.fit(xTrain, yTrain, epochs: 4, batchSize: 256, verbose: true);
```
With `verbose == true`, you will see how the model is updating after each batch of data, and summary information for each epoch, like that:
```
epoch 1/4 |118/118| -> mean time per batch: 388.01ms, mean loss [cross_entropy]: 1.478636, mean accuracy: 68.05%
epoch 2/4 |118/118| -> mean time per batch: 400.87ms, mean loss [cross_entropy]: 0.868227, mean accuracy: 83.50%
epoch 3/4 |118/118| -> mean time per batch: 390.92ms, mean loss [cross_entropy]: 0.691678, mean accuracy: 85.72%
epoch 4/4 |118/118| -> mean time per batch: 386.92ms, mean loss [cross_entropy]: 0.606761, mean accuracy: 86.84%
```
Here you can control batch size and number of epochs. Model's `fit` method returns `history` - Map with buffered information about loss for each epoch.
```dart
print(history);
// {cross_entropy: [1.4786360357701471, 0.868226552200313, 0.6916779963409534, 0.6067611970768912],
//  accuracy: [68.04819915254238, 83.49995586158192, 85.72342867231639, 86.83902718926552]}
```

### Test the model
```dart
  var metrics = model.evaluate(xTest, yTest, verbose: true)
  print(metrics);
  //{mean cross_entropy: 0.5742549607727949, mean accuracy: 0.8719166666666667}
```
Again, with `verbose == true`, you will see a little bit more information:
```
// evaluating batch 100/100 -> mean time per batch: 68.52ms, mean loss [cross_entropy]: 0.574255, mean accuracy: 87.19%
```
Similarly, `evaluate` method returns Map object with stored information about mean loss.
### Use the model
Prediction is performed for many data rows at once, and output is a List of predictions for each input.
```dart
  // data is some List of inputs
  List<List<double>> data = ...;

  List<List<double>> prediction = model.predict(data); // model.predict returns prediction for each row in the data
  print(prediction[0]); 
  // prints something like [0.00, 0.00, 0.00, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
```

### Save and load weights of model
You can save weights and biases of trainable (aka Dense) layers into some directory.
```dart
  // save parameters into the `mnist_classifier/model_weights.bin` file
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

Working Flutter project will be posted on GitHub.

!['1' exmaple](assets/mnist1s.png) !['2' example](assets/mnist2s.png) !['3' example](assets/mnist3s.png) !['5' example](assets/mnist5s.png)

## Additional information
### Supported layers:
- `Dense` - a regular fully connected layer with weights, biases and activation function.
- `LayerNormalization` - the data record normalization layer; supports two normalizations: `min-max` and `z-score`.
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