import 'dart:io' show File;

import 'package:csv/csv.dart' show CsvToListConverter;

import 'package:loredart_nn/loredart_nn.dart';

void main() {
  List<List<List<double>>> data = createTrainData(CsvToListConverter()
    .convert(
      File('example/.csv' //name of the .csv file with data
              )
          .readAsStringSync(),
      shouldParseNumbers: false,
    )
    .sublist(1) // first line of .csv file is column names
  );

  /// spliting data into train and test lists
  List<List<double>> xTrain = data[0].sublist(0, 30000);
  List<List<double>> yTrain = data[1].sublist(0, 30000);
  List<List<double>> xTest = data[0].sublist(30000);
  List<List<double>> yTest = data[1].sublist(30000);
  data.clear();

  /// [NeuralNetwork] with 2 hidden layers, MSE loss
  NeuralNetwork mnistDNN = NeuralNetwork(
    784,
    [
      Dense(128, activation: Activation.relu()),
      LayerNormalization(),
      Dense(64, activation: Activation.relu()),
      Dense(10, activation: Activation.linear())
    ],
    loss: Loss.mse(),
    optimizer: SGD(learningRate: 0.05, momentum: 0.8),
    useAccuracyMetric: true
  );

  var history = mnistDNN.fit(xTrain, yTrain, epochs: 3, batchSize: 128, verbose: true);
  print(history);
  
  var metrics = mnistDNN.evaluate(xTest, yTest, batchSize: 120, verbose: true);
  print(metrics);

  /// [NeuralNetwork] with 1 hidden layer and crossEntropy
  NeuralNetwork ceMnistDNN = NeuralNetwork(
    784, // length of one input record = 784 pixels
    [
      Dense(128, activation: Activation.softplus()), //or leakyReLU or elu
      LayerNormalization(),
      Dense(10, activation: Activation.softmax())
    ],
    loss: Loss.crossEntropy(), // sparseCrossEntropy can be used for sparse data
    optimizer: SGD(learningRate: 0.01, momentum: 0.9),
    useAccuracyMetric: true
  );

  var history2 = ceMnistDNN.fit(xTrain, yTrain, epochs: 4, batchSize: 256, verbose: true);
  // epoch 1/4 |118/118| -> mean time per batch: 388.01ms, mean loss [cross_entropy]: 1.478636, mean accuracy: 68.05%
  // epoch 2/4 |118/118| -> mean time per batch: 400.87ms, mean loss [cross_entropy]: 0.868227, mean accuracy: 83.50%
  // epoch 3/4 |118/118| -> mean time per batch: 390.92ms, mean loss [cross_entropy]: 0.691678, mean accuracy: 85.72%
  // epoch 4/4 |118/118| -> mean time per batch: 386.92ms, mean loss [cross_entropy]: 0.606761, mean accuracy: 86.84%

  print(history2);
  // {cross_entropy: [1.4786360357701471, 0.868226552200313, 0.6916779963409534, 0.6067611970768912],
  //  accuracy: [68.04819915254238, 83.49995586158192, 85.72342867231639, 86.83902718926552]}

  var metrics2 = ceMnistDNN.evaluate(xTest, yTest, batchSize: 120, verbose: true);
  // evaluating batch 100/100 -> mean time per batch: 68.52ms, mean loss [cross_entropy]: 0.574255, mean accuracy: 87.19%

  print(metrics2);
  // {mean cross_entropy: 0.5742549607727949, mean accuracy: 0.8719166666666667}
}

/// Split data into features (or pixels) `x` and target digits (or classes) `y`, which are One-Hot encoded
List<List<List<double>>> createTrainData(List<List<dynamic>> data) {
  List<List<double>> x = [];
  List<double> y = [];
  for (var data1 in data) {
    y.add(double.parse(data1[0].toString()));
    x.add(
        data1.sublist(1).map((e) => double.parse(e.toString()) / 255).toList());
  }
  return [x, oneHotEncodingForMnist(y)];
}

/// Return One-Hot encoded List for mnist digits
List<List<double>> oneHotEncodingForMnist(List<double> numbers) {
  List<List<double>> encoded = [];
  for (double number in numbers) {
    List<double> temp = List<double>.filled(10, 0);
    temp[number.toInt()] = 1;
    encoded.add(temp);
  }
  return encoded;
}
