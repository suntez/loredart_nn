import 'dart:io' show File;

import 'package:csv/csv.dart' show CsvToListConverter;

import 'package:loredart_nn/loredart_nn.dart';

void main() {
  List<List<List<double>>> data = createTrainData(CsvToListConverter()
          .convert(
            File('.csv' //name of the .csv file with data
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
        Normalization(),
        Dense(64, activation: Activation.relu()),
        Dense(10, activation: Activation.linear())
      ],
      loss: Loss.mse(),
      optimizer: SGD(learningRate: 0.05, momentum: 0.8),
      useAccuracyMetric: true);

  mnistDNN.fit(xTrain, yTrain, epochs: 1, verbose: true);
  // epoch 1/1 |100.00%| -> mean secs per train step: 0.00863s, mean loss [mse]: 0.016393, accuracy: 91.57%

  var hist = mnistDNN.evaluate(xTest, yTest, verbose: true);
  // evaluating 12000/12000 |100.00%| -> mean secs per test step: 0.00050s, mean loss [mse]: 0.011336, accuracy: 94.72%

  print(hist); // {mean mse: 0.011336099126190758, accuracy: 0.9471666666666667}

  /// [NeuralNetwork] with 1 hidden layer and crossEntropy
  NeuralNetwork ceMnistDNN = NeuralNetwork(
      784,
      [
        Dense(64, activation: Activation.softplus()), //or leakyReLU or elu
        Normalization(),
        Dense(10, activation: Activation.softmax())
      ],
      loss: Loss
          .crossEntropy(), // sparseCrossEntropy can be used for sparse data with the same result
      optimizer: SGD(learningRate: 0.01, momentum: 0.9),
      useAccuracyMetric: true);

  ceMnistDNN.fit(xTrain, yTrain, epochs: 1, verbose: true);
  // epoch 1/1 |100.00%| -> mean secs per train step: 0.00347s, mean loss [cross_entropy]: 0.332344, accuracy: 90.12%

  var hist2 = ceMnistDNN.evaluate(xTest, yTest, verbose: true);
  // evaluating 12000/12000 |100.00%| -> mean secs per test step: 0.00025s, mean loss [cross_entropy]: 0.237819, accuracy: 92.78%

  print(
      hist2); // {mean cross_entropy: 0.23781868465052503, accuracy: 0.9278333333333333}
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
