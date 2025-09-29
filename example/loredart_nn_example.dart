import 'package:loredart_nn/loredart_nn.dart';
import 'package:csv/csv.dart' show CsvToListConverter;

import 'dart:io' show File;
import 'dart:convert';

void main() {
  // Read training data from csv file (e.g. taken from https://github.com/phoebetronic/mnist)
  List<List<dynamic>> rawData = CsvToListConverter()
      .convert(
        File(
          'some_mnist_data.csv',
        ).readAsStringSync(),
        shouldParseNumbers: true,
      )
      .sublist(1); // skipping columns names

  // Convert csv content to Tensor and plit it into features and targets
  var (x, y) = splitToFeaturesAndTargets(
    Tensor.constant(rawData),
    targetIndices: [0], // first column is a MNIST class-digit
    featuresDType: DType.float32,
  );
  y = squeeze(y);

  // Train-test split
  int testSize = (x.shape[0] * 0.2).ceil();

  var xTest = slice(x, [0, 0], [testSize, x.shape[1]]);
  var xTrain = slice(x, [testSize, 0], [x.shape[0], x.shape[1]]);

  var yTest = slice(y, [0], [testSize]);
  var yTrain = slice(y, [testSize], [y.shape[0]]);

  // Configure classifier model
  final model = Model(
    [Rescale(scale: 1 / 255), Dense(32, activation: Activations.relu), Dense(32, activation: ReLU()), Dense(10)],
    optimizer: Adam(weightDecay: 1e-4),
    loss: SparseCategoricalCrossentropy(fromLogits: true),
    metrics: [Metrics.sparseCategoricalAccuracy],
    inputShape: [x.shape[-1]], // if we know the input shape - model will be built immediately
  );

  print(model); // if model wan't build no training params statistics will be generated
  // __________________________________
  // Layer       Output shape   Param #
  // ==================================
  // Rescale-1   [784]          0
  // Dense-1     [32]           25120
  // Dense-2     [32]           1056
  // Dense-3     [10]           330
  // ==================================
  // Total trainable params: 26506
  // __________________________________

  // Train model
  final history = model.fit(
    x: xTrain,
    y: yTrain,
    epochs: 2,
    batchSize: 64,
    validationData: [xTest, yTest], // for simplicity using test data as val
  );
  // With `verbose: true` will see training progress:
  //  Straining model training
  //  Epoch 1/2:
  //  125/125 - [=====] - 6 s - 54 ms per step - loss: 0.6042 - sparse_categorical_accuracy: 0.7402 - val_loss: 0.6465 - val_sparse_categorical_accuracy: 0.7935
  //  Epoch 2/2:
  //  125/125 - [=====] - 6 s - 51 ms per step - loss: 0.4763 - sparse_categorical_accuracy: 0.8905 - val_loss: 0.4655 - val_sparse_categorical_accuracy: 0.8647

  print(
    'History:\n$history',
  );
  // {loss: [0.5919618010520935, 0.5008291006088257], sparse_categorical_accuracy: [0.68, 0.888625], val_loss: [0.6536273518577218, 0.467515311203897], val_sparse_categorical_accuracy: [0.8154296875, 0.8603515625]}

  // Evaluate model
  final evalResults = model.evaluate(x: xTest, y: yTest, batchSize: 128);
  //  16/16 - [=====] - 0 s - 32 ms per step - loss: 0.4589 - sparse_categorical_accuracy: 0.8615

  print('Eval results:\n$evalResults'); // {loss: 0.471101189032197, sparse_categorical_accuracy: 0.8615234382450581}

  // Save model
  File f =
      File('mnist_classifer.json')
        ..createSync()
        ..writeAsStringSync(jsonEncode(model.toJson()));

  // Load saved model to use later for predictions
  final loadedModel = Model.fromJson(jsonDecode(f.readAsStringSync()));
  final preds = loadedModel.predict(slice(xTest, [0, 0], [4, xTest.shape[-1]]));

  print(argMax(preds, axis: -1)); // smth like [7, 2, 1, 0]
}
