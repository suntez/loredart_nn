import 'dart:typed_data';

import 'dart:io';
import 'dart:math' as math;

import 'layer.dart';
import 'loss_function.dart';
import 'matrix.dart';
import 'model.dart';
import 'nabla_operator.dart';
import 'optimizer.dart';

/// Neural Network class
///
/// ### Constructor parameters
/// - `inputLenght` - Lenght of the input data List
/// - `layers` - List of [Layer] with lenght >= 1
/// - `loss` - Loss function to minimize
/// - `optimizer` - optional parameter, Optimizer for the NeuralNetwork
/// - `name` - optional parameter, name of the NeuralNetwork
/// - `useAccuracyMetric` - optional parameter, add accuracy metric computation while fitting (learning) or evaluating (testing).
/// Can be used only for classsification tasks, with both One-Hot encoded and labeled target class representation.
///
/// Example
/// ```dart
/// NeuralNetwork nnClasses = NeuralNetwork(784, // input length of 28 by 28 image
///   [
///     LayerNormalization(), // preprocess normalization of data records
///     Dense(128, activation: Activation.elu()), // 1st hidden layer
///     Dense(32, activation: Activation.leakyReLU()), // 2nd hidden layer
///     // ...
///     LayerNormalization(),
///     Dense(10, activation: Activation.softmax()) // output layer for ten classes
///   ],
///   loss: Loss.mse(), /// for sparse can be [Loss.sparseCrossEntropy()]
///   optimizer: SGD(learningRate: 0.05, momentum: 0.99),
///   useAccuracyMetric: true) /// set [true] to compute classififcation accuracy
///
/// NeuralNetwork nnRegression = NeuralNetwork(10, // input length of customer features
///   [
///     Dense(32, activation: Activation.elu()), // hidden layer
///     LayerNormalization()
///     Dense(1, activation: Activation.softmax()) // output layer
///   ],
///   loss: Loss.mae(),
///   optimizer: SGD(learningRate: 0.3, momentum: 0),
///   useAccuracyMetric: false) /// use [false] for regression tasks
/// ```
class NeuralNetwork extends Model {
  late String name;
  late List<Layer> layers;
  late Optimizer optimizer;
  late Loss loss;
  double _meanLoss = 0;
  late bool _useAccuracy;
  double _accuracy = 0;
  late bool _isSparse;
  Map<String, List<double>>? _history;

  /// - `inputLenght` - Lenght of the input data List
  /// - `layers` - List of [Layer] with lenght >= 1
  /// - `loss` - Loss function to minimize
  /// - `optimizer` - optional parameter, Optimizer for the NeuralNetwork
  /// - `name` - optional parameter, name of the NeuralNetwork
  /// - `useAccuracyMetric` - optional parameter, add accuracy metric computation while fitting (learning) or evaluating (testing).
  /// Can be used only for classsification tasks, with both One-Hot encoded and labeled target class representation.
  NeuralNetwork(int inputLength, List<Layer> layers,
      {required this.loss,
      Optimizer? optimizer,
      String? name,
      bool useAccuracyMetric = false})
      : assert(layers.isNotEmpty) {
    this.name = name ?? 'NeuralNetwork';
    this.optimizer = optimizer ?? SGD();
    this.layers = layers.sublist(0);
    this.layers.insert(0, Input(inputLength));
    _useAccuracy = useAccuracyMetric;
    _isSparse = loss.name == 'sparse_cross_entropy';
    _initNN();
  }

  /// Initialization of layers of [this] NeuralNetwork
  void _initNN() {
    layers[0].init();
    for (int i = 1; i < layers.length; i += 1) {
      layers[i].init(layers[i - 1].units);
    }
  }

  double _batchAccuracy(Matrix y, Matrix yP) {
    int truePositive = 0;
    for (int i = 0; i < y.m; i += 1) {
      truePositive += (_isSparse ? y[i][0] : _whichMax(y.getColumn(i))) == _whichMax(yP.getColumn(i)) ? 1 : 0;
    }
    return truePositive / y.m;
  }

  /// Return index of the max element of [probs]
  ///
  /// Used for accuracy metric for classification models
  int _whichMax(Matrix probs) {
    final max = probs.flattenList().reduce(math.max);
    return probs.flattenList().indexOf(max);
  }

  /// Train step function
  /// - feed data batch to the NeuralNetwork
  /// - compute and update loss
  /// - call gradients calculation
  /// - call optimizer to update weights
  void _trainStep(List<List<double>> x, List<List<double>> y) {
    Matrix yTrueBatch = Matrix.fromLists(y).T;
    Matrix yPredictedBatch = layers[0].act(x);
    for (int i = 1; i < layers.length; i += 1) {
      yPredictedBatch = layers[i].act(yPredictedBatch, train: true);
    }
    _meanLoss += loss.function(yTrueBatch, yPredictedBatch);
    if (_useAccuracy) {
      _accuracy += _batchAccuracy(yTrueBatch, yPredictedBatch);
    }
    var gradients = NablaOperator.gradients(
        layers.where((layer) => layer.trainable).toList(),
        loss.dfunction(yTrueBatch, yPredictedBatch));
    optimizer.applyGradients(gradients, layers.where((layer) => layer.trainable).toList());
  }

  /// Trainig function of the NeuralNetwork
  Map<String, List<double>>? _fit(List<List<double>> x, List<List<double>> y, {int epochs = 1, int batchSize = 1, bool verbose = true}) {
    assert(x.length == y.length);
    final generalCard = x.length;
    assert(batchSize <= generalCard);
    Duration meanTrainStepTime;
    DateTime tic;
    DateTime toc;
    _history = {'$loss' : List<double>.filled(epochs, -0)};
    if (_useAccuracy) {
      _history?.addAll({'accuracy': List<double>.filled(epochs, -0)});
    }
    final steps = generalCard ~/ batchSize + (generalCard % batchSize == 0 ? 0 : 1);
    for (int epoch = 0; epoch < epochs; epoch += 1) {
      _meanLoss = 0;
      _accuracy = 0;
      meanTrainStepTime = Duration();
      for (int j = 1; j <= steps; j += 1) {
        tic = DateTime.now();
        if (j == steps) {
          _trainStep(x.sublist(batchSize * (j-1)), y.sublist(batchSize * (j-1)));
        }
        else {
          _trainStep(x.sublist(batchSize * (j-1), batchSize * j), y.sublist(batchSize * (j-1), batchSize * j));
        }
        toc = DateTime.now();
        meanTrainStepTime += toc.difference(tic);
        if (verbose && stdout.hasTerminal) {
          stdout.write('epoch ${epoch + 1}/$epochs |$j/$steps| -> mean time per batch: ' +
            (meanTrainStepTime.inMilliseconds / j).toStringAsFixed(2) + 'ms, '
            'mean loss [$loss]: ' +
            (_meanLoss / j).toStringAsFixed(6) +
            (_useAccuracy ?
              ', mean accuracy: ' + (_accuracy / (j) * 100).toStringAsFixed(2) +
              '%' : '') +
            '\r'
          );
        }
      }
      if (verbose) {
        if (!stdout.hasTerminal) {
          stdout.write('epoch ${epoch + 1}/$epochs |$steps/$steps| -> mean time per batch: ' +
            (meanTrainStepTime.inMilliseconds / steps).toStringAsFixed(2) + 'ms, '
            'mean loss [$loss]: ' +
            (_meanLoss / steps).toStringAsFixed(6) +
            (_useAccuracy ?
              ', mean accuracy: ' + (_accuracy / steps * 100).toStringAsFixed(2) +
              '%' : '') +
            '\r'
          );
        }
        else {
          stdout.writeln();
        }
        _history?['$loss']?[epoch] = _meanLoss / steps;
        if (_useAccuracy) {
          _history?['accuracy']?[epoch] = _accuracy / steps * 100;
        }
      }
    }
    stdout.writeln();
    for (Layer layer in layers) {
      layer.clear();
    }
    return _history;
  }

  /// Call trainig (or fitting) process of [this] NeuralNetwork over given [x] and [y]
  ///
  /// `x` - Input data (features)
  ///
  /// `y` - Target value(s)
  ///
  /// `epochs` - |hyperparam| - The number of iterations (repetitions) of runing backpropagation over given [x] and [y]
  /// 
  /// `batchSize` - |hyperparam| - The cardinality of one mini-batch, should be less then [x.length]
  ///
  /// `verbose` - Write logs to the stdout
  ///
  /// If [verbose] if `true` and `stdout` can be overwriten then throught the process logs will be updated
  /// if there is no `stdout` then only mean values for each [epoch] will be printed
  ///
  /// Example:
  /// ```dart
  /// // training data
  /// final x = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
  /// final y = [[1.0], [1.0], [2.0], [3.0]]; // number of '1' in the list
  ///
  /// final nn = NeuralNetwork(3, // length of input vector
  /// [
  ///   Dense(16, activation: Activation.swish()),
  ///   Dense(1, activation: Activation.relu()) // output layer
  /// ], loss: Loss.mse(), optimizer: SGD(learningRate: 0.2, momentum: 0));
  ///
  /// /// `fiting process`
  /// var history = nn.fit(x, y, epochs: 50, batchSize: 1, verbose: true);
  /// // prints 50 messages like that:
  /// // epoch 13/50 |4/4| -> mean time per batch: 0.00ms, mean loss [mse]: 0.029751
  /// 
  /// print(history); // Output: {mse: [3.741039311383735, 3.5163661460239033, ...., 3.70640438723175e-15]} 
  ///
  /// //testing process
  /// final xTest = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
  /// final yTest = [[1.0], [2.0]];
  /// final param = nn.evaluate(xTest, yTest, verbose: true);
  /// print(param);
  ///
  /// // use network for prediction
  /// final predicted = nn.predict([[1.0, 0.0, 1.0]]);
  /// print(predicted);
  /// ```
  Map<String, List<double>>? fit(List<List<double>> x, List<List<double>> y, {int epochs = 1, int batchSize = 1, bool verbose = false}) {
    return _fit(x, y, epochs: epochs, batchSize: batchSize, verbose: verbose);
  }

  /// Eval step function:
  /// - feed data ифеср to the NeuralNetwork
  /// - compute and update loss
  void _evalStep(List<List<double>> x, List<List<double>> y) {
    Matrix yTrueBatch = Matrix.fromLists(y).T;
    Matrix yPredictedBatch = layers[0].act(x);
    for (int i = 1; i < layers.length; i += 1) {
      yPredictedBatch = layers[i].act(yPredictedBatch, train: false);
    }
    _meanLoss += loss.function(yTrueBatch, yPredictedBatch);
    if (_useAccuracy) {
      _accuracy += _batchAccuracy(yTrueBatch, yPredictedBatch);
    }
  }

  /// Evaluation (testing) function of the NeuralNetwork
  Map<String, double> _evaluate(List<List<double>> x, List<List<double>> y, {int batchSize = 1, bool verbose = true}) {
    final generalCard = x.length;
    assert(batchSize <= generalCard);
    Duration meanTrainStepTime = Duration();
    DateTime tic;
    DateTime toc;
    _meanLoss = 0;
    _accuracy = 0;
    final steps = generalCard ~/ batchSize + (generalCard % batchSize == 0 ? 0 : 1);
    for (int j = 1; j <= steps; j += 1) {
      tic = DateTime.now();
      if (j == steps) {
          _evalStep(x.sublist(batchSize * (j-1)), y.sublist(batchSize * (j-1)));
        }
        else {
          _evalStep(x.sublist(batchSize * (j-1), batchSize * j), y.sublist(batchSize * (j-1), batchSize * j));
        }
      toc = DateTime.now();
      meanTrainStepTime += toc.difference(tic);
      if (verbose && stdout.hasTerminal) {
        stdout.write('evaluating batch $j/$steps -> '
          'mean time per batch: ' +
          (meanTrainStepTime.inMilliseconds / j).toStringAsFixed(2) + 'ms, '
          'mean loss [$loss]: ' +
          (_meanLoss / j).toStringAsFixed(6) +
          (_useAccuracy ?
            ', mean accuracy: ' + (_accuracy / j * 100).toStringAsFixed(2) +
            '%' : '') +
          '\r'
        );
      }
    }
    if (verbose) {
      stdout.hasTerminal ? 
      stdout.writeln() :
      stdout.writeln('evaluating batch $steps/$steps -> '
          'mean time per batch: ' +
          (meanTrainStepTime.inMilliseconds / steps).toStringAsFixed(2) + 'ms, '
          'mean loss [$loss]: ' +
          (_meanLoss / steps).toStringAsFixed(6) +
          (_useAccuracy ?
            ', mean accuracy: ' + (_accuracy / steps * 100).toStringAsFixed(2) +
            '%' : '')
        );
    }

    return _useAccuracy
        ? {'mean $loss': _meanLoss / steps, 'mean accuracy': _accuracy / steps}
        : {'mean $loss': _meanLoss / steps};
  }

  /// Call evaluating or testing process of [this] NeuralNetwork on the given batches
  ///
  /// `x` - Input data (features)
  ///
  /// `y` - Target value(s)
  /// 
  /// `batchSize` - The cardinality of one mini-batch, should be less then [x.length]
  ///
  /// `verbose` - Write logs to the stdout
  ///
  /// If [verbose] if `true` and `stdout` can be overwriten then throught the process logs will be updated
  ///
  /// Example:
  /// ```dart
  /// // training data
  /// final x = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
  /// final y = [[1.0], [1.0], [2.0], [3.0]]; // number of '1' in the list
  ///
  /// final nn = NeuralNetwork(3, // length of input vector
  /// [
  ///   Dense(16, activation: Activation.swish()),
  ///   Dense(1, activation: Activation.relu()) // output layer
  /// ], loss: Loss.mse(), optimizer: SGD(learningRate: 0.2, momentum: 0));
  ///
  /// /// `fiting process`
  /// var history = nn.fit(x, y, epochs: 50, batchSize: 1, verbose: true);
  /// print(history);
  ///
  /// //testing process
  /// final xTest = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
  /// final yTest = [[1.0], [2.0]];
  /// final param = nn.evaluate(xTest, yTest, verbose: true);
  /// // evaluating batch 2/2 -> mean time per batch: 0.00ms, mean loss [mse]: 0.003051
  /// print(param); // Output: {mean mse: 0.0030511005740773813}
  ///
  /// // use network for prediction
  /// final predicted = nn.predict([[1.0, 0.0, 1.0]]);
  /// print(predicted);
  /// ```
  Map<String, double> evaluate(List<List<double>> x, List<List<double>> y, {int batchSize = 1, bool verbose = true}) {
    return _evaluate(x, y, batchSize: batchSize, verbose: verbose);
  }

  /// Prediction function for using of [this] NeuralNetwork
  Matrix _predict(List<List<double>> data) {
    Matrix result = layers[0].act(data);
    for (Layer layer in layers.sublist(1)) {
      result = layer.act(result);
    }
    return result.T;
  }

  /// Call prediction process for [this] NeuralNetwork on the given input data
  ///
  /// `inputs` - data for the [NeuralNetwork]
  ///
  /// Return [List<List<double>>] with prediction for each input [List] from [inputs]
  ///
  /// Example:
  /// ```dart
  /// // training data
  /// final x = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
  /// final y = [[1.0], [1.0], [2.0], [3.0]]; // number of '1' in the list
  ///
  /// final nn = NeuralNetwork(3, // length of input vector
  /// [
  ///   Dense(16, activation: Activation.swish()),
  ///   Dense(1, activation: Activation.relu()) // output layer
  /// ], loss: Loss.mse(), optimizer: SGD(learningRate: 0.2, momentum: 0));
  ///
  /// // fiting process
  /// var history = nn.fit(x, y, epochs: 50, batchSize: 1, verbose: true);
  /// print(history); 
  ///
  /// // testing process
  /// final xTest = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
  /// final yTest = [[1.0], [2.0]];
  /// final param = nn.evaluate(xTest, yTest, verbose: true);
  /// print(param);
  ///
  /// // use network for prediction
  /// final predicted = nn.predict([[1.0, 0.0, 1.0]]);
  /// print(predicted); // Output: [[1.9687742233004546]] ~ there are two '1' in the list, which is pretty close to the ground truth
  /// ```
  List<List<double>> predict(List<List<double>> inputs) {
    return _predict(inputs).matrix;
  }

  /// Save weights and biases of trainable layers of [this] NeuralNetwork to the `$path/model_weights.bin` file
  ///
  /// [path] should be a path to the folder
  ///
  /// Example:
  /// ```dart
  /// // first model
  /// final nn = NeuralNetwork(10,
  ///   [
  ///     Dense(64, activation: Activation.elu()),
  ///     LayerNormalization(),
  ///     Dense(3, activation: Activation.softmax())
  ///   ], loss: Loos.mse());
  ///
  /// nn.saveWeights('class_model');
  ///
  /// // second model with same architecture
  /// final nn2 = NeuralNetwork(10,
  ///   [
  ///     Dense(64, activation: Activation.elu()),
  ///     LayerNormalization(),
  ///     Dense(3, activation: Activation.softmax())
  ///   ], loss: Loos.mse());
  ///
  /// nn2.loadWeights('class_model');
  ///
  /// ```
  ///
  /// `P.S.`
  /// Method saves only weights and biases of trainable layers (Dense), which means, you can add or remove Normaization layers,
  /// or change activation functions of any layer, and still be able to restore the weights via [loadWeights] method.
  /// For now, it's developers responsibiblity to use correct model architecture and elements.
  void saveWeights(String path, [SaveType type = SaveType.bin]) {
    if (type == SaveType.bin) {
      return _saveTrainableWeightsFromLayersToBin(path);
    }
  }

  /// Perform saving of the weights and biases of the trainable layers of [this] NeuralNetwork
  void _saveTrainableWeightsFromLayersToBin(String path) {
    File binSerializationFile = File(path + '/model_weights.bin');
    binSerializationFile.createSync(recursive: true);
    double len = layers.where((element) => element.trainable).length.toDouble();
    List<double> dimensions = [0x1313, len + 1, layers[0].units.toDouble()];
    List<double> weights = [];
    for (Layer layer in layers.where((l) => l.trainable)) {
      dimensions.add(layer.units.toDouble());
      weights.addAll(layer.w!.flattenList());
      weights.addAll(layer.b!.flattenList());
    }
    binSerializationFile.writeAsBytesSync(
        Float64List.fromList(dimensions + weights).buffer.asUint8List());
  }

  /// Load weights and biases of trainable layers of [this] NeuralNetwork from the `$path/model_weights.bin` file
  ///
  /// [path] should be a path to the folder, and `not` be full path to the .bin file
  ///
  /// Example:
  /// ```dart
  /// // first model
  /// final nn = NeuralNetwork(10,
  ///   [
  ///     Dense(64, activation: Activation.elu()),
  ///     LayerNormalization(),
  ///     Dense(3, activation: Activation.softmax())
  ///   ], loss: Loos.mse());
  ///
  /// nn.saveWeights('class_model');
  ///
  /// // second model with same architecture
  /// final nn2 = NeuralNetwork(10,
  ///   [
  ///     Dense(64, activation: Activation.elu()),
  ///     LayerNormalization(),
  ///     Dense(3, activation: Activation.softmax())
  ///   ], loss: Loos.mse());
  ///
  /// nn2.loadWeights('class_model');
  ///
  /// ```
  void loadWeights(String path, [SaveType type = SaveType.bin]) {
    if (type == SaveType.bin) {
      _loadWeightsFromBin(path);
    }
  }

  /// Load weights and biases of trainable layers of [this] NeuralNetwork from `buffer`
  ///
  /// This method specificly created to load [NeuralNetwork] from `Flutter` assets
  ///
  /// Example:
  /// ```dart
  /// // var model = NeuralNetwork(...)
  /// rootBundle.load('assets/models/ce_mnist/model_weights.bin').then((value) {
  ///    model.loadWeightsFromBytes(value.buffer);
  ///  });
  /// ```
  void loadWeightsFromBytes(ByteBuffer buffer) {
    _loadWeightsFromBytes(buffer);
  }

  /// Perform loading of the weights and biases to the trainable layers of [this] NeuralNetwork
  void _loadWeightsFromBin(String path) {
    File binSerializationFile = File(path + '/model_weights.bin');
    if (!binSerializationFile.existsSync()) {
      _loadWeightsFromBytes(binSerializationFile.readAsBytesSync().buffer);
    } else {
      throw Exception("Directory $path doesn't contain model_weights.bin file");
    }
  }

  /// Load weights and biases of trainable layers of [this] NeuralNetwork from `buffer`
  void _loadWeightsFromBytes(ByteBuffer buffer) {
    Float64List weightsData = buffer.asFloat64List();
    if (weightsData[0] == 0x1313 && weightsData[1] >= 3) {
      var len = weightsData[1].toInt();
      int inDim = weightsData[2].toInt();
      int outDim = weightsData[3].toInt();
      int offset = 2 + len;
      int train = 1;
      for (int i = 1; i < len; i += 1) {
        while (!layers[train].trainable) {
          train += 1;
        }
        layers[train].w = Matrix.reshapeFromList(
            weightsData.sublist(offset, offset + inDim * outDim),
            n: outDim,
            m: inDim);
        offset += inDim * outDim;
        layers[train].b =
            Matrix.column(weightsData.sublist(offset, offset + outDim));
        offset += outDim;
        inDim = outDim;
        outDim = weightsData[2 + i + 1].toInt();
        train += 1;
      }
    } else {
      throw Exception('Wrong .bin file');
    }
  }

  @override
  String toString() {
    var str = '#' * 65 + '\n# $name (loss: $loss, optimizer: $optimizer)\n';
    for (var l in layers) {
      str += '#' + '-' * 50 + '\n';
      str += '# ' + l.toString() + '\n';
    }
    str += '#' * 65;
    return str;
  }
}

// Supported files types for saving
enum SaveType { bin }
