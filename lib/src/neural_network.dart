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
///     Normalization(), // preprocess normalization of data
///     Dense(128, activation: Activation.elu()), // 1st hidden layer
///     Dense(32, activation: Activation.leakyReLU()), // 2nd hidden layer
///     // ...
///     Normalization(),
///     Dense(10, activation: Activation.softmax()) // output layer for ten classes
///   ],
///   loss: Loss.mse(), /// for sparse can be [Loss.sparseCrossEntropy()]
///   optimizer: SGD(learningRate: 0.05, momentum: 0.99),
///   useAccuracyMetric: true) /// set [true] to compute classififcation accuracy
/// 
/// NeuralNetwork nnRegression = NeuralNetwork(10, // input length of customer features  
///   [ 
///     Dense(32, activation: Activation.elu()), // hidden layer
///     Normalization()
///     Dense(1, activation: Activation.softmax()) // output layer
///   ],
///   loss: Loss.mae(),
///   optimizer: SGD(learningRate: 0.3, momentum: 0),
///   useAccuracyMetric: false) /// use [false] for regression task
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

  /// - `inputLenght` - Lenght of the input data List
  /// - `layers` - List of [Layer] with lenght >= 1
  /// - `loss` - Loss function to minimize
  /// - `optimizer` - optional parameter, Optimizer for the NeuralNetwork
  /// - `name` - optional parameter, name of the NeuralNetwork
  /// - `useAccuracyMetric` - optional parameter, add accuracy metric computation while fitting (learning) or evaluating (testing).
  /// Can be used only for classsification tasks, with both One-Hot encoded and labeled target class representation.
  NeuralNetwork(int inputLength, List<Layer> layers, {required this.loss, Optimizer? optimizer, String? name, bool useAccuracyMetric = false})
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
      layers[i].init(layers[i-1].units);
    }
  }

  /// Check if predicted [yP] class is equal th the real [y] class
  bool _trueProbs(Matrix y, Matrix yP) {
    if (_isSparse) {
      return y[0][0] == _toSparseFromProbs(yP.flattenList());
    }
    return _toSparseFromProbs(y.flattenList()) == _toSparseFromProbs(yP.flattenList());
  }

  /// Return index of the max element of [probs]
  /// 
  /// Used for accuracy matric for classification models 
  int _toSparseFromProbs(List<double> probs) {
    final max = probs.reduce(math.max);
    return probs.indexOf(max);
  }

  /// Train step function
  /// - feed data to the NeuralNetwork (forward movement)
  /// - compute and update loss
  /// - call gradients calculation
  /// - call optimizers logic
  void _trainStep(List<double> x, List<double> y) {
    Matrix yTrue = Matrix.column(y);
    Matrix yPredicted = layers[0].act(x);
    for (int i = 1; i < layers.length; i += 1) {
      yPredicted = layers[i].act(yPredicted, train: true);
    }
    _meanLoss += loss.function(yTrue, yPredicted);
    if (_useAccuracy) {
      _accuracy += _trueProbs(yTrue, yPredicted) ? 1 : 0;
    }
    var gradients = NablaOperator.gradients(layers.where((layer) => layer.trainable).toList(), loss.dfunction(yTrue, yPredicted));
    optimizer.applyGradients(gradients, layers.where((layer) => layer.trainable).toList());
  }

  /// Trainig function of the NeuralNetwork
  void _fit(List<List<double>> x, List<List<double>> y, {int epochs = 1, bool verbose = true}) {
    assert(x.length == y.length);
    final percent = x.length;
    Duration meanTrainStepTime;
    DateTime tic;
    DateTime toc;
    for (int i = 0; i < epochs; i +=1) {
      _meanLoss = 0;
      _accuracy = 0;
      meanTrainStepTime = Duration();
      for (int j = 0; j < percent; j += 1) {
        tic = DateTime.now();
        _trainStep(x[j], y[j]);
        toc = DateTime.now();
        meanTrainStepTime += toc.difference(tic);
        if (verbose && stdout.hasTerminal) {
          stdout.write(
            'epoch ${i+1}/$epochs |' + ((j+1) / percent * 100).toStringAsFixed(2) + '%| -> '
            'mean secs per train step: ' + (meanTrainStepTime.inSeconds / (j+1)).toStringAsFixed(5) + 's, '
            'mean loss [$loss]: '+ (_meanLoss / (j+1)).toStringAsFixed(6) +
            (_useAccuracy ? ', accuracy: ' + (_accuracy / (j+1) * 100).toStringAsFixed(2) + '%' : '' ) +
            '\r'
          );
        }
      }
      if (verbose) {
        if (stdout.hasTerminal) {
          stdout.write(
            'epoch ${i+1}/$epochs |100%| -> '
            'mean secs per train step: ' + (meanTrainStepTime.inSeconds / (percent)).toStringAsFixed(5) + 's, '
            'mean loss [$loss]: '+ (_meanLoss / (percent)).toStringAsFixed(6) +
            (_useAccuracy ? ', accuracy: ' + (_accuracy / (percent) * 100).toStringAsFixed(2) + '%' : '' ) +
            '\r'
          );
        }
        else {
          stdout.writeln();
        }
      }
    }
    stdout.writeln();
    for (Layer layer in layers) {
      layer.clear();
    }
  }

  /// Call trainig (or fitting) process of [this] NeuralNetwork over given [x] and [y]
  /// 
  /// `x` - Input data (features)
  /// 
  /// `y` - Target value(s)
  /// 
  /// `epochs` - The number of iterations (repetitions) of runing backpropagation over given [x] and [y]
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
  /// nn.fit(x, y, epochs: 100, verbose: true);
  /// // print 100 messages like that:
  /// // epoch 14/100 |100.00%| -> mean secs per train step: 0.00000s, mean loss [mse]: 0.010607
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
  void fit(List<List<double>> x, List<List<double>> y, {int epochs = 1, bool verbose = false}) {
    _fit(x, y, epochs: epochs, verbose: verbose);
  }

  /// Eval step function:
  /// - feed data to the NeuralNetwork
  /// - compute and update loss
  void _evalStep(List<double> x, List<double> y) {
    Matrix yTrue = Matrix.column(y);
    Matrix yPredicted = layers[0].act(x);
    for (int i = 1; i < layers.length; i += 1) {
      yPredicted = layers[i].act(yPredicted, train: true);
    }
    _meanLoss += loss.function(yTrue, yPredicted);
    if (_useAccuracy) {
      _accuracy += _trueProbs(yTrue, yPredicted) ? 1 : 0;
    }
  }

  /// Evaluation (testing) function of the NeuralNetwork
  Map<String, double> _evaluate(List<List<double>> x, List<List<double>> y, {bool verbose = true}) {
    _meanLoss = 0;
    _accuracy = 0;
    final percent = x.length;
    Duration meanTrainStepTime = Duration();
    DateTime tic;
    DateTime toc;
    for (int j = 0; j < percent; j += 1) {
      tic = DateTime.now();
      _evalStep(x[j], y[j]);
      toc = DateTime.now();
      meanTrainStepTime += toc.difference(tic);
      if (verbose) {
        stdout.write(
          'evaluating ${j+1}/$percent |' + ((j+1) / percent * 100).toStringAsFixed(2) + '%| -> '
          'mean secs per test step: ' + (meanTrainStepTime.inSeconds / (j+1)).toStringAsFixed(5) + 's, '
          'mean loss [$loss]: '+ (_meanLoss / (j+1)).toStringAsFixed(6) +
          (_useAccuracy ? ', accuracy: ' + (_accuracy / (j+1) * 100).toStringAsFixed(2) + '%' : '' ) +
          '\r'
        );
      }
    }
    if (verbose) {
      stdout.writeln();
    }

    return _useAccuracy ? {'mean $loss' : _meanLoss / percent, 'accuracy' : _accuracy / percent} : {'mean $loss' : _meanLoss / percent};
  }

  /// Call evaluating or testing process of [this] NeuralNetwork
  ///
  /// `x` - Input data (features)
  /// 
  /// `y` - Target value(s)
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
  /// /// fiting process
  /// nn.fit(x, y, epochs: 100, verbose: true);
  /// 
  /// /// `testing process`
  /// final xTest = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
  /// final yTest = [[1.0], [2.0]];
  /// 
  /// final param = nn.evaluate(xTest, yTest, verbose: true);
  /// // print message like:
  /// // evaluating 2/2 |100.00%| -> mean secs per test step: 0.00000s, mean loss [mse]: 0.003051
  /// print(param); // output: {mean mse: 0.0030511005740773813}
  /// 
  /// // use network for prediction
  /// final predicted = nn.predict([[1.0, 0.0, 1.0]]);
  /// print(predicted);
  /// ```
  Map<String, double> evaluate(List<List<double>> x, List<List<double>> y, {bool verbose = true}) {
    return _evaluate(x, y, verbose: verbose);
  }

  /// Prediction step Function
  Matrix _predictStep(List<double> data, [int from = 1, int to = -1]) {
    Matrix result = layers[0].act(data);
    if (to == -1) {
      for (Layer layer in layers.sublist(from)) {
        result = layer.act(result);
      }
    }
    else {
      for (Layer layer in layers.sublist(from, to+1)) {
        result = layer.act(result);
      }
    }
    return result;
  }

  /// Prediction function for using of [this] NeuralNetwork
  List<Matrix> _predict(List<List<double>> data) {
    return List<Matrix>.generate(data.length, (index) => _predictStep(data[index]));
  }

  /// Call prediction process for [this] NeuralNetwork
  /// 
  /// `inputs` - data for the [neuralNetwork]
  /// 
  /// Return [List<Matrix>] with prediction for each input [List] from [inputs] 
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
  /// /// fiting process
  /// nn.fit(x, y, epochs: 100, verbose: true);
  /// 
  /// /// testing process
  /// final xTest = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
  /// final yTest = [[1.0], [2.0]];
  /// final param = nn.evaluate(xTest, yTest, verbose: true);
  /// 
  /// /// `use network for prediction`
  /// final predicted = nn.predict([[1.0, 0.0, 1.0]]);
  /// print(predicted); /// `matrix 1⨯1 [[1.9687742233004546]] - pretty close to 2`
  /// ```
  List<Matrix> predict(List<List<double>> inputs) {
    return _predict(inputs);
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
  ///     Normalization(),
  ///     Dense(3, activation: Activation.softmax())
  ///   ], loss: Loos.mse());
  /// 
  /// nn.saveWeights('class_model');
  /// 
  /// // second model with same architecture
  /// final nn2 = NeuralNetwork(10,
  ///   [
  ///     Dense(64, activation: Activation.elu()),
  ///     Normalization(),
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
    List<double> dimensions = [0x1313, len+1, layers[0].units.toDouble()];
    List<double> weights = [];
    for (Layer layer in layers.where((l) => l.trainable)) {
      dimensions.add(layer.units.toDouble());
      weights.addAll(layer.w!.flattenList());
      weights.addAll(layer.b!.flattenList());
    }
    binSerializationFile.writeAsBytesSync(Float64List.fromList(dimensions + weights).buffer.asUint8List());
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
  ///     Normalization(),
  ///     Dense(3, activation: Activation.softmax())
  ///   ], loss: Loos.mse());
  /// 
  /// nn.saveWeights('class_model');
  /// 
  /// // second model with same architecture
  /// final nn2 = NeuralNetwork(10,
  ///   [
  ///     Dense(64, activation: Activation.elu()),
  ///     Normalization(),
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
    }
    else {
      throw Exception("Directory $path don't contains model_weights.bin file");
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
        layers[train].w = Matrix.reshapeFromList(weightsData.sublist(
          offset, offset + inDim*outDim
        ), n: outDim, m: inDim);
        offset += inDim*outDim;
        layers[train].b = Matrix.column(weightsData.sublist(offset, offset + outDim));
        offset += outDim;
        inDim = outDim;
        outDim = weightsData[2+i+1].toInt();
        train += 1;
      }
    }
    else {
      throw Exception('Wrong .bin file');
    }
  }

  @override
  String toString() {
    var str = '#'*65 + '\n# $name (loss: $loss, optimizer: $optimizer)\n';
    for (var l in layers) {
      str += '#' + '-'*50 + '\n';
      str += '# ' + l.toString() + '\n';
    }
    str += '#'*65;
    return str;
  }
}

// Supported files types for saving
enum SaveType {
  bin
}