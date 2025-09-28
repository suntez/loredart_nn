import 'dart:io' show stdout;
import 'dart:math' as math;
import 'package:dolumns/dolumns.dart';
import 'package:loredart_nn/src/utils/serialization_utils.dart';
import 'package:loredart_tensor/loredart_tensor.dart';

import 'layers.dart';
import '/src/optimizers/optimizers.dart';
import '/src/metrics_and_losses/losses.dart';
import '/src/metrics_and_losses/metrics.dart';

class _DurationCounter {
  Duration total = Duration.zero;
  int count = 0;
  Duration last = Duration.zero;
  double get perStep => count == 0 ? 0 : total.inMilliseconds / count;

  void updateState(Duration duration) {
    count += 1;
    total += duration;
    last = duration;
  }

  void resetState() {
    count = 0;
    total = Duration.zero;
  }
}

/// History of loss and metrics during training epochs
typedef History = Map<String, List<num>>;

/// A Model is a complete neural network made up of an ordered list of [Layer]s.
///
/// Encapsulates the complete training and inference logic, including "compiling"
/// with an [optimizer], [loss] function, and [metrics].
///
/// The model itself inherits from [Layer], allowing it to be part of other Models.
class Model extends Layer {
  static int _modelCounter = 0;

  /// The ordered sequence of layers that make up the neural network.
  late List<Layer> layers;

  /// The [Optimizer] used for updating the model's trainable parameters during training.
  late Optimizer optimizer;

  /// The [Loss] function to be minimized during training.
  late Loss loss;

  /// The list of [Metric]s to be evaluated during training and testing.
  late List<Metric> metrics;

  bool _built = false;

  /// Keeps the number of trainable parameters for each layer after [build].
  late List<int> _numberOfParamsInLayers;

  final _DurationCounter _trainDurationCounter = _DurationCounter();
  final _DurationCounter _testDurationCounter = _DurationCounter();

  /// Gets a flatten list of all [trainableParams] from all [layers] in order.
  @override
  List<Tensor> get trainableParams => [for (var layer in layers) ...layer.trainableParams];

  /// Creates a Model from an ordered list of [layers].
  ///
  /// Requires a [loss] function and an [optimizer] to be configured.
  /// Optional [metrics] can be provided for tracking performance during training.
  Model(
    this.layers, {
    required this.loss,
    required this.optimizer,
    this.metrics = const [],
    String? name,
    bool trainable = true,
    List<int>? inputShape,
  }) : super(trainable) {
    if (name == null) {
      _modelCounter += 1;
    }
    this.name = name ?? 'Model-$_modelCounter';
    if (inputShape != null) {
      build(inputShape);
    }
  }

  /// Builds the model by calling [Layer.build] sequentially on all its [layers].
  /// Usually called internally and not by user.
  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, -1);

      _numberOfParamsInLayers = [];
      List<int> outputShape = List.from(inputShape);

      for (var layer in layers) {
        layer.build(outputShape);
        outputShape = layer.outputShape;
        _numberOfParamsInLayers.add(layer.trainableParams.length);
      }

      _built = true;
      this.inputShape = inputShape;
      this.outputShape = outputShape;
    }
  }

  /// Updates the trainable parameters of all layers with the provided [updatedParams].
  ///
  /// The flattened list of [updatedParams] is sliced and distributed
  /// back to the corresponding layers based on the parameter counts stored in
  /// [_numberOfParamsInLayers].
  @override
  void updateTrainableParams(List<Tensor> updatedParams) {
    if (trainable) {
      int cumulativeSum = 0;
      for (int i = 0; i < _numberOfParamsInLayers.length; i += 1) {
        layers[i].updateTrainableParams(
          updatedParams.sublist(cumulativeSum, cumulativeSum + _numberOfParamsInLayers[i]),
        );
        cumulativeSum += _numberOfParamsInLayers[i];
      }
    }
  }

  /// Runs the batched [input] Tensor through all [layers] sequentially.
  ///
  /// If [training] is true, it triggers each layer's `call` method with `training: true`,
  /// which in turn constructs the layer's [Layer.gradient] function.
  /// Otherwise [layers] are called in the inference mode.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    for (var layer in layers) {
      input = layer(input, training: training);
    }
    if (training) {
      _constructGradient();
    }

    return input;
  }

  /// Constructs the model's [gradient] function by chaining the gradients of its layers.
  void _constructGradient() {
    gradient = (upstream) {
      List<Tensor> gradients = [];
      for (var layer in layers.reversed) {
        List<Tensor> grads = layer.gradient!(upstream);
        upstream = grads[0];
        gradients.addAll(grads.sublist(1).reversed);
      }
      return [upstream, ...gradients.reversed];
    };
  }

  /// Clears the [Layer.gradient] function on all layers.
  void _disposeGradients() {
    for (Layer layer in layers) {
      layer.gradient = null;
    }
  }

  /// Utility to get all metrics results and loss value as single map
  Map<String, dynamic> _getMetricsWithLoss(num lossValue, [String prefix = ""]) {
    return Map.fromIterables(['loss', ...metrics.map((m) => m.name)].map((e) => prefix + e), [
      lossValue,
      ...metrics.map((m) => m.result),
    ]);
  }

  String _buildLogMessage(
    Map<String, dynamic> logValues, {
    required bool train,
    required bool last,
    int digits = 4,
    bool includeTime = true,
  }) {
    List<String> values = [];
    _DurationCounter _durationCounter = train ? _trainDurationCounter : _testDurationCounter;
    if (includeTime) {
      if (last) {
        values.add("${_durationCounter.total.inSeconds} s");
      }
      values.add(
        last ? "${_durationCounter.perStep.round()} ms per step" : "${_durationCounter.last.inMilliseconds} ms",
      );
    }
    values.addAll([for (var key in logValues.keys) "$key: ${logValues[key]?.toStringAsFixed(digits)}"]);

    return values.join(' - ');
  }

  /// Logs single step for the [batch].
  void _logStep(int batch, int batches, String log) {
    if (stdout.hasTerminal) {
      int percent = (batch / batches) * 100 ~/ 20;
      String process =
          '[' + (percent == 0 ? '>....' : (percent == 5 ? '=====' : '=' * percent + '>' + '.' * (4 - percent))) + ']';
      stdout.write(
        '$batch/$batches - $process - $log'
        '   '
        '\r',
      );
    } else if (batch == batches) {
      stdout.write('$batch/$batches - [=====] - $log');
    }
  }

  /// Logic of the single evaluation step
  double _testStep(Tensor x, Tensor y) {
    final tik = DateTime.now();

    final yPred = call(x, training: true);
    // mean will ensure that loss is a scalar even if loss reduction is a [sum] or [none]
    var lossValue = (mean(loss(y, yPred, training: true)) as NumericTensor).buffer[0];
    for (var metric in metrics) {
      metric.updateState(y, yPred);
    }
    final tok = DateTime.now();
    _testDurationCounter.updateState(tok.difference(tik));

    return lossValue;
  }

  /// Evaluates model's performance on a given dataset ([x], [y]).
  ///
  /// Iterates through the data in batches, calculates the [loss] and updates
  /// all registered [metrics]. If called with verbosity - logs all results.
  ///
  /// The batch size is automatically determined (as 1/4 of of the inputs if total number if bigger then 16) if not provided.
  ///
  /// Returns a map of the final aggregated metric values and the average loss.
  Map<String, dynamic> evaluate({
    required Tensor x,
    required Tensor y,
    int? batchSize,
    bool verbose = true,
    String prefix = '',
  }) {
    batchSize ??= x.shape[0] > 16 ? x.shape[0] ~/ 4 : x.shape[0];
    int numberOfSteps = (x.shape[0] / batchSize).ceil();

    for (var metric in metrics) {
      metric.resetState();
    }
    _testDurationCounter.resetState();

    Map<String, dynamic> metricsValues = {};
    List<double> losses = [];

    for (int step = 1; step <= numberOfSteps; step += 1) {
      int fromIndex = (step - 1) * batchSize;
      int toIndex = math.min(step * batchSize, x.shape[0]);
      final lossValue = _testStep(
        slice(x, [fromIndex, ...List.filled(x.rank - 1, 0)], [toIndex, ...x.shape.list.sublist(1)]),
        slice(y, [fromIndex, ...List.filled(y.rank - 1, 0)], [toIndex, ...y.shape.list.sublist(1)]),
      );
      losses.add(lossValue);
      metricsValues = _getMetricsWithLoss(lossValue, prefix = prefix);
      if (verbose) {
        _logStep(step, numberOfSteps, _buildLogMessage(metricsValues, last: step == numberOfSteps, train: false));
      }
    }
    if (verbose) {
      stdout.writeln();
    }
    metricsValues[prefix + 'loss'] = losses.reduce((l1, l2) => l1 + l2) / losses.length;
    return metricsValues;
  }

  /// Performs a single optimization step (forward pass, loss calculation, backpropagation, and parameter update) on a batch of data.
  ///
  /// The gradient from the [loss] function is passed to the model's [gradient]
  /// function, and the resulting weight gradients are applied via the [optimizer].
  ///
  /// Returns loss value for the input [batch].
  double _trainStep(Tensor x, Tensor y) {
    final tik = DateTime.now();

    final yPred = call(x, training: true);
    double lossValue = (loss(y, yPred, training: true) as NumericTensor).buffer[0];

    final upstream = Tensor.ones(yPred.shape.list);
    List<Tensor> gradients = gradient!(loss.gradient!(upstream)[0]);

    updateTrainableParams(optimizer.applyGradients(trainableParams, gradients.sublist(1)));

    for (var metric in metrics) {
      metric.updateState(y, yPred);
    }

    final tok = DateTime.now();
    _trainDurationCounter.updateState(tok.difference(tik));

    return lossValue;
  }

  /// Trains the model for a fixed number of [epochs] on a dataset ([x], [y]).
  ///
  /// Iterates through the training data in batches. If called with verbosity - logs loss and metrics for each step/epoch.
  ///
  /// Optionally, validation data can be provided via [validationData] to perform evaluation at the end of each epoch.
  ///
  /// Returns a history of loss and metrics per epoch.
  History fit({
    required Tensor x,
    required Tensor y,
    required int epochs,
    int? batchSize,
    List<Tensor>? validationData,
    int? validationBatchSize,
    bool verbose = true,
  }) {
    if (epochs < 1) {
      throw ArgumentError();
    }
    bool hasValData = false;
    if (validationData != null && validationData.length != 2) {
      throw ArgumentError(
        "Expected validation data to be of length 2 (x and y), but received length: ${validationData.length}",
      );
    } else if (validationData != null && validationData.length == 2) {
      hasValData = true;
    }

    batchSize ??= x.shape[0];
    int numberOfSteps = (x.shape[0] / batchSize).ceil();

    final String valPrefix = "val_";
    History history = {};
    if (verbose) {
      stdout.writeln('Straining model training');
    }

    for (int epoch = 1; epoch <= epochs; epoch += 1) {
      for (var metric in metrics) {
        metric.resetState();
      }
      _trainDurationCounter.resetState();

      if (verbose) {
        stdout.writeln('Epoch $epoch/$epochs:');
      }

      for (int step = 1; step <= numberOfSteps; step += 1) {
        // training loop
        int fromIndex = (step - 1) * batchSize;
        int toIndex = math.min(step * batchSize, x.shape[0]);
        final lossValue = _trainStep(
          slice(x, [fromIndex, ...List.filled(x.rank - 1, 0)], [toIndex, ...x.shape.list.sublist(1)]),
          slice(y, [fromIndex, ...List.filled(y.rank - 1, 0)], [toIndex, ...y.shape.list.sublist(1)]),
        );

        Map<String, dynamic> metricsValues = _getMetricsWithLoss(lossValue);
        if (step == numberOfSteps && hasValData) {
          Map<String, dynamic> valMetrics = evaluate(
            x: validationData![0],
            y: validationData[1],
            verbose: false,
            batchSize: validationBatchSize ?? batchSize,
            prefix: valPrefix,
          );
          metricsValues.addAll(valMetrics);
        }

        if (verbose) {
          _logStep(step, numberOfSteps, _buildLogMessage(metricsValues, last: step == numberOfSteps, train: true));
        }

        // updating history
        if (step == numberOfSteps) {
          for (var key in metricsValues.keys) {
            if (!history.containsKey(key)) history[key] = [];
            history[key]!.add(metricsValues[key]);
          }
        }
      }

      if (verbose) {
        stdout.writeln();
      }
    }
    _disposeGradients();

    return history;
  }

  /// Runs the forward pass on a single batch of data for prediction/inference.
  Tensor _predictOnBatch(Tensor x) {
    return call(x, training: false);
  }

  /// Generates output predictions for the input samples.
  ///
  /// Prediction can be done in batches. Automatically processes each batch in inference model, concatenated back into a single output tensor.
  ///
  /// Returns output predictions from model.
  Tensor predict(Tensor input, {int? batchSize}) {
    batchSize ??= math.max(input.shape[0] ~/ 4, 1);
    List<Tensor> batchedOutput = [];

    for (int step = 1; step <= (input.shape[0] / batchSize).ceil(); step += 1) {
      int fromIndex = (step - 1) * batchSize;
      int toIndex = math.min(step * batchSize, input.shape[0]);
      batchedOutput.add(
        _predictOnBatch(
          slice(input, [fromIndex, ...List.filled(input.rank - 1, 0)], [toIndex, ...input.shape.list.sublist(1)]),
        ),
      );
    }
    return concat(batchedOutput, axis: 0);
  }

  /// Constructs a text summary of the model structure, including layer names, output shapes,
  /// and the number of trainable parameters per layer and total.
  String summary() {
    if (_built) {
      List<List<Object>> layersParams = [
        for (var layer in layers)
          [
            layer.name,
            layer.outputShape,
            layer.trainableParams.isNotEmpty
                ? layer.trainableParams.map((w) => w.shape.size).reduce((v, e) => e + v)
                : 0,
          ],
      ];

      int totalParams = layersParams.map((e) => e[2] as int).reduce((v, e) => e + v);
      String table = dolumnify(
        <List<Object>>[
              ["Layer", "Output shape", "Param #"],
            ] +
            layersParams,
        columnSplitter: '   ',
        headerIncluded: true,
        headerSeparator: '=',
      );
      int frameLen = table.split('\n')[0].length;
      return "Model: $name\n${'_' * frameLen}\n$table\n${'=' * frameLen}\nTotal trainable params: $totalParams\n${'_' * frameLen}";
    } else {
      return "Model: $name";
    }
  }

  @override
  String toString() => summary();

  @override
  void setWeights({List<Tensor>? trainableWeights, List<Tensor>? nonTrainableWeights}) {
    throw UnsupportedError("Cannot set weights on the model");
  }

  /// Constructs and returns model's configuration as a JSON-serializable [Map].
  ///
  /// Includes the configuration for all layers, the loss function, optimizer,
  /// and metrics. If [withWeights] is true, the weights of all layers are also
  /// included in the output.
  @override
  Map<String, dynamic> toJson({bool withWeights = true}) {
    if (!_built) {
      throw ModuleSerializationError.unbuildModel(name);
    }
    return {
      'name': name,
      'trainable': trainable,
      'inputShape': inputShape,
      'loss': loss.toJson(),
      'optimizer': optimizer.toJson(),
      'metrics': [for (Metric metric in metrics) metric.toJson()],
      'layers': [for (Layer layer in layers) layer.toJson(withWeights: withWeights)],
    };
  }

  /// Constructs a [Model] from a JSON-serializable configuration [config].
  @override
  factory Model.fromJson(Map<String, dynamic> config) => Model(
    [for (Map<String, dynamic> layerConfig in config['layers']) Layer.fromJson(layerConfig)],
    loss: Loss.fromJson(config['loss']),
    optimizer: Optimizer.fromJson(config['optimizer']),
    metrics: [for (Map<String, dynamic> metricConfig in config['metrics']) Metric.fromJson(metricConfig)],
    inputShape: config['inputShape'].cast<int>(),
    trainable: config['trainable'],
    name: config['name'],
  );
}
