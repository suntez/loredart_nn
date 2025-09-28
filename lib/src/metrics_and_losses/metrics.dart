import 'package:loredart_nn/src/nn_ops/activations.dart';
import 'package:loredart_nn/src/utils/deserialization_utils.dart';
import 'package:loredart_tensor/loredart_tensor.dart';

import '../nn_ops/metric_ops.dart';

/// Base class for representing a metric for evaluating model performance.
abstract class Metric {
  /// Accumulates metric statistics based on values of [yTrue] and [yTest].
  ///
  /// The [yTrue] and [yPred] tensors should have equal shapes, unless dealing with sparse metrics.
  void updateState(Tensor yTrue, Tensor yPred);

  /// Reset, i.e. clean, all metric's state variables.
  void resetState();

  /// The scalar metric value computed from state variables.
  num? get result;

  /// The name of the metrics displayed in the [Model]'s history.
  String get name;

  ///
  factory Metric.fromJson(Map<String, dynamic> config) => deserializeMetric(config.remove('type'), config);

  ///
  Map<String, dynamic> toJson();
}

/// Metric that compute mean of accumulated states.
class MeanMetric implements Metric {
  @override
  late final String name;

  /// The accumulation of metric statistics (e.g. sum)
  double _total = 0;

  /// Number of updates
  int _count = 0;

  MeanMetric({required this.name});

  @override
  void resetState() {
    _total = 0;
    _count = 0;
  }

  @override
  void updateState(Tensor yTrue, Tensor yPred) => throw UnimplementedError();

  @override
  double? get result => _count == 0 ? null : _total / _count;

  void _checkShapes(TensorShape yTrueShape, TensorShape yPredShape, {bool fromSparse = false}) {
    if (!fromSparse && !yTrueShape.equalTo(yPredShape)) {
      throw ArgumentError(
        'yTrue and yPred should have equal shapes, but metric $name received $yTrueShape and $yPredShape',
      );
    } else if (fromSparse) {
      if (yTrueShape.rank != yPredShape.rank - 1) {
        throw ArgumentError(
          'Expected yTrue.rank = yPred.rank-1, but metric $name received yTrue.rank: ${yTrueShape.rank} and yPred.rank: ${yPredShape.rank}.',
        );
      }
      if (!yTrueShape.equalTo(TensorShape(yPredShape.list.sublist(0, yPredShape.rank - 1)))) {
        throw ArgumentError(
          'yTrue and yPred should have equal batch shapes, but metric $name received $yTrueShape and $yPredShape',
        );
      }
    }
  }

  @override
  String toString() => '$runtimeType(name: $name)';

  @override
  Map<String, dynamic> toJson() => throw UnimplementedError();
}

/// The mean absolute error between labels and predictions.
///
/// Example:
/// ```dart
/// final mae = MeanAbsoluteErrorMetric();
///
/// mae.updateState(
///   Tensor.constant([[0, 1], [0, 0]]), // yTrue
///   Tensor.constant([[0, 1], [1, 0]])  // yPred
/// );
///
/// print(mae.result); // 0.25
/// ```
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [MeanAbsoluteErrorMetric()]
///   // or with Metrics.meanAbsoluteError()
/// );
/// ```
class MeanAbsoluteErrorMetric extends MeanMetric {
  MeanAbsoluteErrorMetric({String? name}) : super(name: name ?? 'mae');
  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);
    NumericTensor value = mean(abs(yTrue - yPred), dType: DType.float32) as NumericTensor;
    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'MeanAbsoluteErrorMetric', 'name': name};
}

/// The mean squared error between labels and predictions.
///
/// Example:
/// ```dart
/// final mse = MeanSquaredErrorMetric();
///
/// mse.updateState(
///   Tensor.constant([[0, 1], [0, 0]]), // yTrue
///   Tensor.constant([[0, 1], [1, 0]])  // yPred
/// );
///
/// print(mse.result); // 0.25
/// ```
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [MeanSquaredErrorMetric()]
///   // or with Metrics.meanSquaredError()
/// );
/// ```
class MeanSquaredErrorMetric extends MeanMetric {
  MeanSquaredErrorMetric({String? name}) : super(name: name ?? 'mse');
  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);
    NumericTensor value = mean(squareDifference(yTrue, yPred)) as NumericTensor;
    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'MeanSquaredErrorMetric', 'name': name};
}

/// The root of mean squared error between labels and predictions.
///
/// Example:
/// ```dart
/// final rmse = RootMeanSquaredErrorMetric();
///
/// rmse.updateState(
///   Tensor.constant([[0, 1], [0, 0]]), // yTrue
///   Tensor.constant([[0, 1], [1, 0]])  // yPred
/// );
///
/// print(rmse.result); // 0.5
/// ```
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [RootMeanSquaredErrorMetric()]
///   // or with Metrics.rootMeanSquaredErrorMetric()
/// );
/// ```
class RootMeanSquaredErrorMetric extends MeanMetric {
  RootMeanSquaredErrorMetric({String? name}) : super(name: name ?? 'root_mse');
  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);
    NumericTensor value = sqrt(mean(squareDifference(yTrue, yPred))) as NumericTensor;
    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'RootMeanSquaredErrorMetric', 'name': name};
}

/// The accuracy metric.
///
/// Computes the mean match of predictions and one-hot encoded label classes.
/// The [yPred] can be represented as logits, since argmax will be the same as with probabilities.
///
/// Example:
/// ```dart
/// final acc = CategoricalAccuracyMetric();
///
/// acc.updateState(
///   Tensor.constant([[0,     1], [1,       0]]), // yTrue
///   Tensor.constant([[0.1, 0.9], [0.45, 0.55]])  // yPred
/// );
///
/// print(acc.result); // 0.5
/// ```
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [CategoricalAccuracyMetric()]
///   // or with Metrics.categoricalAccuracy()
/// );
/// ```
class CategoricalAccuracyMetric extends MeanMetric {
  CategoricalAccuracyMetric({String? name}) : super(name: name ?? 'categorical_accuracy');

  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);

    NumericTensor value =
        mean(equal(argMax(yTrue, axis: -1), argMax(yPred, axis: -1)), dType: DType.float32) as NumericTensor;

    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'CategoricalAccuracyMetric', 'name': name};
}

/// The sparse version of accuracy metric.
///
/// Computes the mean match of predictions and label indices.
/// The [yPred] can be represented as logits, since argmax will be the same as with probabilities.
///
/// Example:
/// ```dart
/// final sacc = SparseCategoricalAccuracyMetric();
///
/// sacc.updateState(
///   Tensor.constant([0, 1]), // yTrue
///   Tensor.constant([[0.1, 0.9], [0.45, 0.55]])  // yPred
/// );
///
/// print(sacc.result); // 0.5
/// ```
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [SparseCategoricalAccuracy()]
///   // or with Metrics.sparseCategoricalAccuracy()
/// );
/// ```
class SparseCategoricalAccuracyMetric extends MeanMetric {
  SparseCategoricalAccuracyMetric({String? name}) : super(name: name ?? 'sparse_categorical_accuracy');

  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape, fromSparse: true);

    NumericTensor value = mean(equal(yTrue, argMax(yPred, axis: -1)), dType: DType.float32) as NumericTensor;

    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'SparseCategoricalAccuracyMetric', 'name': name};
}

/// The binary accuracy metric.
///
/// Computes the mean match of predictions and binary label on the given [threshold].
///
/// Example:
/// ```dart
/// final bacc = BinaryAccuracyMetric(); // with default threshold 0.5
///
/// bacc.updateState(
///   Tensor.constant([1, 0, 1, 0]), // yTrue
///   Tensor.constant([0.9, 0.1, 0.3, 0.51])  // yPred
/// );
///
/// print(bacc.result); // 0.5
/// ```
///
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [BinaryAccuracyMetric()]
///   // or with Metrics.binaryAccuracy()
/// );
/// ```
class BinaryAccuracyMetric extends MeanMetric {
  late double threshold;
  BinaryAccuracyMetric({this.threshold = 0.5, String? name}) : super(name: name ?? 'binary_accuracy') {
    if (threshold < 0 || threshold >= 1) {
      throw ArgumentError('The threshold value must be from [0, 1) interval, but metric $name received $threshold');
    }
  }

  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);

    NumericTensor value =
        mean(
              equal(
                cast(yTrue, yPred.dType),
                greaterEqual(yPred, threshold),
              ),
              dType: DType.float32,
            )
            as NumericTensor;

    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'BinaryAccuracyMetric', 'name': name, 'threshold': threshold};
}

/// The crossentropy between labels and predictions.
///
/// Assumes that labels are one-hot encoded.
///
/// The predictions can be either probabilities or logits (with [fromLogits] set to true)
///
/// Example:
/// ```dart
/// final ccs = CategoricalCrossentropyMetric();
///
/// ccs.updateState(
///   Tensor.constant([[1, 0], [0, 1]]), // yTrue
///   Tensor.constant([[0.9, 0.1], [0.49, 0.51]])  // yPred
/// );
///
/// print(ccs.result); // 0.38935
/// ```
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [CategoricalCrossentropyMetric()]
///   // or with Metrics.categoricalCrossentropy()
/// );
/// ```
class CategoricalCrossentropyMetric extends MeanMetric {
  late bool fromLogits;
  late bool sparse;
  CategoricalCrossentropyMetric({this.fromLogits = false, this.sparse = false, String? name})
    : super(name: name ?? 'categorical_crossentropy');

  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape, fromSparse: sparse);
    if (sparse) {
      yTrue = oneHotTensor(yTrue, depth: yPred.shape[-1], dType: yPred.dType);
    }
    if (fromLogits) {
      yPred = softmax(yPred);
    }

    NumericTensor value = mean(crossEntropy(cast(yTrue, yPred.dType), yPred)) as NumericTensor;

    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'CategoricalCrossentropyMetric', 'name': name, 'fromLogits': fromLogits};
}

/// The sparse version of crossentropy between labels and predictions.
///
/// Assumes that labels are indices.
///
/// The predictions can be either probabilities or logits (with [fromLogits] set to true)
///
/// Example:
/// ```dart
/// final scce = SparseCategoricalCrossentropyMetric();
///
/// scce.updateState(
///   Tensor.constant([0, 1]), // yTrue
///   Tensor.constant([[0.9, 0.1], [0.49, 0.51]])  // yPred
/// );
///
/// print(scce.result); // 0.38935
/// ```
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [SparseCategoricalCrossentropyMetric()]
///   // or with Metrics.sparseCategoricalCrossentropy()
/// );
/// ```
class SparseCategoricalCrossentropyMetric extends CategoricalCrossentropyMetric {
  SparseCategoricalCrossentropyMetric({bool fromLogits = false, String? name})
    : super(fromLogits: fromLogits, sparse: true, name: name ?? 'sparse_categorical_crossentropy');

  @override
  Map<String, dynamic> toJson() => {
    'type': 'SparseCategoricalCrossentropyMetric',
    'name': name,
    'fromLogits': fromLogits,
  };
}

/// The binary crossentropy between labels and predictions.
///
/// Assumes that labels are binary classes.
///
/// The predictions can be either probabilities or logits (with [fromLogits] set to true)
///
/// Example:
/// ```dart
/// final bcce = BinaryCrossentropyMetric();

/// bcce.updateState(
///   Tensor.constant([[1, 0, 1, 0]]), // yTrue
///   Tensor.constant([[0.9, 0.1, 0.3, 0.51]])  // yPred
/// );
///
/// print(bcce.result); // 0.53201
/// ```
///
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [BinaryCrossentropyMetric()]
///   // or with Metrics.binaryCrossentropy()
/// );
/// ```
class BinaryCrossentropyMetric extends MeanMetric {
  late bool fromLogits;
  BinaryCrossentropyMetric({this.fromLogits = false, String? name}) : super(name: name ?? 'binary_crossentropy');

  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);
    if (fromLogits) {
      yPred = sigmoid(yPred);
    }

    NumericTensor value = mean(binaryCrossentropy(cast(yTrue, yPred.dType), yPred)) as NumericTensor;

    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'BinaryCrossentropyMetric', 'name': name, 'fromLogits': fromLogits};
}

/// The log cosh error between labels and predictions.
///
/// Example:
/// ```dart
/// final lce = LogCoshErrorMetric();
///
/// lce.updateState(
///   Tensor.constant([[0, 1], [0, 0]]), // yTrue
///   Tensor.constant([[0, 1], [1, 0]])  // yPred
/// );
///
/// print(lce.result); // 0.10845
/// ```
///
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [LogCoshErrorMetricMetric()]
///   // or with Metrics.logCoshError()
/// );
/// ```
class LogCoshErrorMetric extends MeanMetric {
  LogCoshErrorMetric({String? name}) : super(name: name ?? 'log_cosh');

  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);

    NumericTensor value = mean(logCosh(yTrue, yPred)) as NumericTensor;

    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'LogCoshErrorMetric', 'name': name};
}

/// The Kullback-Leibler divergence metric between true and predicted values.
///
/// Example:
/// ```dart
/// final kld = KLDivergenceMetric();
///
/// kld.updateState(
///   Tensor.constant([[1, 0], [0, 1]]), // yTrue
///   Tensor.constant([[0.1, 0.9], [0.45, 0.55]])  // yPred
/// );
///
/// print(kld.result); // 1.45021
/// ```
///
/// Usage in the model:
/// ```dart
/// Model model = Model(
///   ...
///   metrics: [KLDivergenceMetric()]
///   // or with Metrics.klDivergence()
/// );
/// ```
class KLDivergenceMetric extends MeanMetric {
  KLDivergenceMetric({String? name}) : super(name: name ?? 'kl_divergence');

  @override
  void updateState(Tensor yTrue, Tensor yPred) {
    _checkShapes(yTrue.shape, yPred.shape);

    NumericTensor value = mean(klDivergence(cast(yTrue, yPred.dType), yPred)) as NumericTensor;

    _total += value.buffer[0];
    _count += 1;
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'KLDivergenceMetric', 'name': name};
}

/// The collection of supported metrics.
class Metrics {
  /// The mean absolute error metric.
  static Metric get mae => MeanAbsoluteErrorMetric();

  /// The mean squared error metric.
  static Metric get mse => MeanSquaredErrorMetric();

  /// The root mean squared error metric.
  static Metric get rootSquareMeanError => RootMeanSquaredErrorMetric();

  /// The accuracy metric.
  static Metric get categoricalAccuracy => CategoricalAccuracyMetric();

  /// The sparse accuracy metric.
  static Metric get sparseCategoricalAccuracy => SparseCategoricalAccuracyMetric();

  /// The binary accuracy.
  static Metric get binaryAccuracy => BinaryAccuracyMetric();

  /// The categorical crossentropy metric.
  static Metric get categoricalCrossentropy => CategoricalCrossentropyMetric();

  /// The categorical crossentropy metric from logits.
  static Metric get categoricalCrossentropyFromLogits => CategoricalCrossentropyMetric(fromLogits: true);

  /// The sparse categorical crossentropy metric.
  static Metric get sparseCategoricalCrossentropy => SparseCategoricalCrossentropyMetric();

  /// The sparse categorical crossentropy metric from logits.
  static Metric get sparseCategoricalCrossentropyFromLogits => SparseCategoricalCrossentropyMetric(fromLogits: true);

  /// The binary crossentropy metric.
  static Metric get binaryCrossentropy => BinaryCrossentropyMetric();

  /// The Binary crossentropy metric from logits.
  static Metric get binaryCrossentropyFromLogits => BinaryCrossentropyMetric(fromLogits: true);

  /// The log cosh error metric.
  static Metric get logCoshError => LogCoshErrorMetric();

  /// The KL divergence metric.
  static Metric get klDivergence => KLDivergenceMetric();
}
