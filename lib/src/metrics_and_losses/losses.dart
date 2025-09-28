import 'package:loredart_nn/src/nn_ops/gradients/metric_grads.dart';
import 'package:loredart_nn/src/utils/deserialization_utils.dart';
import 'package:loredart_tensor/loredart_tensor.dart';
import '../nn_ops/activations.dart';
import '../nn_ops/metric_ops.dart';
import '/src/modules/module.dart';

/// Type of loss reduction.
///
/// Reduction affects the shape and value(s) of the loss and its gradients.
enum Reduction {
  /// Scalar [sum] divided by the number of the loss elements.
  sumOverBatchSize,

  /// Scalar sum of the losses.
  sum,

  /// Absence of additional loss reduction.
  ///
  /// Loss with [none] reduction may return non-scalar Tensor,
  /// However, the [Model] will print the averaged value anyway.
  none,
}

/// The loss base class.
abstract class Loss implements ComputationalNode {
  /// Type of reduction that applies to loss.
  late final Reduction reduction;

  /// The name of the loss
  late final String name;

  /// Creates new Loss with [name] and [reduction] rule.
  Loss({required this.reduction, required this.name});

  /// Computes loss according to it's logic using [yTrue] and [yPred].
  ///
  /// If [training] is true, additionally constructs the `gradient` function of the loss.
  ///
  /// Returns the [Tensor] of the same [DType] as [yTrue] and [yPred].
  /// The shape of the output depends on the [reduction].
  Tensor call(Tensor yTrue, Tensor yPred, {bool training = false});

  factory Loss.fromJson(Map<String, dynamic> config) => deserializeLoss(config.remove('type'), config);

  Map<String, dynamic> toJson();
}

/// The mean of squares of errors between labels and predictions.
///
/// `loss = squareDifference(yTrue, yPred)`
///
/// [yTrue] and [yPred] must be equal-shaped [Tensor]s of the same [DType].
///
/// Example of usage:
/// ```dart
/// // two batches with two entries
/// final yTrue = Tensor.constant([[1.0, 0.0], [1.0, 0.0]]);
/// final yPred = Tensor.constant([[1.0, 1.0], [0.0, 1.0]]);
///
/// // Loss with batch reduction
/// Loss mse = MeanSquaredError();
/// mse(yTrue, yPred); // Tensor(shape: [1], values: [[0.75]], ...)
///
/// // With sum reduction
/// mse = MeanSquaredError(reduction: Reduction.sum);
/// mse(yTrue, yPred); // Tensor(shape: [1], values: [[1.5]], ...)
///
/// // Without reduction
/// mse = MeanSquaredError(reduction: Reduction.none);
/// mse(yTrue, yPred); // Tensor(shape: [2], values: [[0.5, 1.0]], ...)
/// ```
///
/// Example as loss parameter in a [Model]:
/// ```dart
/// Model model = Model(
///   [...],
///   optimizer: Adam(),
///   loss: MeanSquaredError()
/// );
///```
class MeanSquaredError extends Loss {
  @override
  List<Tensor> Function(Tensor)? gradient;

  MeanSquaredError({Reduction reduction = Reduction.sumOverBatchSize}) : super(reduction: reduction, name: 'MSE');

  @override
  Tensor call(Tensor yTrue, Tensor yPred, {bool training = false}) {
    final sqrtDiff = reduceMean(squareDifference(yTrue, yPred), axis: [-1]);

    if (training) {
      int batches = yPred.rank == 1 ? 1 : yPred.shape.list.sublist(0, yPred.rank - 1).reduce((e1, e2) => e1 * e2);
      gradient =
          (upstream) => [
            upstream * (yPred - yTrue) * 2 / (reduction == Reduction.sumOverBatchSize ? batches : yPred.shape[-1]),
          ];
    }

    if (reduction == Reduction.none) {
      return sqrtDiff;
    } else if (reduction == Reduction.sum) {
      return reduceSum(sqrtDiff);
    } else {
      return mean(sqrtDiff);
    }
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'MeanSquaredError', 'reduction': reduction.index};
}

/// The mean of absolute difference between labels and predictions.
///
/// `loss = abs(yTrue - yPred)`
///
/// [yTrue] and [yPred] must be equal-shaped [Tensor]s of the same [DType].
///
/// Example of usage:
/// ```dart
/// // two batches with two entries
/// final yTrue = Tensor.constant([[1.0, 0.0], [1.0, 0.0]]);
/// final yPred = Tensor.constant([[1.0, 1.0], [0.0, 1.0]]);
///
/// // Loss with batch reduction
/// Loss mae = MeanAbsoluteError();
/// mae(yTrue, yPred); // Tensor(shape: [1], values: [[0.75]], ...)
///
/// // With sum reduction
/// mae = MeanAbsoluteError(reduction: Reduction.sum);
/// mae(yTrue, yPred); // Tensor(shape: [1], values: [[1.5]], ...)
///
/// // Without reduction
/// mae = MeanAbsoluteError(reduction: Reduction.none);
/// mae(yTrue, yPred); // Tensor(shape: [2], values: [[0.5, 1.0]], ...)
/// ```
///
/// Example as loss parameter in a [Model]:
/// ```dart
/// Model model = Model(
///   [...],
///   optimizer: Adam(),
///   loss: MeanAbsoluteError()
/// );
///```
class MeanAbsoluteError extends Loss {
  @override
  List<Tensor> Function(Tensor)? gradient;

  MeanAbsoluteError({Reduction reduction = Reduction.sumOverBatchSize}) : super(reduction: reduction, name: 'MAE');

  @override
  Tensor call(Tensor yTrue, Tensor yPred, {bool training = false}) {
    final asbDiff = reduceMean(abs(yTrue - yPred), axis: [-1]);

    if (training) {
      int batches = yPred.rank == 1 ? 1 : yPred.shape.list.sublist(0, yPred.rank - 1).reduce((e1, e2) => e1 * e2);
      gradient =
          (upstream) => [
            upstream * sign(yPred - yTrue) / (reduction == Reduction.sumOverBatchSize ? batches : yPred.shape[-1]),
          ];
    }

    if (reduction == Reduction.none) {
      return asbDiff;
    } else if (reduction == Reduction.sum) {
      return reduceSum(asbDiff);
    } else {
      return mean(asbDiff);
    }
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'MeanAbsoluteError', 'reduction': reduction.index};
}

/// The crossentropy loss between (binary) true labels and predicted labels.
///
/// `loss = -(yTrue*log(yPred) + (1-yTrue)*log(1-yPred))`
///
/// Should be used for the binary (0-1) classification.
///
/// Assumes that [yPred] is a 1-D binary Tensor with values 0, 1.
/// Expects that [yPred] is a 1-D float-based Tensor filled with either:
///  - `logit` - values from range [-inf, +inf] with `this.fromLogits == true`
///  - or `probabilities` (for a class 1) - values between [0, 1] with `this.fromLogits == false`
///
/// As in [TensorFlow version](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) -
/// the recommended usage is to set `from_logits: true`.
///
/// [yTrue] and [yPred] must be equal-shaped [Tensor]s of the same [DType].
///
/// Example of usage with logits:
/// ```dart
/// // two batches with 2 entries
/// final yTrue = Tensor.constant([[1.0, 0.0], [1.0, 0.0]]);
/// final yPred = Tensor.constant([[13.1, -0.5], [-1.3, 0.0]]);
///
/// // Loss with batch reduction
/// Loss bce = BinaryCrossentropy(fromLogits: true);
/// bce(yTrue, yPred); // 0.677058
///
/// // With sum reduction
/// bce = BinaryCrossentropy(fromLogits: true, reduction: Reduction.sum);
/// bce(yTrue, yPred); // 1.354117
///
/// // Without reduction
/// bce = BinaryCrossentropy(fromLogits: true, reduction: Reduction.none);
/// bce(yTrue, yPred); [0.237039, 1.117077]
/// ```
///
/// Example with probabilities:
/// ```dart
/// // two batches with 2 entries
/// final yTrue = Tensor.constant([[1.0, 0.0], [1.0, 0.0]]);
/// // probabilities
/// final yPred = Tensor.constant([[0.9, 0.05], [0.1, 0.1]]);
///
/// Loss bce = BinaryCrossentropy();
/// print(bce(yTrue, yPred)); // 0.641149
/// ```
///
/// Example as loss parameter in a [Model]:
/// ```dart
/// // Recommended usage:
/// Model modelWithLogits = Model(
///   [
///     ...,
///     Dense(1)
///   ],
///   optimizer: Adam(),
///   loss: BinaryCrossentropy(fromLogits: true)
/// );
///
/// // Possible usage:
/// Model modelWithProbs = Model(
///   [
///     ...,
///     Dense(1, activation: Activations.sigmoid)
///   ],
///   optimizer: Adam(),
///   loss: BinaryCrossentropy()
/// );
/// ```
class BinaryCrossentropy extends Loss {
  /// Whether the prediction Tensor is expected to be a logit
  late final bool fromLogits;

  @override
  List<Tensor> Function(Tensor)? gradient;

  BinaryCrossentropy({this.fromLogits = false, Reduction reduction = Reduction.sumOverBatchSize})
    : super(reduction: reduction, name: 'BinaryCrossentropy');

  @override
  Tensor call(Tensor yTrue, Tensor yPred, {bool training = false}) {
    if (fromLogits) {
      yPred = sigmoid(yPred);
    }
    final bce = binaryCrossentropy(yTrue, yPred);

    if (training) {
      int batches = yPred.rank == 1 ? 1 : yPred.shape.list.sublist(0, yPred.rank - 1).reduce((e1, e2) => e1 * e2);
      gradient =
          (upstream) => [
            upstream *
                binaryCrossentropyGrad(yTrue, yPred, fromLogits: fromLogits) /
                (reduction == Reduction.sumOverBatchSize ? batches : 1),
          ];
    }

    if (reduction == Reduction.none) {
      return bce;
    } else if (reduction == Reduction.sum) {
      return reduceSum(bce);
    } else {
      return mean(bce);
    }
  }

  @override
  Map<String, dynamic> toJson() => {
    'type': 'BinaryCrossentropy',
    'fromLogits': fromLogits,
    'reduction': reduction.index,
  };
}

/// The crossentropy loss between true labels and predicted labels.
///
/// `loss = -reduceSum(yTrue*log(yPred), axis: [-1])`
///
/// Can be used for the multi-class classification problem.
///
/// Expects [yPred] as a one-hot Tensor representation. For labels encoded as integers see [SparseCategoricalCrossentropy] or use `sparse: true`.
///
/// Expects [yPred] is a [batchSize, numClasses] shaped, float-based Tensor filled with either:
///  - `logits` - values from range [-inf, +inf] with `this.fromLogits == true`
///  - or `probabilities` - values between [0, 1] with `this.fromLogits == false`
///
///
/// [yTrue] and [yPred] must be equal-shaped [Tensor]s (of shape [batchSize, numClasses]).
/// The labels may be provided as integer-based Tensor, the loss will automatically cast it.
///
/// Example of usage with logits:
/// ```dart
/// // single batch with 2 entries for a 3-class problem
/// final yTrue = Tensor.constant([[0, 1, 0], [0, 0, 1]]);
/// final yPred = Tensor.constant([[2.0, 10.9, -0.5], [-1.0, 5.0, 4.5]]);
///
/// // Loss with batch reduction
/// Loss cce = CategoricalCrossentropy(fromLogits: true);
/// cce(yTrue, yPred); // 0.487883
///
/// // With sum reduction
/// cce = CategoricalCrossentropy(fromLogits: true, reduction: Reduction.sum);
/// cce(yTrue, yPred); // 0.975766
///
/// // Without reduction
/// cce = CategoricalCrossentropy(fromLogits: true, reduction: Reduction.none);
/// cce(yTrue, yPred); // [0.000147, 0.975618]
/// ```
///
/// Example with probabilities:
/// ```dart
/// // single batch with 2 entries for a 3-class problem
/// final yTrue = Tensor.constant([[0, 1, 0], [0, 0, 1]]);
/// // probabilities
/// final yPred = Tensor.constant([[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]]);
///
/// Loss cce = CategoricalCrossentropy();
/// cce(yTrue, yPred); // 1.176939
/// ```
///
/// Example as loss parameter in a [Model]:
/// ```dart
/// // Recommended usage:
/// Model modelWithLogits = Model(
///   [
///     ...,
///     Dense(n)
///   ],
///   optimizer: Adam(),
///   loss: CategoricalCrossentropy(fromLogits: true)
/// );
///
/// // Possible usage:
/// Model modelWithProbs = Model(
///   [
///     ...,
///     Dense(n, activation: Activations.softmax)
///   ],
///   optimizer: Adam(),
///   loss: CategoricalCrossentropy()
/// );
/// ```
class CategoricalCrossentropy extends Loss {
  /// Whether the labels are expected to be in a sparse form.
  late final bool sparse;

  /// Whether the prediction Tensor is expected to be a logits
  late final bool fromLogits;

  @override
  List<Tensor> Function(Tensor)? gradient;

  CategoricalCrossentropy({
    this.fromLogits = false,
    this.sparse = false,
    Reduction reduction = Reduction.sumOverBatchSize,
  }) : super(reduction: reduction, name: 'CategoricalCrossentropy');

  @override
  Tensor call(Tensor yTrue, Tensor yPred, {bool training = false}) {
    if (sparse) {
      yTrue = oneHotTensor(yTrue, depth: yPred.shape[-1], dType: yPred.dType);
    }
    if (fromLogits) {
      yPred = softmax(yPred, stable: true);
    }
    if (yTrue.dType.isInt) {
      yTrue = cast(yTrue, yPred.dType);
    }
    final cce = crossEntropy(yTrue, yPred);

    if (training) {
      int batches = yPred.rank == 1 ? 1 : yPred.shape.list.sublist(0, yPred.rank - 1).reduce((e1, e2) => e1 * e2);
      gradient =
          (upstream) => [
            upstream *
                crossEntropyGrad(yTrue, yPred, fromLogits: fromLogits) /
                (reduction == Reduction.sumOverBatchSize ? batches : 1),
          ];
    }

    if (reduction == Reduction.none) {
      return cce;
    } else if (reduction == Reduction.sum) {
      return reduceSum(cce);
    } else {
      return mean(cce);
    }
  }

  @override
  Map<String, dynamic> toJson() => {
    'type': 'CategoricalCrossentropy',
    'fromLogits': fromLogits,
    'reduction': reduction.index,
  };
}

/// The crossentropy loss between true labels and predicted labels.
///
/// `loss = -reduceSum(yTrue*log(yPred), axis: [-1])`
///
/// Can be used for the multi-class classification problem where the labels are integer encoded.
/// Essentially [this] is just a [CategoricalCrossentropy] with `sparse: true`.
///
/// Expects [yPred] as a batched, integer-based Tensor with classes indexes.
///
/// Expects [yPred] as a [batchSize, numClasses] shaped, float-based Tensor filled with either:
///  - `logits` - values from range [-inf, +inf] with `this.fromLogits == true`
///  - or `probabilities` - values between [0, 1] with `this.fromLogits == false`
///
/// The number of classes is derived as a `yPred.shape[-1]`.
///
/// [yTrue] must be integer-based Tensor.
///
/// Example of usage with logits:
/// ```dart
/// // single batch with 2 entries for a 3-class problem
/// final yTrue = Tensor.constant([1, 2]);
/// final yPred = Tensor.constant([[2.0, 10.9, -0.5], [-1.0, 5.0, 4.5]]);
///
/// // Loss with batch reduction
/// Loss scce = SparseCategoricalCrossentropy(fromLogits: true);
/// scce(yTrue, yPred); // 0.487883
///
/// // With sum reduction
/// scce = SparseCategoricalCrossentropy(fromLogits: true, reduction: Reduction.sum);
/// scce(yTrue, yPred); // 0.975766
///
/// // Without reduction
/// scce = SparseCategoricalCrossentropy(fromLogits: true, reduction: Reduction.none);
/// scce(yTrue, yPred); // [0.000147, 0.975618]
/// ```
///
/// Example with probabilities:
/// ```dart
/// // single batch with 2 entries for a 3-class problem
/// final yTrue = Tensor.constant([1, 2]);
/// // probabilities
/// final yPred = Tensor.constant([[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]]);
///
/// Loss scce = SparseCategoricalCrossentropy();
/// scce(yTrue, yPred); // 1.176939
/// ```
///
/// Example as loss parameter in a [Model]:
/// ```dart
/// // Recommended usage:
/// Model modelWithLogits = Model(
///   [
///     ...,
///     Dense(n)
///   ],
///   optimizer: Adam(),
///   loss: SparseCategoricalCrossentropy(fromLogits: true)
/// );
///
/// // Possible usage:
/// Model modelWithProbs = Model(
///   [
///     ...,
///     Dense(n, activation: Activations.softmax)
///   ],
///   optimizer: Adam(),
///   loss: SparseCategoricalCrossentropy()
/// );
/// ```
class SparseCategoricalCrossentropy extends CategoricalCrossentropy {
  SparseCategoricalCrossentropy({bool fromLogits = false, Reduction reduction = Reduction.sumOverBatchSize})
    : super(fromLogits: fromLogits, sparse: true, reduction: reduction);

  @override
  Map<String, dynamic> toJson() => {
    'type': 'SparseCategoricalCrossentropy',
    'fromLogits': fromLogits,
    'reduction': reduction.index,
  };
}

/// The logarithm of the hyperbolic cosine of errors between labels and predictions.
///
/// `loss = log(cosh(yTrue - yPred))`
///
/// [yTrue] and [yPred] must be equal-shaped [Tensor]s of the same [DType].
///
/// Example of usage:
/// ```dart
/// // two batches with two entries
/// final yTrue = Tensor.constant([[1.0, 0.0], [1.0, 0.0]]);
/// final yPred = Tensor.constant([[1.0, 1.0], [0.0, 1.0]]);
///
/// // Loss with batch reduction
/// Loss lc = LogCosh();
/// print(lc(yTrue, yPred)); // Tensor(shape: [1], values: [[0.32533]], ...)
///
/// // With sum reduction
/// lc = LogCosh(reduction: Reduction.sum);
/// print(lc(yTrue, yPred)); // Tensor(shape: [1], values: [[0.65067]], ...)
///
/// // Without reduction
/// lc = LogCosh(reduction: Reduction.none);
/// print(lc(yTrue, yPred)); // Tensor(shape: [2], values: [[0.21689, 0.43378]], ...)
/// ```
///
/// Example as loss parameter in a [Model]:
/// ```dart
/// Model model = Model(
///   [...],
///   optimizer: Adam(),
///   loss: LogCosh()
/// );
///```
class LogCoshError extends Loss {
  @override
  List<Tensor> Function(Tensor)? gradient;

  LogCoshError({Reduction reduction = Reduction.sumOverBatchSize}) : super(reduction: reduction, name: 'LogCosh');

  @override
  Tensor call(Tensor yTrue, Tensor yPred, {bool training = false}) {
    final logcosh = logCosh(yTrue, yPred);

    if (training) {
      int batches = yPred.rank == 1 ? 1 : yPred.shape.list.sublist(0, yPred.rank - 1).reduce((e1, e2) => e1 * e2);
      gradient =
          (upstream) => [
            upstream *
                logCoshGrad(yTrue, yPred) /
                (reduction == Reduction.sumOverBatchSize ? batches : yPred.shape[-1]),
          ];
    }

    if (reduction == Reduction.none) {
      return logcosh;
    } else if (reduction == Reduction.sum) {
      return reduceSum(logcosh);
    } else {
      return mean(logcosh);
    }
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'LogCoshError', 'reduction': reduction.index};
}
