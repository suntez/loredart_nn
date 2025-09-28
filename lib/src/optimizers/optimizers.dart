import 'package:loredart_nn/src/utils/deserialization_utils.dart';
import 'package:loredart_tensor/loredart_tensor.dart';

import 'dart:math' as dart_math;

/// The optimizer base class.
abstract class Optimizer {
  /// The name of an optimizer.
  late final String name;

  /// The learning rate value.
  late double learningRate;

  /// The weight decay value.
  late final double? weightDecay;

  int _iterations = 0;

  /// The number of iterations [this] has run.
  int get iterations => _iterations;

  /// Creates new optimizer with [learningRate], [weightDecay] and given [name].
  ///
  /// The learning rate must be from (0, 1], otherwise will throw an ArgumentError.
  Optimizer({required this.learningRate, this.weightDecay, required this.name}) {
    if (learningRate > 1 || learningRate <= 0) {
      throw ArgumentError('Learning rate must be between (0, 1], but received $learningRate', 'learningRate');
    }
  }

  /// Applies the [gradients] to a [weights] according to the optimizer's logic.
  ///
  /// The [gradients] and [weights] must be of the same length, otherwise will throw an ArgumentError.
  List<Tensor> applyGradients(List<Tensor> weights, List<Tensor> gradients);

  ///
  Map<String, dynamic> toJson() => {'name': name, 'learningRate': learningRate, 'weightDecay': weightDecay};

  factory Optimizer.fromJson(Map<String, dynamic> config) => deserializeOptimizer(config.remove('type'), config);
}

/// The (mini-batched) gradient descent with momentum and weight decay.
class SGD extends Optimizer {
  /// The momentum of the SGD
  late final double? momentum;

  ///
  late List<Tensor>? momentums;

  /// Creates SGD optimizer with [learningRate] of a given [name].
  ///
  /// The learning rate must be from (0, 1], otherwise will throw an ArgumentError.
  ///
  /// To use a momentum SGD, provide the [momentum] value from [0, 1] range.
  /// If [momentum] is not null and is out of the [0,1] interval will throw an ArgumentError.
  ///
  /// If [weightDecay] is not null will apply weight decay to the weights.
  SGD({double learningRate = 1e-2, double? weightDecay, this.momentum, String name = 'SGD'})
    : super(learningRate: learningRate, weightDecay: weightDecay, name: name) {
    if (momentum != null && (momentum! < 0 || momentum! > 1)) {
      throw ArgumentError('Momentum must be between [0, 1] interval, but received $momentum', 'momentum');
    }
    momentums = (momentum == null || momentum == 0) ? null : [];
  }

  @override
  List<Tensor> applyGradients(List<Tensor> weights, List<Tensor> gradients) {
    if (weights.length != gradients.length) {
      throw ArgumentError(
        'Weights and gradients must be of the same length, but weights.length: ${weights.length} != gradients.length: ${gradients.length}',
      );
    }

    if (weightDecay != null || weightDecay == 0) {
      for (Tensor w in weights) {
        w = w - w * weightDecay!;
      }
    }

    if (momentum == null || momentum == 0) {
      for (int i = 0; i < gradients.length; i += 1) {
        weights[i] -= gradients[i] * learningRate;
      }
    } else {
      if (_iterations == 0) {
        momentums = [for (var w in weights) Tensor.zeros(w.shape.list)];
      }
      for (int i = 0; i < gradients.length; i += 1) {
        momentums![i] = momentums![i] * momentum! - gradients[i] * learningRate;
        weights[i] = weights[i] + momentums![i];
      }
    }
    _iterations += 1;
    return weights;
  }

  @override
  Map<String, dynamic> toJson() =>
      super.toJson()..addAll({
        'type': 'SGD',
        'momentum': momentum,
      });
}

/// Optimizer that implements the Lion algorithm, derived from a [project](https://github.com/google/automl/tree/master/lion).
class Lion extends Optimizer {
  late final double beta1;
  late final double beta2;
  late List<Tensor> _mEMA;

  Lion({double learningRate = 1e-3, double? weightDecay, this.beta1 = 0.9, this.beta2 = 0.99, String name = 'Lion'})
    : super(learningRate: learningRate, name: name, weightDecay: weightDecay);

  @override
  List<Tensor> applyGradients(List<Tensor> weights, List<Tensor> gradients) {
    if (weights.length != gradients.length) {
      throw ArgumentError(
        'Weights and gradients must be of the same length, but weights.length: ${weights.length} != gradients.length: ${gradients.length}',
      );
    }

    if (_iterations == 0) {
      _mEMA = [for (var w in weights) Tensor.zeros(w.shape.list)];
    }
    for (int i = 0; i < gradients.length; i += 1) {
      Tensor c = _mEMA[i] * beta1 + gradients[i] * (1 - beta1);
      weights[i] = weights[i] - (sign(c) + weights[i] * (weightDecay ?? 0)) * learningRate;
      _mEMA[i] = _mEMA[i] * beta2 + gradients[i] * (1 - beta2);
    }
    _iterations += 1;
    return weights;
  }

  @override
  Map<String, dynamic> toJson() =>
      super.toJson()..addAll({
        'type': 'Lion',
        'beta1': beta1,
        'beta2': beta2,
      });
}

/// Optimizer that implements the Adam algorithm, derived from a [paper](https://arxiv.org/abs/2010.07468).
class Adam extends Optimizer {
  late final double beta1;
  late final double beta2;
  late List<Tensor> _mEMA;
  late List<Tensor> _vEMA;
  late final double epsilon;

  Adam({
    double learningRate = 1e-3,
    double? weightDecay,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-7,
    String name = 'Adam',
  }) : super(learningRate: learningRate, name: name, weightDecay: weightDecay);

  @override
  List<Tensor> applyGradients(List<Tensor> weights, List<Tensor> gradients) {
    if (weights.length != gradients.length) {
      throw ArgumentError(
        'Weights and gradients must be of the same length, but weights.length: ${weights.length} != gradients.length: ${gradients.length}',
      );
    }

    if (_iterations == 0) {
      _mEMA = [for (var w in weights) Tensor.zeros(w.shape.list)];
      _vEMA = [for (var w in weights) Tensor.zeros(w.shape.list)];
    }
    for (int i = 0; i < gradients.length; i += 1) {
      _mEMA[i] = _mEMA[i] * beta1 + gradients[i] * (1 - beta1);
      _vEMA[i] = _vEMA[i] * beta2 + square(gradients[i]) * (1 - beta2);
      Tensor mBiasedCorr = _mEMA[i] * (1 / (1 - dart_math.pow(beta1, _iterations + 1)));
      Tensor vBiasedCorr = _vEMA[i] * (1 / (1 - dart_math.pow(beta2, _iterations + 1)));
      weights[i] =
          weights[i] - (mBiasedCorr / (sqrt(vBiasedCorr) + epsilon) + weights[i] * (weightDecay ?? 0)) * learningRate;
    }
    _iterations += 1;
    return weights;
  }

  @override
  Map<String, dynamic> toJson() =>
      super.toJson()..addAll({
        'type': 'Adam',
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon,
      });
}

/// Optimizer that implements the AdaBelief algorithm, derived from a [paper](https://arxiv.org/abs/2010.07468).
class AdaBelief extends Optimizer {
  late final double beta1;
  late final double beta2;
  late List<Tensor> _mEMA;
  late List<Tensor> _sEMA;
  late final double epsilon;

  AdaBelief({
    double learningRate = 1e-3,
    double? weightDecay,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-7,
    String name = 'AdaBelief',
  }) : super(learningRate: learningRate, name: name, weightDecay: weightDecay);

  @override
  List<Tensor> applyGradients(List<Tensor> weights, List<Tensor> gradients) {
    if (weights.length != gradients.length) {
      throw ArgumentError(
        'Weights and gradients must be of the same length, but weights.length: ${weights.length} != gradients.length: ${gradients.length}',
      );
    }

    if (_iterations == 0) {
      _mEMA = [for (var w in weights) Tensor.zeros(w.shape.list)];
      _sEMA = [for (var w in weights) Tensor.zeros(w.shape.list)];
    }
    for (int i = 0; i < gradients.length; i += 1) {
      _mEMA[i] = _mEMA[i] * beta1 + gradients[i] * (1 - beta1);
      _sEMA[i] = _sEMA[i] * beta2 + square(gradients[i] - _mEMA[i]) * (1 - beta2) + epsilon;
      Tensor mBiasedCorr = _mEMA[i] * (1 / (1 - dart_math.pow(beta1, _iterations + 1)));
      Tensor sBiasedCorr = _sEMA[i] * (1 / (1 - dart_math.pow(beta2, _iterations + 1)));
      weights[i] =
          weights[i] - (mBiasedCorr / (sqrt(sBiasedCorr) + epsilon) + weights[i] * (weightDecay ?? 0)) * learningRate;
    }
    _iterations += 1;
    return weights;
  }

  @override
  Map<String, dynamic> toJson() =>
      super.toJson()..addAll({
        'type': 'AdaBelief',
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon,
      });
}
