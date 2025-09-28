import 'package:loredart_nn/src/nn_ops/gradients/math_grads.dart';
import 'package:loredart_tensor/loredart_tensor.dart';

import '../utils/deserialization_utils.dart';
import '/src/nn_ops/activations.dart';
import '/src/modules/module.dart';

/// The activations base class.
///
/// [Activation] is a wrapper of activation function, that can construct gradient function w.r.t to the inputs.
abstract class Activation implements ComputationalNode {
  const Activation();

  factory Activation.fromJson(Map<String, dynamic> config) => deserializeActivation(config.remove('type'), config);

  /// Applies activation to the input [x].
  ///
  /// If [training] is true will construct the [gradient] function.
  ///
  /// Returns the [Tensor] of the same shape as [x].
  Tensor call(Tensor x, {bool training = false});

  Map<String, dynamic> toJson() => {'type': runtimeType.toString()};

  @override
  String toString() => "$runtimeType()";
}

/// Linear activation function.
///
/// `linear(x) = x`
///
/// Linear is a default function of the Layer.
///
/// Also accessible via [Activations.linear].
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = Linear()(x);
/// print(y); // [-5.0, 0.0, 5.0]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.linear),
///   ...
/// ], ...);
/// ```
class Linear extends Activation {
  @override
  List<Tensor> Function(Tensor)? gradient = (upstream) => [upstream * Tensor.ones(upstream.shape.list)];

  @override
  Tensor call(Tensor x, {bool training = false}) {
    return x;
  }
}

/// Sigmoid activation function.
///
/// `sigmoid(x) = 1/(1 + exp(-x))`
///
/// Sigmoid is equivalent to the 2-element softmax, with second element being 0.
/// Returned values always from the interval [0, 1].
///
/// Also accessible via [Activations.sigmoid].
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = Sigmoid()(x);
/// print(y); // [0.006692, 0.5, 0.993307]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.sigmoid),
///   ...
/// ], ...);
/// ```
class Sigmoid extends Activation {
  @override
  List<Tensor> Function(Tensor)? gradient;

  @override
  Tensor call(Tensor x, {bool training = false}) {
    final s = sigmoid(x);
    if (training) {
      gradient = (upstream) => [upstream * s * (-s + 1)];
    }
    return s;
  }
}

/// Softplus activation function.
///
/// `softplus(x) = log(exp(x) + 1)`
///
/// Softplus can be seen as a smooth approx. of ReLU, and it always returns [Tensor] with positive elements.
///
/// Also accessible via [Activations.softplus].
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = Softplus()(x);
/// print(y); // [0.006715, 0.693147, 5.006715]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.softplus),
///   ...
/// ], ...);
/// ```
class Softplus extends Activation {
  @override
  List<Tensor> Function(Tensor)? gradient;

  @override
  Tensor call(Tensor x, {bool training = false}) {
    if (training) {
      gradient = (upstream) => [upstream * sigmoid(x)];
    }
    return softplus(x);
  }
}

/// Softminus activation function.
///
/// `softminus(x) = x - softmax(x)`
///
/// Also accessible via [Activations.softminus].
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = Softminus()(x);
/// print(y); // [-5.006715, -0.693147, -0.00671]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.softminus),
///   ...
/// ], ...);
/// ```
class Softminus extends Activation {
  @override
  List<Tensor> Function(Tensor)? gradient;

  @override
  Tensor call(Tensor x, {bool training = false}) {
    if (training) {
      gradient = (upstream) => [upstream * (-sigmoid(x) + 1)];
    }
    return softminus(x);
  }
}

/// Hyperbolic tangent activation function.
///
/// `tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))`
///
/// Tanh always returns values from range [-1, 1].
///
/// Also accessible via [Activations.tanh].
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = Tanh()(x);
/// print(y); // [-0.999, 0.0, 0.999]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.tanh),
///   ...
/// ], ...);
/// ```
class Tanh extends Activation {
  @override
  List<Tensor> Function(Tensor)? gradient;

  @override
  Tensor call(Tensor x, {bool training = false}) {
    if (training) {
      gradient = (upstream) => [upstream * (square(sech(x)))];
    }
    return tanh(x);
  }
}

/// Softmax activation function.
///
/// `softmax(x) = exp(x) / reduceSum(exp(x), axis: -1)`
///
/// Softmax transforms a vector of values (logits) into a probability distribution.
/// Values of resulting Tensor are in range (0, 1) and sum up to 1.
///
/// The Softmax activation cannot work with 1D Tensors.
/// The axis along which compute activation is fixed to -1.
///
/// Also accessible via [Activations.softmax].
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = Softmax()(x);
/// print(y); // [[0.09, 0.244728, 0.665241]]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   ...,
///   Dense(n, activation: Activations.softmax)
/// ], ...);
/// ```
class Softmax extends Activation {
  @override
  List<Tensor> Function(Tensor)? gradient;

  /// Applies softmax activation to the input logits [x].
  ///
  /// If [training] is true will construct the [gradient] function.
  ///
  /// Returns the [Tensor] of the same shape as [x].
  ///
  /// If [x.rank] < 2 will throw an [ArgumentError].
  @override
  Tensor call(Tensor x, {bool training = false}) {
    if (x.rank < 2) {
      throw ArgumentError('Softmax cannot works with 1D Tensors, but received x.shape: ${x.shape}');
    }
    final softmax_ = softmax(x);
    if (training) {
      gradient = (upstream) {
        final jacobian =
            expandDims(softmax_, -1) *
            matrixTranspose(
              Tensor.eye(
                    softmax_.shape[-1],
                    batchShape: softmax_.rank > 1 ? softmax_.shape.list.sublist(0, softmax_.rank - 1) : null,
                  ) -
                  expandDims(softmax_, -1),
            );
        return [reshape(matmul(jacobian, expandDims(upstream, -1)), upstream.shape.list)];
      };
    }
    return softmax_;
  }
}

/// [Swish](https://arxiv.org/abs/1710.05941) activation function.
///
/// `swish(x) = x*sigmoid(x)`
///
/// Swish is an alternative to a ReLU activations, that can propagate negative values.
/// This implementation of swish has predefined beta = 1.
///
/// Also accessible via [Activations.swish]
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = Swish()(x);
/// print(y); // [-0.033464, 0.0, 4.966536]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.swish),
///   ...
/// ], ...);
/// ```
class Swish extends Activation {
  @override
  List<Tensor> Function(Tensor p1)? gradient;

  @override
  Tensor call(Tensor x, {bool training = false}) {
    final s = sigmoid(x);
    final swish = x * s;
    if (training) {
      gradient = (upstream) => [upstream * (swish + s * (-swish + 1))];
    }
    return swish;
  }
}

/// Rectified Linear Unit (ReLU) activation function.
///
/// `relu(x) = max(x ,0)`
///
/// Also accessible via [Activations.relu]
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = ReLU()(x);
/// print(y); // [0.0, 0.0, 5.0]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.relu),
///   ...
/// ], ...);
/// ```
class ReLU extends Activation {
  @override
  List<Tensor> Function(Tensor x)? gradient;

  @override
  Tensor call(Tensor x, {bool training = false}) {
    if (training) {
      gradient = (upstream) => [upstream * reluGrad(x)];
    }
    return relu(x);
  }
}

/// Leaky ReLU activation function.
///
/// `LeakyReLU(x) = max(x, alpha*x)`, where `alpha` is a small positive number.
///
/// The default value of [alpha] is 0.1.
///
/// Also accessible via [Activations.leakyReLU]
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = LeakyReLU()(x);
/// print(y); // [-0.5, 0.0, 5.0]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.leakyReLU),
///   ...
/// ], ...);
/// ```
class LeakyReLU extends Activation {
  /// Small positive parameter of LeakyReLU function.
  late final double alpha;

  /// Creates LeakyReLU activation function with given [alpha] value.
  ///
  /// Throws an [ArgumentError] if [alpha] is negative.
  LeakyReLU({this.alpha = 0.1}) {
    if (alpha <= 0) {
      throw ArgumentError('LeakyReLU activation expected alpha to be positive, but received alpha: $alpha');
    }
  }

  @override
  List<Tensor> Function(Tensor x)? gradient;

  @override
  Tensor call(Tensor x, {bool training = false}) {
    if (training) {
      gradient = (upstream) => [upstream * leakyReLUGrad(x, alpha)];
    }
    return leakyReLU(x, alpha);
  }

  @override
  String toString() => "LeakyReLU(alpha: $alpha)";

  @override
  Map<String, dynamic> toJson() => {'type': 'LeakyReLU', 'alpha': alpha};
}

/// Exponential Linear Unit (ELU) activation function.
///
/// `elu(x) = x if x > 0, and alpha * (exp(x) - 1) if x <= 0`, where `alpha` is a small positive number.
///
/// The default value of [alpha] is 0.1.
///
/// Also accessible via [Activations.elu]
///
/// Example of using as a function:
/// ```dart
/// final x = Tensor.constant([-5, 0, 5], dType: DType.float32);
/// final y = ELU()(x);
/// print(y); // [-0.099, 0.0, 5.0]
/// ```
///
/// Example as activation in a layer:
/// ```dart
/// Model model = Model([
///   Dense(1, activation: Activations.elu),
///   ...
/// ], ...);
/// ```
class ELU extends Activation {
  /// Small positive parameter of LeakyReLU function.
  late final double alpha;

  /// Creates ELU activation function with given [alpha] value.
  ///
  /// Throws an [ArgumentError] if [alpha] is negative.
  ELU({this.alpha = 0.1}) {
    if (alpha <= 0) {
      throw ArgumentError('ELU activation expected alpha to be positive, but received alpha: $alpha');
    }
  }

  @override
  List<Tensor> Function(Tensor)? gradient = (upstream) => [upstream * Tensor.ones(upstream.shape.list)];

  @override
  Tensor call(Tensor x, {bool training = false}) {
    if (training) {
      gradient = (upstream) => [upstream * eluGrad(x, alpha)];
    }
    return elu(x, alpha);
  }

  @override
  String toString() => "ELU(alpha: $alpha)";

  @override
  Map<String, dynamic> toJson() => {'type': 'ELU', 'alpha': alpha};
}

/// Collection of possible activation function.
class Activations {
  /// [Linear] activation function.
  static Activation get linear => Linear();

  /// [Softplus] activation function.
  static Activation get softplus => Softplus();

  /// [Softminus] activation function.
  static Activation get softminus => Softminus();

  /// [Sigmoid] activation function.
  static Activation get sigmoid => Sigmoid();

  /// Rectified Linear Unit ([ReLU]) activation function.
  static Activation get relu => ReLU();

  /// [LeakyReLU] activation function.
  static Activation get leakyReLU => LeakyReLU();

  /// Hyperbolic tangent ([Tanh]) activation function.
  static Activation get tanh => Tanh();

  /// [Swish] activation function.
  static Activation get swish => Swish();

  /// [Softmax] activation function.
  static Activation get softmax => Softmax();

  /// Exponential Linear Unit ([ELU]) activation function.
  static Activation get elu => ELU();
}
