import 'dart:math' as math;

import 'package:loredart_nn/loredart_nn.dart';

import 'math_utils.dart';

/// Activation function class
///
/// ### Activation functions constructors
/// ```dart
/// Activation.sigmoid() /// `Sigmoid` activation function
/// Activation.swish() /// `Swish` activation function
/// Activation.softplus() /// `Softplus` activation function
/// Activation.softmax() /// `Softmax` activation function
/// Activation.relu() /// `ReLU` activation function
/// Activation.leakyReLU() /// `Leaky ReLU` activation function
/// Activation.elu() /// `ELU` activation function
/// ```
class Activation {
  /// The activation function per se
  late final Matrix Function(Matrix, [dynamic param]) function;

  /// The derivative of activation function
  late final List<Matrix> Function(Matrix, [dynamic param]) dfunction;

  /// The name of the activation function
  late final String name;

  /// Sigmoid activation function
  ///
  /// sigmoid(x) = `1/(1+e^-x)`
  ///
  /// Example:
  /// ```dart
  /// final sigmoid = Activation.sigmoid();
  /// final x = Matrix.row([-1, 0, 1]);
  /// final y = sigmoid.function(x);
  /// print(y); // output: matrix 1⨯3 [[0.2689414213699951, 0.5, 0.7310585786300049]]
  /// ```
  Activation.sigmoid() {
    function = (Matrix m, [dynamic param]) =>
        m.apply((double x) => 1 / (1 + math.exp(-x)));
    dfunction = (Matrix m, [dynamic param]) {
      final sigmMatrix = function(m);
      return [sigmMatrix % (-sigmMatrix.addedScalar(-1))];
    };
    name = 'sigmoid';
  }

  /// Swish activation function with beta = 1
  ///
  /// swish(x) = `x*sigmoid(x)`
  ///
  /// In `loredart` [swish] is equivalent to [SiLU]
  ///
  /// Example:
  /// ```dart
  /// final swish = Activation.swish();
  /// final x = Matrix.row([-1, 0, 1]);
  /// final y = swish.function(x);
  /// print(y); // output: matrix 1⨯3 [[-0.2689414213699951, 0.0, 0.7310585786300049]]
  /// ```
  Activation.swish() {
    function = (Matrix m, [dynamic param]) =>
        m.apply((double x) => x / (1 + math.exp(-x)));
    dfunction = (Matrix m, [dynamic param]) {
      final swishMatrix = function(m);
      return [swishMatrix +
          (m.apply((double x) => 1 / (1 + math.exp(-x))) %
              (-swishMatrix.addedScalar(-1)))];
    };
    name = 'swish';
  }

  /// Softplus activation function
  ///
  /// softplus(x) = `ln(e^x+1)`
  ///
  /// Softplus is an antiderivative of sigmoid
  ///
  /// Example:
  /// ```dart
  /// final softplus = Activation.softplus();
  /// final x = Matrix.row([-1, 0, 1]);
  /// final y = softplus.function(x);
  /// print(y); // output: matrix 1⨯3 [[0.31326168751822286, 0.6931471805599453, 1.3132616875182228]]
  /// ```
  Activation.softplus() {
    function = (Matrix m, [dynamic param]) =>
        m.apply((double x) => math.log(math.exp(x) + 1));
    dfunction = (Matrix m, [dynamic param]) =>
        [m.apply((double x) => 1 / (1 + math.exp(-x)))];
    name = 'softplus';
  }

  /// Tanh activation function
  ///
  /// tanh(x) = `sinh(x)/cosh(x) = (e^x-e^-x)/(e^x+e^-x)`
  ///
  /// Example:
  /// ```dart
  /// final tanh = Activation.tanh();
  /// final x = Matrix.row([-1, 0, 1]);
  /// final y = tanh.function(x);
  /// print(y); // output: matrix 1⨯3 [[-0.7615941559557649, 0.0, 0.7615941559557649]]
  /// ```
  Activation.tanh() {
    function = (Matrix m, [dynamic param]) => m.apply((double x) => tanh(x));
    dfunction = (Matrix m, [dynamic param]) =>
        [m.apply((double x) => 1 / math.pow(cosh(x), 2))];
    name = 'tanh';
  }

  /// Softmax activation function
  ///
  /// softmax(x) = `{e^x_i/sum(e^x) | i = 1,n}`
  ///
  /// Softmax calculate 'probabilities' from input vector x
  ///
  /// Softmax is a vector to vector function, that's why it`s derivative is a Jacobi matrix
  ///
  /// Example:
  /// ```dart
  /// final softmax = Activation.softmax();
  /// final x = Matrix.row([-1, 0, 1]);
  /// final y = softmax.function(x);
  /// print(y); // output: matrix 1⨯3 [[0.09003057317038045, 0.2447284710547976, 0.6652409557748218]]
  /// ```
  Activation.softmax() {
    function = (Matrix m, [dynamic param]) {
      Matrix resultMatrix = Matrix.zero(n: 0, m: 0);
      for (int i = 0; i < m.m; i += 1) {
        Matrix exps = m.getColumn(i);
        exps = exps.addedScalar(max(exps)).apply(math.exp);
        if (i == 0) {
          resultMatrix = exps.scaled(1/exps.flattenList().reduce((value, element) => value + element));
        }
        else {
          resultMatrix = MatrixOperation.columnBind(resultMatrix, exps.scaled(1/exps.flattenList().reduce((value, element) => value + element)));
        }
      }
      return resultMatrix;
    };
    dfunction = (Matrix m, [dynamic param]) {
      final softMatrix = function(m);
      // Jacobian for each (mini)batch sample
      return List<Matrix>.generate(
        m.m,
        (index) => Matrix.diag(diag: softMatrix.getColumn(index).flattenList())
        - softMatrix.getColumn(index) * softMatrix.getColumn(index).T
      );
    };
    name = 'softmax';
  }

  /// Linear activation function
  ///
  /// linear(x) = `x`
  ///
  /// Example:
  /// ```dart
  /// final linear = Activation.linear();
  /// final x = Matrix.row([-1, 0, 1]);
  /// final y = linear.function(x);
  /// print(y); // output: matrix 1⨯3 [[-1.0, 0.0, 1.0]]
  /// ```
  Activation.linear() {
    function = (Matrix m, [dynamic param]) => m.apply((double x) => x);
    dfunction = (Matrix m, [dynamic param]) =>
        [Matrix.zero(n: m.n, m: m.m).addedScalar(1)];
    name = 'linear';
  }

  /// ReLU (Rectified Linear Unit) activation function
  ///
  /// relu(x) = `max(0,x)`
  ///
  /// Example:
  /// ```dart
  /// final relu = Activation.relu();
  /// final x = Matrix.row([-1, 1]);
  /// final y = relu.function(x);
  /// print(y); // output: matrix 1⨯2 [[0.0, 1.0]]
  /// ```
  Activation.relu() {
    function =
        (Matrix m, [dynamic param]) => m.apply((double x) => math.max(x, 0));
    dfunction =
        (Matrix m, [dynamic param]) => [m.apply((double x) => x > 0 ? 1 : 0)];
    name = 'relu';
  }

  /// Leaky ReLU activation function
  ///
  /// leakyRelu(x) = `max(0,x)`
  ///
  /// Example:
  /// ```dart
  /// final leakyRelu = Activation.leakyReLU();
  /// final x = Matrix.row([-1, 1]);
  /// final y = leakyRelu.function(x);
  /// final y2 = leakyRelu.function(x, 0.3);
  /// print(y); // output: matrix 1⨯2 [[-0.1, 1.0]]
  /// print(y2); // output: matrix 1⨯2 [[-0.3, 1.0]]
  /// ```
  Activation.leakyReLU() {
    function = (Matrix m, [dynamic alpha = 0.1]) =>
        m.apply((double x) => x > 0 ? x : x * (alpha as double));
    dfunction = (Matrix m, [dynamic alpha = 0.1]) =>
        [m.apply((double x) => x > 0 ? 1 : (alpha as double))];
    name = 'leaky_relu';
  }

  /// ELU (Exponential Linear Unit) activation function
  ///
  /// elu(x) = `x if x > 0 else a(e^x - 1)`
  ///
  /// Example:
  /// ```dart
  /// final elu = Activation.elu();
  /// final x = Matrix.row([-1, 1]);
  /// final y = elu.function(x);
  /// final y2 = elu.function(x, 0.3);
  /// print(y); // output: matrix 1⨯2 [[-0.06321205588285576, 1.0]]
  /// print(y2); // output: matrix 1⨯2 [[-0.1896361676485673, 1.0]]
  /// ```
  Activation.elu() {
    function = (Matrix m, [dynamic alpha = 0.1]) => m
        .apply((double x) => x > 0 ? x : (alpha as double) * (math.exp(x) - 1));
    dfunction = (Matrix m, [dynamic alpha = 0.1]) =>
        [m.apply((double x) => x > 0 ? 1 : (alpha as double) * math.exp(x))];
    name = 'elu';
  }

  @override
  String toString() => name;
}
