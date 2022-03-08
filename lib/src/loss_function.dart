import 'dart:math' as math;

import 'matrix.dart';

/// Loss function class
///
/// Compute mean error of a (mini)batch
///
/// ### Loss functions constructors
/// ```dart
/// Loss.mae() /// `Mean Absolute Error`
/// Loss.mse() /// `Mean Square Error`
/// Loss.crossEntropy() /// `Cross Entropy`
/// Loss.sparseCrossEntropy() /// `Sparse Cross Entropy`
/// ```
class Loss {
  /// The loss function
  late double Function(Matrix, Matrix, [dynamic parameter]) function;

  /// The derivative of loss function
  late Matrix Function(Matrix, Matrix, [dynamic parameter]) dfunction;

  /// The name of the loss function
  late String name;

  /// Mean Square Error loss function for batch
  ///
  /// mse(y, yP) = `1/n * sum (y - yP)^2`
  /// Example:
  /// ```dart
  /// final y = Matrix.column([0, 1, 0]);
  /// final yP = Matrix.column([0.1, 0.1, 0.8]);
  ///
  /// final mse = Loss.mse();
  /// double loss = mse.function(y, yP);
  /// print(loss); // output: 0.48666666666666675
  /// ```
  Loss.mse() {
    function = (Matrix y, Matrix yP, [dynamic parameter]) {
      return (y - yP)
              .apply((double x) => math.pow(x, 2).toDouble())
              .reduceSum() /
          y.n /
          y.m;
    };
    dfunction = (Matrix y, Matrix yP, [dynamic parameter]) {
      return (yP - y).scaled(2 / y.n);
    };
    name = 'mse';
  }

  /// Mean Absolute Error loss function for batch
  ///
  /// mse(y, yP) = `1/n * sum |y - yP|`
  /// Example:
  /// ```dart
  /// final y = Matrix.column([0, 1, 0]);
  /// final yP = Matrix.column([0.1, 0.1, 0.8]);
  ///
  /// final mae = Loss.mae();
  /// double loss = mae.function(y, yP);
  /// print(loss); // output: 0.6
  /// ```
  Loss.mae() {
    function = (Matrix y, Matrix yP, [dynamic parameter]) {
      return (y - yP).apply((double x) => x.abs()).reduceSum() / y.n / y.m;
    };
    dfunction = (Matrix y, Matrix yP, [dynamic parameter]) {
      return (yP - y).apply((double x) => x > 0 ? 1 : -1).scaled(1 / y.n);
    };
    name = 'mae';
  }

  /// Cross Entropy loss function for batch
  ///
  /// crossEntropy(y, yP) = `-sum y*ln(yP)` for y being `One-Hot encoded`
  ///
  /// Example:
  /// ```dart
  /// final y = Matrix.column([0, 1, 0]);
  /// final yP = Matrix.column([0.1, 0.1, 0.8]);
  ///
  /// final crossEntropy = Loss.crossEntropy();
  /// double loss = crossEntropy.function(y, yP);
  /// print(loss); // output: 2.3025850929940455
  /// ```
  Loss.crossEntropy() {
    function = (Matrix y, Matrix yP, [dynamic parameter]) {
      return -(y % yP.apply((x) => x != 0 ? math.log(x) : math.log(x + 1e-4)))
              .reduceSum() /
          y.m;
    };
    dfunction = (Matrix y, Matrix yP, [dynamic fromSoftmax = false]) {
      if ((fromSoftmax as bool)) {
        return yP - y;
      }
      return -y % (yP.apply((x) => x != 0 ? 1 / x : 1e4));
    };
    name = 'cross_entropy';
  }

  /// Sparse Cross Entropy loss function for batch
  ///
  /// sparceCrossEntropy(y, yP) = `-sum y*ln(yP)` for y being `labeled`
  ///
  /// Example:
  /// ```dart
  /// final y = Matrix.column([1]); // second label, which is eqv. to [0, 1, 0]
  /// final yP = Matrix.column([0.1, 0.1, 0.8]);
  ///
  /// final sparseCrossEntropy = Loss.sparseCrossEntropy();
  /// double loss = sparseCrossEntropy.function(y, yP);
  /// print(loss); // output: 2.3025850929940455
  /// ```
  Loss.sparseCrossEntropy() {
    function = (Matrix y, Matrix yP, [dynamic parameter]) {
      Matrix categorical = Matrix.zero(n: yP.n, m: yP.m);
      if (y.m == 1) {
        for (int i = 0; i < y.n; i += 1) {
          categorical[y[i][0].toInt()][i] = 1;
        }
      } else {
        for (int i = 0; i < y.m; i += 1) {
          categorical[y[0][i].toInt()][i] = 1;
        }
      }
      return -(categorical %
                  yP.apply((x) => x != 0 ? math.log(x) : math.log(x + 1e-4)))
              .reduceSum() /
          y.m;
    };
    dfunction = (Matrix y, Matrix yP, [dynamic fromSoftmax = false]) {
      Matrix categorical = Matrix.zero(n: yP.n, m: yP.m);
      if (y.m == 1) {
        for (int i = 0; i < y.n; i += 1) {
          categorical[y[i][0].toInt()][i] = 1;
        }
      } else {
        for (int i = 0; i < y.m; i += 1) {
          categorical[y[0][i].toInt()][i] = 1;
        }
      }
      if ((fromSoftmax as bool)) {
        return yP - categorical;
      }
      return -categorical % (yP.apply((x) => x != 0 ? 1 / x : 1e4));
    };
    name = 'sparse_cross_entropy';
  }

  @override
  String toString() => name;
}
