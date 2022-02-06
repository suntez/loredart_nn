import 'dart:math' as math;

import 'matrix.dart';

/// Loss function class
/// 
/// ### Loss functions constructors
/// ```dart
/// Loss.mae() /// `Mean Absolute Error` loss
/// Loss.mse() /// `Mean Square Error` loss
/// Loss.crossEntropy() /// `Cross Entropy` loss
/// Loss.sparseCrossEntropy() /// `Sparse Cross Entropy` loss
/// ```
class Loss {
  /// The loss function
  late double Function(Matrix, Matrix, [dynamic parametr]) function;
  /// The derivative of loss function
  late Matrix Function(Matrix, Matrix, [dynamic parametr]) dfunction;
  /// The name of the loss function
  late String name;
  
  /// Mean Square Error loss function
  /// 
  /// mse(y, yP) = `1/n * sum (y - yP)^2`
  /// Example:
  /// ```dart
  /// final y = Matrix.row([0, 1, 0]);
  /// final yP = Matrix.row([0.1, 0.1, 0.8]);
  /// 
  /// final mse = Loss.mse();
  /// double loss = mse.function(y, yP);
  /// print(loss); // output: 0.48666666666666675
  /// ```  
  Loss.mse() {
    function = (Matrix y, Matrix yP, [dynamic parametr]) {
      return (y-yP).apply((double x) => math.pow(x,2).toDouble()).reduceSum() / y.n / y.m;
    };
    dfunction = (Matrix y, Matrix yP, [dynamic parametr]) {
      return (yP-y).scaled(2/y.n/y.m);
    };
    name = 'mse';
  }

  /// Mean Absolute Error loss function
  /// 
  /// mse(y, yP) = `1/n * sum |y - yP|`
  /// Example:
  /// ```dart
  /// final y = Matrix.row([0, 1, 0]);
  /// final yP = Matrix.row([0.1, 0.1, 0.8]);
  /// 
  /// final mae = Loss.mae();
  /// double loss = mae.function(y, yP);
  /// print(loss); // output: 0.6
  /// ```  
  Loss.mae() {
    function = (Matrix y, Matrix yP, [dynamic parametr]) {
      return (y-yP).apply((double x) => x.abs()).reduceSum() / y.n / y.m;
    };
    dfunction = (Matrix y, Matrix yP, [dynamic parametr]) {
      return (yP-y).apply((double x) => x > 0 ? 1 : -1).scaled(y.n / y.m);
    };
    name = 'mae';
  }

  /// Cross Entropy loss function
  /// 
  /// crossEntropy(y, yP) = `-sum y*ln(yP)` for y being `One-Hot endoded`
  /// 
  /// Example:
  /// ```dart
  /// final y = Matrix.row([0, 1, 0]);
  /// final yP = Matrix.row([0.1, 0.1, 0.8]);
  /// 
  /// final crossEntropy = Loss.crossEntropy();
  /// double loss = crossEntropy.function(y, yP);
  /// print(loss); // output: 2.3025850929940455
  /// ```  
  Loss.crossEntropy() {
    function = (Matrix y, Matrix yP, [dynamic parametr]) {
      return -(y%yP.apply((x) => x != 0 ? math.log(x) : math.log(x+1e-5))).reduceSum();
    };
    dfunction = (Matrix y, Matrix yP, [dynamic fromSoftmax = false]) {
      if (y.reduceSum() == 1) {
        if ((fromSoftmax as bool)) {
          return yP-y;
        }
        return -y%(yP.apply((x) => x != 0 ?  1/x : 1/1e-5));
      }
      else {
        throw Exception('y must be One-Hot Encoded and sum(yPredicted) must be equal 1');
      }
    };
    name = 'cross_entropy';
  }

  /// Sparse Cross Entropy loss function
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
    function = (Matrix y, Matrix yP, [dynamic parametr]) {
      if (yP.n == 1 || yP.m == 1) {
        Matrix categorical = Matrix.zero(n: yP.n, m: yP.m);
        if (categorical.m == 1) {
          categorical[y[0][0].toInt()][0] = 1;
          //print('cater $categorical');
        }
        else {
          categorical[0][y[0][0].toInt()] = 1;
        }
        return -(categorical%yP.apply((x) => x != 0 ? math.log(x) : math.log(x+1e-5))).reduceSum();
      }
      else {
        throw Exception('yPredicted must be a vector');
      }
    };
    dfunction = (Matrix y, Matrix yP, [dynamic fromSoftmax = false]) {
      if (yP.n == 1 || yP.m == 1) {
        Matrix categorical = Matrix.zero(n: yP.n, m: yP.m);
        if (categorical.m == 1) {
          categorical[y[0][0].toInt()][0] = 1;
        }
        else {
          categorical[0][y[0][0].toInt()] = 1;
        }
        if ((fromSoftmax as bool)) {
          return yP-categorical;
        }
        return -categorical%(yP.apply((x) => x != 0 ?  1/x : 1/1e-5));
      }
      else {
        throw Exception('yPredicted must be a vector');
      }
    };
    name = 'sparse_cross_entropy';
  }

  @override
  String toString() => name;
}