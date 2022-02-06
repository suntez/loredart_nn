import 'dart:math' as math;
import 'matrix.dart';
/// return hyperbolic sin of x evaluated as (e^x - e^-x)/2
double sinh(double x) => (math.exp(x) - math.exp(-x)) / 2;
/// return hyperbolic cos of x evaluated as (e^x + e^-x)/2
double cosh(double x) => (math.exp(x) + math.exp(-x)) / 2;
/// return hyperbolic tan of x evaluated as sinh(x)/cosh(x)
double tanh(double x) => sinh(x) / cosh(x);

/// Find min and max element of the [Matrix]
List<double> range(Matrix matrix) {
  double max = matrix[0][0];
  double min = matrix[0][0];
  for (int i = 0; i < matrix.n; i += 1) {
    for (int j = 0; j < matrix.m; j += 1) {
      if (matrix[i][j] > max) {
        max = matrix[i][j];
      }
      else if (matrix[i][j] < min) {
        min = matrix[i][j];
      }
    }
  }
  return [min, max];
}
/// Find max element of the [Matrix]
double max(Matrix matrix) {
  double max = matrix[0][0];
  for (int i = 0; i < matrix.n; i += 1) {
    for (int j = 0; j < matrix.m; j += 1) {
      if (matrix[i][j] > max) {
        max = matrix[i][j];
      }
    }
  }
  return max;
}

/// Find min element of the [Matrix]
double min(Matrix matrix) {
  double min = matrix[0][0];
  for (int i = 0; i < matrix.n; i += 1) {
    for (int j = 0; j < matrix.m; j += 1) {
      if (matrix[i][j] < min) {
        min = matrix[i][j];
      }
    }
  }
  return min;
}