import 'package:loredart_tensor/loredart_tensor.dart';

import 'other_ops.dart';

/// Computes the binary crossentropy between equal-shaped Tensors [yTrue] and [yPred] along given [axis].
///
/// The [epsilon] value is used for a clipping of the [yPred] Tensor for a safe computation of log.
Tensor binaryCrossentropy(Tensor yTrue, Tensor yPred, {List<int> axis = const [-1], double epsilon = 1e-7}) {
  yPred = clipByValue(yPred, epsilon, 1 - epsilon);
  return reduceMean(-xlogy(yTrue, yPred) + xlogy(yTrue - 1, -yPred + 1), axis: List.from(axis));
}

/// Computes the crossentropy between equal-shaped Tensors [yTrue] and [yPred] along given [axis].
///
/// The [epsilon] value is used for a clipping of the [yPred] Tensor for a safe computation of log.
Tensor crossEntropy(Tensor yTrue, Tensor yPred, {List<int> axis = const [-1], double epsilon = 1e-7}) {
  yPred = clipByValue(yPred, epsilon, 1 - epsilon);
  return reduceSum(-xlogy(yTrue, yPred), axis: List.from(axis));
}

/// Computes the logarithm of the hyperbolic cosine of differences between equal-shaped Tensors [yTrue] and [yPred] along given [axis].
Tensor logCosh(Tensor yTrue, Tensor yPred, {List<int> axis = const [-1]}) {
  return reduceMean(log(cosh(yTrue - yPred)), axis: List.from(axis));
}

/// Computes the Kullback-Leibler divergence between equal-shaped Tensors [yTrue] and [yPred] along given [axis].
///
/// The [epsilon] value is used for a clipping of the [yPred] and [yTrue] Tensors for a safe computation of log and division.
Tensor klDivergence(Tensor yTrue, Tensor yPred, {List<int> axis = const [-1], double epsilon = 1e-7}) {
  yTrue = clipByValue(yTrue, epsilon, 1);
  yPred = clipByValue(yPred, epsilon, 1);
  return reduceSum(yTrue * log(yTrue / yPred), axis: List.from(axis));
}
