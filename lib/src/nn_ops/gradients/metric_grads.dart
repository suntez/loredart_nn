import 'package:loredart_tensor/loredart_tensor.dart';

/// Computes the gradients of the cross-entropy function w.r.t. [yPred].
Tensor crossEntropyGrad(Tensor yTrue, Tensor yPred, {bool fromLogits = false}) {
  if (fromLogits) {
    return (yPred - yTrue);
  }
  // keras version: sum(true)/sum(pred) - true/pred
  return -((yTrue / yPred) +
      reduceSum(yTrue, axis: [-1], keepDims: true) / reduceSum(yPred, axis: [-1], keepDims: true));
}

/// Computes the gradients of the binary cross-entropy function w.r.t. [yPred].
Tensor binaryCrossentropyGrad(Tensor yTrue, Tensor yPred, {double epsilon = 1e-8, bool fromLogits = false}) {
  if (fromLogits) {
    return (yPred - yTrue) / yPred.shape[-1];
  }
  return (yPred - yTrue) / (yPred * (-yPred + 1)) / yPred.shape[-1];
}

/// Computes the gradients of the log-cosh function w.r.t. [yPred].
Tensor logCoshGrad(Tensor yTrue, Tensor yPred) => tanh(yPred - yTrue);
