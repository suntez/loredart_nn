import 'package:loredart_tensor/loredart_tensor.dart';
import 'dart:math' as math;

/// Applies the ReLU activation function to the elements of [x].
///
/// Throws an [ArgumentError] if the input tensor [x] is not a [NumericTensor].
///
/// Returns a [Tensor] with the same shape as [x] and float-based [DType].
///
/// Note: The output type is [DType.float64] is [x.dType] is float64 and [DType.float32] otherwise.
Tensor relu(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
    List buffer = emptyBuffer(dType, x.shape.size);
    for (int i = 0; i < x.shape.size; i += 1) {
      buffer[i] = math.max<num>(0.0, x.buffer[i]);
    }
    return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError('Expected NumericTensor, but got ${x.runtimeType}', 'x');
  }
}

/// Applies the LeakyReLU activation function (with [alpha]) to the elements of [x].
///
/// Throws an [ArgumentError] if the input tensor [x] is not a [NumericTensor].
///
/// Returns a [Tensor] with the same shape as [x] and float-based [DType].
///
/// Note: The output type is [DType.float64] is [x.dType] is float64 and [DType.float32] otherwise.
Tensor leakyReLU(Tensor x, double alpha) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
    List buffer = emptyBuffer(dType, x.shape.size);
    for (int i = 0; i < x.shape.size; i += 1) {
      buffer[i] = x.buffer[i] > 0 ? x.buffer[i] : x.buffer[i] * alpha;
    }
    return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError('Expected NumericTensor, but got ${x.runtimeType}', 'x');
  }
}

/// Applies the ELU activation function (with [alpha]) to the elements of [x].
///
/// Throws an [ArgumentError] if the input tensor [x] is not a [NumericTensor].
///
/// Returns a [Tensor] with the same shape as [x] and float-based [DType].
///
/// Note: The output type is [DType.float64] is [x.dType] is float64 and [DType.float32] otherwise.
Tensor elu(Tensor x, double alpha) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
    List buffer = emptyBuffer(dType, x.shape.size);
    for (int i = 0; i < x.shape.size; i += 1) {
      buffer[i] = x.buffer[i] > 0 ? x.buffer[i] : (math.exp(x.buffer[i]) - 1) * alpha;
    }
    return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError('Expected NumericTensor, but got ${x.runtimeType}', 'x');
  }
}

/// Applies the softmax activation function to the [x] along [axis].
///
/// The [stable] value indicates whether use numerically stable implementation of the softmax.
///
/// Returns a [Tensor] with the same shape as [x].
Tensor softmax(Tensor x, {int axis = -1, bool? stable}) {
  if (stable != null && stable) {
    x = x - reduceMax(x, axis: [axis], keepDims: true);
  }
  final exponents = exp(x);
  return exponents / reduceSum(exponents, axis: [axis], keepDims: true);
}
