import 'package:loredart_tensor/loredart_tensor.dart';

/// Clips [x] to a specific [minValue] and [maxValue].
///
/// Returns Tensor of the same shape and DType as [x].
///
/// Example:
/// ```dart
/// final x = Tensor.constant([
///   [0.0,  0.5, 1.1],
///   [-0.1, 0.9, 13.0]
/// ]);
/// final y = clipByValue(x, 1e-3, 1.0);
/// print(y);
/// // Tensor(shape: [2, 3], values:
/// //  [[0.001, 0.5, 1.0]
/// //   [0.001, 0.9, 1.0]], dType: float32)
/// ```
Tensor clipByValue(Tensor x, minValue, maxValue) {
  if (x is NumericTensor) {
    List buffer = emptyBuffer(x.dType, x.shape.size);
    for (int i = 0; i < x.shape.size; i += 1) {
      buffer[i] = x.buffer[i] > maxValue ? maxValue : (x.buffer[i] < minValue ? minValue : x.buffer[i]);
    }
    return Tensor.fromTypedDataList(buffer, x.shape.list, dType: x.dType);
  } else {
    throw ArgumentError('Expected NumericTensor, but got ${x.runtimeType}', 'x');
  }
}

/// Computes the dropout of [x] and additionally returns dropout mask.
/// For explanation see [dropout] function.
List<Tensor> dropoutWithMask(Tensor x, double rate, {List<int>? noiseShape, int? seed}) {
  if (rate < 0 || rate >= 1) {
    throw ArgumentError('The rate should de between [0, 1), but received rate: $rate', 'rate');
  }

  if (noiseShape != null && !x.shape.broadcastableWith(TensorShape(noiseShape))) {
    throw ArgumentError('noiseShape: $noiseShape is not broadcastable with x.shape: ${x.shape}', 'noiseShape');
  }
  if (rate == 0) {
    return [x, Tensor.ones(x.shape.list)];
  }
  final resultDType = x.dType == DType.float64 ? DType.float64 : DType.float32;

  Tensor mask = greaterEqual(uniform(noiseShape ?? x.shape.list, seed: seed, dType: resultDType), rate);
  if (mask.dType != x.dType) {
    x = cast(x, resultDType);
  }
  return [x * mask * (1 / (1 - rate)), mask];
}

/// Computes the dropout of [x], i.e., with probability [rate] set each element to zero.
/// Remaining non-dropped elements are scaled by 1/(1-[rate]) to preserve the mean.
///
/// If the [noiseShape] is null, than each element considered independently.
/// But if it's set to a broadcastable (with x.shape) shape, than only dimensions with
/// [noiseShape[i]] == [x.shape[i]] will make independent decisions.
///
/// [rate] should be from an interval [0, 1) and [noiseShape] should be broadcastable with [x.shape]
/// otherwise will throw an [ArgumentError].
///
/// Returns Tensor of the same shape as [x] and float DType.
///
/// Examples:
/// ```dart
/// final x = Tensor.ones([4,4], dType: DType.int32);
/// final y = dropout(x, 0.5);
/// print(y);
/// // Tensor(shape: [4, 4], values:
/// //  [[0.0, 0.0, 2.0, 0.0]
/// //  [2.0, 0.0, 0.0, 2.0]
/// //  [2.0, 0.0, 0.0, 2.0]
/// //  [2.0, 0.0, 0.0, 2.0]], dType: float32)
/// ```
/// Using noise shape to drop whole channels:
/// ```dart
/// final x = Tensor.ones([4,4], dType: DType.int32);
/// final v = dropout(x, 0.5, noiseShape: [1, 4]);
/// print(v);
/// // Tensor(shape: [4, 4], values:
/// //  [[0.0, 2.0, 2.0, 0.0]
/// //  [0.0, 2.0, 2.0, 0.0]
/// //  [0.0, 2.0, 2.0, 0.0]
/// //  [0.0, 2.0, 2.0, 0.0]], dType: float32)
/// ```
Tensor dropout(Tensor x, double rate, {List<int>? noiseShape, int? seed}) {
  return dropoutWithMask(x, rate, noiseShape: noiseShape, seed: seed)[0];
}
