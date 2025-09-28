import 'dart:math' as math;

import 'package:loredart_tensor/loredart_tensor.dart';

/// Calculates gradients of `matmul` operation w.r.t [operationInputs]
///
/// [upstream] aka grad inputs, that are used for VJP evaluation.
List<Tensor> matmulGrad(
  List<Tensor> operationInputs,
  List<Tensor> operationOutputs,
  List<Tensor> upstream, {
  bool wasTransposedA = false,
  bool wasTransposedB = false,
}) {
  if (!wasTransposedA && !wasTransposedB) {
    return matmulGradEval(
      x1: upstream[0],
      transposeX1: false,
      x2: operationInputs[1], // d(AB)/dA = B
      transposeX2: true,
      y1: operationInputs[0], // d(AB)/dB = A
      transposeY1: true,
      y2: upstream[0],
      transposeY2: false,
    );
  } else {
    return [];
  }
  //TODO: finish matmul grad function
}

/// Evaluates the gradient of the matmul operation with given matrices.
List<Tensor> matmulGradEval({
  required Tensor x1,
  required bool transposeX1,
  required Tensor x2,
  required bool transposeX2,
  required Tensor y1,
  required bool transposeY1,
  required Tensor y2,
  required bool transposeY2,
}) {
  final Tensor dx = matmul(x1, x2, transposeA: transposeX1, transposeB: transposeX2);
  final Tensor dy = matmul(y1, y2, transposeA: transposeY1, transposeB: transposeY2);
  return [dx, dy];
}

/// Computes the gradients of ReLU activation w.r.t. input tensor.
Tensor reluGrad(Tensor x) => leakyReLUGrad(x, 0);

/// Computes the gradients of LeakyReLU activation w.r.t. input tensor.
Tensor leakyReLUGrad(Tensor x, double alpha) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
    List buffer = emptyBuffer(dType, x.shape.size);
    for (int i = 0; i < x.shape.size; i += 1) {
      buffer[i] = x.buffer[i] > 0.0 ? 1.0 : alpha;
    }
    return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError('Expected NumericTensor, but got ${x.runtimeType}', 'x');
  }
}

/// Computes the gradients of ELU activation w.r.t. input tensor.
Tensor eluGrad(Tensor x, double alpha) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
    List buffer = emptyBuffer(dType, x.shape.size);
    for (int i = 0; i < x.shape.size; i += 1) {
      buffer[i] = x.buffer[i] > 0.0 ? 1.0 : alpha * math.exp(x.buffer[i]);
    }
    return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError('Expected NumericTensor, but got ${x.runtimeType}', 'x');
  }
}
