import 'package:loredart_tensor/loredart_tensor.dart';
import 'dart:math' as dart_math;

/// Splits input [Tensor] into two (feature and targets) by columns and optionally casts any of those into specific [DType]s.
///
/// Allows to split single Tesnsor (e.g. from csv file) into two, by extracting columns from `targetIndices` as targets.
/// Only supports cases when targets are first or last `m` columns, other slicing will be incorrect.
/// Input [Tensor] should be a matrix, other will raise ArgumentError.
(Tensor, Tensor) splitToFeaturesAndTargets(
  Tensor values, {
  required List<int> targetIndices,
  DType? targetsDType,
  DType? featuresDType,
}) {
  if (values.rank != 2) {
    throw ArgumentError("Expected to see matrix, but received tensor of rank ${values.rank}", 'values');
  }
  final featuresStart =
      targetIndices.contains(0) ? dart_math.min<int>(targetIndices.reduce(dart_math.max) + 1, values.shape[1]) : 0;
  final featuresEnd = targetIndices.contains(0) ? values.shape[1] : targetIndices.reduce(dart_math.min);

  return (
    cast(slice(values, [0, featuresStart], [values.shape[0], featuresEnd]), featuresDType ?? values.dType),
    cast(
      slice(
        values,
        [0, targetIndices[0]],
        [values.shape[0], dart_math.min<int>(targetIndices.reduce(dart_math.max) + 1, values.shape[1])],
      ),
      targetsDType ?? values.dType,
    ),
  );
}
