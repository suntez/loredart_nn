import 'dart:math' as math;

import 'package:loredart_tensor/loredart_tensor.dart';

import 'convolution.dart' show computePaddingConv2D, computeOutputShapeConv2D, Padding;

/// Supported types of pooling
enum PoolingType { max, avg }

/// Computes a 2D pooling of given 4D [input] with 2D window of size [kSize].
///
/// With given [kSize], [strides] and [padding] this function performs a pooling operation on [input] images,
/// producing Tensor of the same dType as [input].
///
/// The [input] Tensor must be of the rank 4 with shape structure as `[batchSize, height, width, channels]`, otherwise will throw an ArgumentError.
/// (This function only supports 'channels_last' format of images.)
///
/// The [kSize] must be of the length 2 with shape structure `[kernelHeight, kernelWidth]` or will throw an ArgumentError.
///
/// The [type] controls which pooling method to use: max or averaging.
///
/// Using [padding] one can control the output shape of the operation.
/// [Padding.valid] means no padding will be used, while with [Padding.same] the [input] will be padded with zeros.
/// Unlike the convolution, windows that extend outside of the input are not included in the output.
/// The padding policy is derived from [notes](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding).
///
/// The [strides] must be a 4-element list with first and last elements being 1, like [1, sh, sw, 1], where
/// the [sw] and [sh] are the strides of the pooling window in width and height dimensions.
Tensor pool2D(
  Tensor input,
  List<int> kSize, {
  PoolingType type = PoolingType.max,
  required List<int> strides,
  required Padding padding,
}) {
  if (input.rank != 4) {
    throw ArgumentError(
      "Input must be of rank 4 with shape [batchSize, height, width, channels], but received rank ${input.rank}",
      'input',
    );
  }
  if (kSize.length != 2) {
    throw ArgumentError("Kernel size must be of length 2, but received kSize: $kSize", 'kSize');
  }
  if (strides.length != 4 || strides[0] != 1 || strides[3] != 1) {
    throw ArgumentError(
      "Strides should be of the length 4, with first and last elements being 1: [1, x, y, 1], but received $strides",
      'strides',
    );
  }

  List<List<int>> explicitPadding = [
    [0, 0],
    [0, 0],
  ];
  if (padding == Padding.same) {
    List sameExplicitPadding =
        (computePaddingConv2D(inputShape: input.shape.list, filterShape: kSize, strides: strides) as NumericTensor)
            .buffer;
    explicitPadding[0] = [sameExplicitPadding[2], sameExplicitPadding[3]];
    explicitPadding[1] = [sameExplicitPadding[4], sameExplicitPadding[5]];
  }
  List<int> outputShape = computeOutputShapeConv2D(
    inputShape: [
      input.shape[0],
      input.shape[1] + explicitPadding[0][0] + explicitPadding[0][1],
      input.shape[2] + explicitPadding[1][0] + explicitPadding[1][1],
      input.shape[-1],
    ],
    filterShape: [...kSize, input.shape[-1], input.shape[-1]],
    strides: strides,
  );

  List buffer = emptyBuffer(input.dType, outputShape.reduce((e1, e2) => e1 * e2));

  int size = outputShape[1] * outputShape[2] * outputShape[3];
  for (int i = 0; i < outputShape[1]; i += 1) {
    for (int j = 0; j < outputShape[2]; j += 1) {
      int index = math.max(0, i * strides[1] - explicitPadding[0][0]);
      int indexEnd = math.min(i * strides[1] + kSize[0] - explicitPadding[0][0], input.shape[1]);

      int jndex = math.max(0, j * strides[2] - explicitPadding[1][0]);
      int jndexEnd = math.min(j * strides[2] + kSize[1] - explicitPadding[1][0], input.shape[2]);

      Tensor patch = slice(input, [0, index, jndex, 0], [input.shape[0], indexEnd, jndexEnd, input.shape[-1]]);

      NumericTensor pooledValue =
          (type == PoolingType.max ? reduceMax(patch, axis: [1, 2]) : reduceMean(patch, axis: [1, 2])) as NumericTensor;

      // filling the output buffer according to BHWC
      for (int b = 0; b < outputShape[0]; b += 1) {
        for (int c = 0; c < outputShape[3]; c += 1) {
          buffer[b * size + (i * outputShape[2] + j) * outputShape[3] + c] = pooledValue.buffer[b * outputShape[3] + c];
        }
      }
    }
  }

  return Tensor.fromTypedDataList(buffer, outputShape, dType: input.dType);
}

/// Performs 2D max pooling of 4D [input] across spatial dims with window of size [kSize].
///
/// See [pool2D] function for parameters interpretation.
///
/// See [MaxPool2D] layer for using max pooling in the neural network.
///
/// Example:
/// ```dart
/// final input = uniform([1, 16, 16, 3]);
///
/// final y = maxPool2D(input, [2,2]);
/// print(y.shape); // [1, 15, 15, 32]
///
/// final t = maxPool2D(input, [2,2], padding: Padding.same);
/// print(t.shape); // [1, 16, 16, 3]
///
/// final v = maxPool2D(input, [2,2], strides: [1, 2, 2, 1]);
/// print(v.shape); // [1, 8, 8, 3]
/// ```
Tensor maxPool2D(
  Tensor input,
  List<int> kSize, {
  List<int> strides = const [1, 1, 1, 1],
  Padding padding = Padding.valid,
}) {
  return pool2D(input, kSize, type: PoolingType.max, strides: strides, padding: padding);
}

/// Performs 2D averaging pooling of 4D [input] across spatial dims with window of size [kSize].
///
/// See [pool2D] function for parameters interpretation.
///
/// See [AvgPool2D] layer for using averaging pooling in the neural network.
///
/// Example:
/// ```dart
/// final input = uniform([1, 16, 16, 3]);
///
/// final y = avgPool2D(input, [2,2]);
/// print(y.shape); // [1, 15, 15, 32]
///
/// final t = avgPool2D(input, [2,2], padding: Padding.same);
/// print(t.shape); // [1, 16, 16, 3]
///
/// final v = avgPool2D(input, [2,2], strides: [1, 2, 2, 1]);
/// print(v.shape); // [1, 8, 8, 3]
/// ```
Tensor avgPool2D(
  Tensor input,
  List<int> kSize, {
  List<int> strides = const [1, 1, 1, 1],
  Padding padding = Padding.valid,
}) {
  return pool2D(input, kSize, type: PoolingType.avg, strides: strides, padding: padding);
}

/// Computes a 1D pooling of given 3D [input] with 1D window of size [kSize].
///
/// This function reshapes input Tensors and performs [pool2d] on them.
///
/// With given [kSize], [strides] and [padding] this function performs a pooling operation on [input] series,
/// producing Tensor of the same dType as [input].
///
/// The [input] Tensor must be of the rank 3 with shape structure as `[batchSize, width, channels]`, otherwise will throw an ArgumentError.
/// (This function only supports 'channels_last' format of images.)
///
/// The [type] controls which pooling method to use: max or averaging.
///
/// Using [padding] one can control the output shape of the operation.
/// [Padding.valid] means no padding will be used, while with [Padding.same] the [input] will be padded with zeros.
/// Unlike the convolution, windows that extend outside of the input are not included in the output.
/// The padding policy is derived from [notes](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding).
///
/// The [strides] must be a 3-element list with first and last elements being 1, like [1, sw, 1], where
/// [sw] is the strides of the pooling window in width dimension.
Tensor pool1D(
  Tensor input,
  int kSize, {
  PoolingType type = PoolingType.max,
  required List<int> strides,
  required Padding padding,
}) {
  input = expandDims(input, 1);
  strides = [strides[0], 1, strides[1], strides[2]];
  Tensor pooledResult = pool2D(input, [1, kSize], type: type, strides: strides, padding: padding);
  return reshape(pooledResult, [pooledResult.shape[0], pooledResult.shape[2], pooledResult.shape[3]]);
}

/// Performs 1D max pooling of 3D [input] across spatial dim with window of size [kSize].
///
/// See [pool1D] function for parameters interpretation.
///
/// See [MaxPool1D] layer for using max pooling in the neural network.
///
/// Example:
/// ```dart
/// final input = uniform([1, 16, 3]);
///
/// final y = maxPool1D(input, [2,2]);
/// print(y.shape); // [1, 15, 32]
///
/// final t = maxPool1D(input, [2,2], padding: Padding.same);
/// print(t.shape); // [1, 16, 3]
///
/// final v = maxPool1D(input, [2,2], strides: [1, 2, 1]);
/// print(v.shape); // [1, 8, 3]
/// ```
Tensor maxPool1D(Tensor input, int kSize, {List<int> strides = const [1, 1, 1], Padding padding = Padding.valid}) {
  return pool1D(input, kSize, type: PoolingType.max, strides: strides, padding: padding);
}

/// Performs 1D averaging pooling of 3D [input] across spatial dim with window of size [kSize].
///
/// See [pool1D] function for parameters interpretation.
///
/// See [AvgPool1D] layer for using averaging pooling in the neural network.
///
/// Example:
/// ```dart
/// final input = uniform([1, 16, 3]);
///
/// final y = avgPool1D(input, [2,2]);
/// print(y.shape); // [1, 15, 32]
///
/// final t = avgPool1D(input, [2,2], padding: Padding.same);
/// print(t.shape); // [1, 16, 3]
///
/// final v = avgPool1D(input, [2,2], strides: [1, 2, 1]);
/// print(v.shape); // [1, 8, 3]
/// ```
Tensor avgPool1D(Tensor input, int kSize, {List<int> strides = const [1, 1, 1], Padding padding = Padding.valid}) {
  return pool1D(input, kSize, type: PoolingType.avg, strides: strides, padding: padding);
}

/// Computes the gradients of 2D max pooling w.r.t. the input.
Tensor maxPool2DBackprop(
  Tensor input,
  List<int> kSize,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  List<List<int>> explicitPadding = [
    [0, 0],
    [0, 0],
  ];
  if (padding == Padding.same) {
    List sameExplicitPadding =
        (computePaddingConv2D(inputShape: input.shape.list, filterShape: kSize, strides: strides) as NumericTensor)
            .buffer;
    explicitPadding[0] = [sameExplicitPadding[2], sameExplicitPadding[3]];
    explicitPadding[1] = [sameExplicitPadding[4], sameExplicitPadding[5]];
  }
  List<int> outputShape = computeOutputShapeConv2D(
    inputShape: [
      input.shape[0],
      input.shape[1] + explicitPadding[0][0] + explicitPadding[0][1],
      input.shape[2] + explicitPadding[1][0] + explicitPadding[1][1],
      input.shape[-1],
    ],
    filterShape: [...kSize, input.shape[-1], input.shape[-1]],
    strides: strides,
  );

  List buffer = emptyBuffer(outputGrad.dType, input.shape.size);
  outputGrad as NumericTensor;

  int size = input.shape[1] * input.shape[2] * input.shape[3];

  for (int i = 0; i < outputShape[1]; i += 1) {
    for (int j = 0; j < outputShape[2]; j += 1) {
      int index = math.max(0, i * strides[1] - explicitPadding[0][0]);
      int indexEnd = math.min(i * strides[1] + kSize[0] - explicitPadding[0][0], input.shape[1]);

      int jndex = math.max(0, j * strides[2] - explicitPadding[1][0]);
      int jndexEnd = math.min(j * strides[2] + kSize[1] - explicitPadding[1][0], input.shape[2]);

      List indices =
          (reduceLocalArgMax(
                    slice(input, [0, index, jndex, 0], [input.shape[0], indexEnd, jndexEnd, input.shape[-1]]),
                    axis: [1, 2],
                  )
                  as NumericTensor)
              .buffer; // global indices of a patch, but not of the input

      for (int b = 0; b < input.shape[0]; b += 1) {
        for (int c = 0; c < input.shape[-1]; c += 1) {
          int tensorPositionX =
              indices[b * input.shape[-1] + c] ~/ (jndexEnd - jndex) + index; // global index of patch arg max
          int tensorPositionY =
              indices[b * input.shape[-1] + c] % (jndexEnd - jndex) + jndex; // global jndex of patch arg max
          buffer[b * size + (tensorPositionX * input.shape[2] + tensorPositionY) * input.shape[-1] + c] +=
              outputGrad.buffer[((b * outputShape[1] + i) * outputShape[2] + j) * outputShape[3] + c];
        }
      }
    }
  }
  return Tensor.fromTypedDataList(buffer, input.shape.list, dType: outputGrad.dType);
}

/// Computes the gradients of 2D avg pooling w.r.t. the input.
Tensor avgPool2DBackprop(
  TensorShape inputShape,
  List<int> kSize,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  List<List<int>> explicitPadding = [
    [0, 0],
    [0, 0],
  ];
  if (padding == Padding.same) {
    List sameExplicitPadding =
        (computePaddingConv2D(inputShape: inputShape.list, filterShape: kSize, strides: strides) as NumericTensor)
            .buffer;
    explicitPadding[0] = [sameExplicitPadding[2], sameExplicitPadding[3]];
    explicitPadding[1] = [sameExplicitPadding[4], sameExplicitPadding[5]];
  }
  List<int> outputShape = computeOutputShapeConv2D(
    inputShape: [
      inputShape[0],
      inputShape[1] + explicitPadding[0][0] + explicitPadding[0][1],
      inputShape[2] + explicitPadding[1][0] + explicitPadding[1][1],
      inputShape[-1],
    ],
    filterShape: [...kSize, inputShape[-1], inputShape[-1]],
    strides: strides,
  );

  List buffer = emptyBuffer(outputGrad.dType, inputShape.size);
  outputGrad as NumericTensor;

  int size = inputShape[1] * inputShape[2] * inputShape[3];
  for (int i = 0; i < outputShape[1]; i += 1) {
    for (int j = 0; j < outputShape[2]; j += 1) {
      int index = math.max(0, i * strides[1] - explicitPadding[0][0]);
      int indexEnd = math.min(i * strides[1] + kSize[0] - explicitPadding[0][0], inputShape[1]);

      int jndex = math.max(0, j * strides[2] - explicitPadding[1][0]);
      int jndexEnd = math.min(j * strides[2] + kSize[1] - explicitPadding[1][0], inputShape[2]);

      double meanPerPixel = 1 / (indexEnd - index) / (jndexEnd - jndex);
      for (int b = 0; b < inputShape[0]; b += 1) {
        for (int c = 0; c < inputShape[-1]; c += 1) {
          for (int ki = index; ki < indexEnd; ki += 1) {
            for (int kj = jndex; kj < jndexEnd; kj += 1) {
              buffer[b * size + (ki * inputShape[2] + kj) * inputShape[-1] + c] +=
                  meanPerPixel *
                  outputGrad.buffer[((b * outputShape[1] + i) * outputShape[2] + j) * outputShape[3] + c];
            }
          }
        }
      }
    }
  }
  return Tensor.fromTypedDataList(buffer, inputShape.list, dType: outputGrad.dType);
}

/// Computes the gradients of 1D max pooling w.r.t. the input.
Tensor maxPool1DBackprop(
  Tensor input,
  int kSize,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  input = expandDims(input, 1);
  outputGrad = expandDims(outputGrad, 1);
  strides = [strides[0], 1, strides[1], strides[2]];
  return squeeze(maxPool2DBackprop(input, [1, kSize], outputGrad, strides: strides, padding: padding), axis: [1]);
}

/// Computes the gradients of 1D avg pooling w.r.t. the input.
Tensor avgPool1DBackprop(
  TensorShape inputShape,
  int kSize,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  outputGrad = expandDims(outputGrad, 1);
  final extInputShape = TensorShape([inputShape[0], 1, inputShape[1], inputShape[2]]);
  return squeeze(
    avgPool2DBackprop(extInputShape, [1, kSize], outputGrad, strides: strides, padding: padding),
    axis: [1],
  );
}
