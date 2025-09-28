import 'dart:math' as math;
import 'package:loredart_tensor/loredart_tensor.dart';

/// The type of padding algorithm
enum Padding { same, valid }

/// Computes the shape of the 2D convolution result excluding the batch dim.
///
/// Assumes tha [inputShape] is a 3-element list.
List<int> computeBatchlessOutputShapeConv2D({
  required List<int> inputShape,
  required List<int> filterShape,
  required List<int> strides,
  required Padding padding,
}) {
  inputShape = [0, ...inputShape];
  if (padding == Padding.same) {
    List padValues =
        (computePaddingConv2D(inputShape: inputShape, filterShape: filterShape, strides: strides) as NumericTensor)
            .buffer;
    inputShape[1] += padValues[2] + padValues[3] as int;
    inputShape[2] += padValues[4] + padValues[5] as int;
  }
  return computeOutputShapeConv2D(inputShape: inputShape, filterShape: filterShape, strides: strides).sublist(1);
}

/// Computes the shape of the 1D convolution result excluding the batch dim.
///
/// Assumes that [inputShape] is a 2-element list.
List<int> computeBatchlessOutputShapeConv1D({
  required List<int> inputShape,
  required List<int> filterShape,
  required List<int> strides,
  required Padding padding,
}) {
  List<int> shape2d = computeBatchlessOutputShapeConv2D(
    inputShape: [1, ...inputShape],
    filterShape: [1, ...filterShape],
    strides: [strides[0], 1, strides[1], strides[2]],
    padding: padding,
  );
  assert(shape2d[0] == 1 && shape2d.length == 3);
  return shape2d.sublist(1);
}

/// Computes the shape of the 2D convolution result according to the [note](https://www.tensorflow.org/api_docs/python/tf/nn?version=nightly#notes_on_padding_2).
List<int> computeOutputShapeConv2D({
  required List<int> inputShape,
  required List<int> filterShape,
  required List<int> strides,
}) {
  List<int> outputShape = [inputShape[0], 0, 0, filterShape[3]];
  outputShape[1] = ((inputShape[1] - filterShape[0] + 1) / strides[1]).ceil();
  outputShape[2] = ((inputShape[2] - filterShape[1] + 1) / strides[2]).ceil();
  return outputShape;
}

/// Computes the size of the padding for the 2D convolution operation according to the [note](https://www.tensorflow.org/api_docs/python/tf/nn?version=nightly#notes_on_padding_2).
Tensor computePaddingConv2D({
  required List<int> inputShape,
  required List<int> filterShape,
  required List<int> strides,
}) {
  int heightPadding = math.max(
    0,
    filterShape[0] - (inputShape[1] % strides[1] == 0 ? strides[1] : (inputShape[1] % strides[1])),
  );
  int widthPadding = math.max(
    0,
    filterShape[1] - (inputShape[2] % strides[2] == 0 ? strides[2] : (inputShape[2] % strides[2])),
  );

  return Tensor.constant([
    [0, 0],
    [heightPadding ~/ 2, heightPadding - heightPadding ~/ 2],
    [widthPadding ~/ 2, widthPadding - widthPadding ~/ 2],
    [0, 0],
  ], dType: DType.int32);
}

/// Computes a 2D convolution of given 4D [input] with 4D [filter].
///
/// With given [filters], [strides] and [padding] this function performs a cross-corelation operation on [input] images,
/// producing convolved Tensor of the same dType as [filters] and [input].
///
/// See [Conv2D] layer for using convolution in the neural network.
///
/// The [input] Tensor must be of the rank 4 with shape structure as `[batchSize, height, width, channels]`, otherwise will throw an ArgumentError.
/// (This function only supports 'channels_last' format of images.)
///
/// The [filter] Tensor must be of the rank 4 with shape structure `[filterHeight, filterWidth, inChannels, outChannels]` or will throw an ArgumentError.
/// If [filter.shape[-2]] is not equal to [input.shape[-1]] - will throw an ArgumentError.
///
/// The [padding] indicates the type of algorithm to use be before convolution.
/// [Padding.valid] means no padding will be used, while with [Padding.same] the [input] will be padded with zeros.
/// The padding policy is derived from [notes](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding).
///
/// The [strides] must be a 4-element list with first and last elements being 1, like [1, sh, sw, 1], where
/// the [sw] and [sh] are the strides of the convolution window in width and height dimensions.
///
/// Example:
/// ```dart
/// final input = Tensor.ones([1, 16, 16, 3]);
/// final filter = uniform([5, 5, 3, 32]);
///
/// final y = conv2D(input, filter);
/// print(y.shape); // [1, 12, 12, 32]
///
/// final t = conv2D(input, filter, padding: Padding.same);
/// print(t.shape); // [1, 16, 16, 32]
///
/// final v = conv2D(input, filter, strides: [1, 2, 2, 1]);
/// print(v.shape); // [1, 6, 6, 32]
/// ```
Tensor conv2D(Tensor input, Tensor filter, {List<int> strides = const [1, 1, 1, 1], Padding padding = Padding.valid}) {
  if (input.rank != 4) {
    throw ArgumentError(
      "The input must be of rank 4 with shape [batchSize, height, width, channels], but received a tensor with shape ${input.shape} of rank ${input.rank}",
      'input',
    );
  }
  if (filter.rank != 4) {
    throw ArgumentError(
      "Filter must be of rank 4 with shape [kernelHeight, kernelWidth, channelsIn, channelsOut], but received a tensor with shape ${filter.shape} of rank ${filter.rank}",
      'filter',
    );
  }
  if (filter.shape[-2] != input.shape[-1]) {
    throw ArgumentError(
      "The number of input channels must be equal to a number of in-filters', but ${input.shape[-1]} != ${filter.shape[3]}",
    );
  }
  if (input.dType != filter.dType) {
    throw ArgumentError(
      'Input tensor and filter must be of the same dType, but received ${input.dType} != ${filter.dType}',
    );
  }
  if (strides.length != 4 || strides[0] != 1 || strides[3] != 1) {
    throw ArgumentError(
      "The strides should be of the length 4, with first and last elements being 1, [1, x, y, 1], but received $strides",
      'strides',
    );
  }

  if (padding == Padding.same) {
    input = pad(
      input,
      computePaddingConv2D(inputShape: input.shape.list, filterShape: filter.shape.list, strides: strides),
    );
  }
  List<int> outputShape = computeOutputShapeConv2D(
    inputShape: input.shape.list,
    filterShape: filter.shape.list,
    strides: strides,
  );

  // reshaping in order to use matmul with input patches
  List<int> filterShape = filter.shape.list;
  filter = reshape(filter, [filterShape[0] * filterShape[1] * filterShape[2], filterShape[3]]);

  List buffer = emptyBuffer(input.dType, outputShape.reduce((e1, e2) => e1 * e2));
  int size = outputShape[1] * outputShape[2] * outputShape[3];
  for (int i = 0; i < outputShape[1]; i += 1) {
    for (int j = 0; j < outputShape[2]; j += 1) {
      Tensor patch = reshape(
        slice(
          input,
          [0, i * strides[1], j * strides[2], 0],
          [input.shape[0], i * strides[1] + filterShape[0], j * strides[2] + filterShape[1], input.shape[-1]],
        ),
        [input.shape[0], filter.shape[0]], // [batch, patch size]
      );

      // convolution itself
      NumericTensor convValue = matmul(patch, filter) as NumericTensor;

      // filling the output buffer according to BHWC
      for (int b = 0; b < outputShape[0]; b += 1) {
        for (int c = 0; c < outputShape[3]; c += 1) {
          buffer[b * size + (i * outputShape[2] + j) * outputShape[3] + c] = convValue.buffer[b * outputShape[3] + c];
        }
      }
    }
  }

  return Tensor.fromTypedDataList(buffer, outputShape, dType: input.dType);
}

/// Computes the gradients of 2D convolution with respect to the filter.
Tensor conv2DBackpropFilter(
  Tensor input,
  TensorShape filterSize,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  if (input.rank != 4) {
    throw ArgumentError(
      "The input must be of rank 4 with shape [batchSize, height, width, channels], but received tensor with shape ${input.shape} of rank ${input.rank}",
      'input',
    );
  }
  if (filterSize.rank != 4) {
    throw ArgumentError(
      "FilterSize must be a rank 4 shape, but received shape $filterSize of rank ${filterSize.rank}",
      'filterSize',
    );
  }
  if (filterSize[-2] != input.shape[-1]) {
    throw ArgumentError(
      "The number of input channels must be equal to a number of in-filters, but ${input.shape[-1]} != ${filterSize[3]}",
    );
  }
  if (input.dType != outputGrad.dType) {
    throw ArgumentError(
      'The input tensor and gradients w.r.t. to output tensor must be of the same dType, but received ${input.dType} != ${outputGrad.dType}',
    );
  }
  if (strides.length != 4 || strides[0] != 1 || strides[3] != 1) {
    throw ArgumentError(
      "The strides should be of the length 4, with first and last elements being 1, [1, x, y, 1], but received $strides",
      'strides',
    );
  }

  if (padding == Padding.same) {
    input = pad(
      input,
      computePaddingConv2D(inputShape: input.shape.list, filterShape: filterSize.list, strides: strides),
    );
  }

  List<int> outputShape = computeOutputShapeConv2D(
    inputShape: input.shape.list,
    filterShape: filterSize.list,
    strides: strides,
  );

  List buffer = emptyBuffer(input.dType, filterSize.size);

  for (int i = 0; i < outputShape[1]; i += 1) {
    for (int j = 0; j < outputShape[2]; j += 1) {
      Tensor patch = reshape(
        slice(
          input,
          [0, i * strides[1], j * strides[2], 0],
          [input.shape[0], i * strides[1] + filterSize[0], j * strides[2] + filterSize[1], input.shape[-1]],
        ),
        [input.shape[0], filterSize[0] * filterSize[1], filterSize[2], 1], // [batch, fh*fw, channels, 1]
      );

      Tensor gradPatch = squeeze(
        slice(outputGrad, [0, i, j, 0], [outputGrad.shape[0], i + 1, j + 1, outputGrad.shape[3]]),
        axis: [1],
      ); // [batch, 1, channelsOut]

      for (int f = 0; f < filterSize[0] * filterSize[1]; f += 1) {
        Tensor fPatch = squeeze(
          slice(patch, [0, f, 0, 0], [input.shape[0], f + 1, filterSize[2], 1]),
          axis: [1],
        ); // [batch, channelsIn, 1]
        NumericTensor convValue = reduceSum(matmul(fPatch, gradPatch), axis: [0]) as NumericTensor; // [cIn, cOut]
        for (int c = 0; c < filterSize[2]; c += 1) {
          for (int cOut = 0; cOut < filterSize[3]; cOut += 1) {
            buffer[(f * filterSize[2] + c) * filterSize[3] + cOut] += convValue.buffer[c * filterSize[3] + cOut];
          }
        }
      }
    }
  }

  return Tensor.fromTypedDataList(buffer, filterSize.list, dType: input.dType);
}

/// Computes the gradients of 2D convolution with respect to the input.
Tensor conv2DBackpropInput(
  TensorShape inputSize,
  Tensor filter,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  if (inputSize.rank != 4) {
    throw ArgumentError(
      "The input shape must be of rank 4, but received shape $inputSize of rank ${inputSize.rank}",
      'input',
    );
  }
  if (filter.rank != 4) {
    throw ArgumentError(
      "Filter must be of rank 4 with shape [kernelHeight, kernelWidth, channelsIn, channelsOut], but received tensor with shape ${filter.shape} of rank ${filter.rank}",
      'filter',
    );
  }
  if (filter.shape[-2] != inputSize[-1]) {
    throw ArgumentError(
      "The number of input channels must be equal to a number of in-filters, but ${inputSize[-1]} != ${filter.shape[3]}",
    );
  }
  if (outputGrad.dType != filter.dType) {
    throw ArgumentError(
      'The filter tensor and gradients w.r.t. to output tensor must be of the same dType, but received ${outputGrad.dType} != ${filter.dType}',
    );
  }
  if (strides.length != 4 || strides[0] != 1 || strides[3] != 1) {
    throw ArgumentError(
      "The strides should be of the length 4, with first and last elements being 1, [1, x, y, 1], but received $strides",
      'strides',
    );
  }

  List<int> inputShape = inputSize.list;
  late List padValues;
  if (padding == Padding.same) {
    padValues =
        (computePaddingConv2D(inputShape: inputSize.list, filterShape: filter.shape.list, strides: strides)
                as NumericTensor)
            .buffer;
    inputShape[1] += padValues[2] + padValues[3] as int;
    inputShape[2] += padValues[4] + padValues[5] as int;
  }

  List<int> filterShape = filter.shape.list;
  filter = reshape(filter, [filter.shape[0] * filter.shape[1] * filter.shape[2], filter.shape[-1]]);

  List buffer = emptyBuffer(filter.dType, inputSize.size);
  int size = inputShape[1] * inputShape[2] * inputShape[3];
  for (int i = 0; i < outputGrad.shape[1]; i += 1) {
    for (int j = 0; j < outputGrad.shape[2]; j += 1) {
      int index = i * strides[1];
      int jndex = j * strides[2];

      for (int b = 0; b < inputSize[0]; b += 1) {
        Tensor gradPatch = squeeze(
          slice(outputGrad, [b, i, j, 0], [b + 1, i + 1, j + 1, outputGrad.shape[3]]),
          axis: [0, 1],
        ); // [1, channelsOut]
        NumericTensor gradValue = matmul(filter, gradPatch, transposeB: true) as NumericTensor;

        for (int f1 = 0; f1 < filterShape[0]; f1 += 1) {
          for (int f2 = 0; f2 < filterShape[1]; f2 += 1) {
            for (int c = 0; c < filterShape[2]; c += 1) {
              buffer[b * size + ((f1 + index) * inputShape[2] + f2 + jndex) * inputShape[3] + c] +=
                  gradValue.buffer[(f1 * filterShape[1] + f2) * filterShape[2] + c];
            }
          }
        }
      }
    }
  }
  if (padding == Padding.valid) {
    return Tensor.fromTypedDataList(buffer, inputSize.list, dType: filter.dType);
  } else {
    final result = Tensor.fromTypedDataList(buffer, inputShape, dType: filter.dType);
    return slice(
      result,
      [0, padValues[2], padValues[4], 0],
      [inputSize[0], inputShape[1] - padValues[3] as int, inputShape[2] - padValues[5] as int, inputSize[3]],
    );
  }
}

/// Computes a 1D convolution of given 3D [input] with 3D [filter].
///
/// With given [filters], [strides] and [padding] this function performs a cross-corelation operation on [input] images,
/// producing convolved Tensor of the same dType as [filters] and [inputs].
///
/// See [Conv1D] layer for using convolution in the neural network.
///
/// This function reshapes input Tensors and performs [conv2d] on them.
///
/// The [input] Tensor must be of the rank 3 with shape structure as `[batchSize, width, channels]`, otherwise will throw an ArgumentError.
///
/// The [filter] Tensor must be of the rank 3 with shape structure `[filterWidth, inChannels, outChannels]` or will throw an ArgumentError.
/// If [filter.shape[-2]] is not equal to [input.shape[-1]] - will throw an ArgumentError.
///
/// The [padding] indicates the type of algorithm to use be before convolution.
/// [Padding.valid] means no padding will be used, while with [Padding.same] the [input] will be padded with zeros.
/// The padding policy is derived from [notes](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding).
///
/// The [strides] must be a 3-element list with first and last elements being 1, like [1, sw, 1], where
/// [sw] is the stride of the convolution window.
///
/// Example:
/// ```dart
/// final input = Tensor.ones([1, 16, 3]);
/// final filter = uniform([5, 3, 32]);
///
/// final y = conv1D(input, filter);
/// print(y.shape); // [1, 12, 32]
///
/// final t = conv1D(input, filter, padding: Padding.same);
/// print(t.shape); // [1, 16, 32]
///
/// final v = conv1D(input, filter, strides: [1, 2, 1]);
/// print(v.shape); // [1, 6, 32]
/// ```
Tensor conv1D(Tensor input, Tensor filter, {List<int> strides = const [1, 1, 1], Padding padding = Padding.valid}) {
  input = expandDims(input, 1);
  filter = expandDims(filter, 0);
  strides = [strides[0], 1, strides[1], strides[2]];

  final Tensor convResult = conv2D(input, filter, strides: strides, padding: padding);
  if (convResult.shape[1] != 1) {
    throw Exception('Algorithm error');
  }
  return reshape(convResult, [convResult.shape[0], convResult.shape[2], convResult.shape[3]]);
}

/// Computes the gradients of 1D convolution with respect to the input.
Tensor conv1DBackpropInput(
  TensorShape inputSize,
  Tensor filter,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  outputGrad = expandDims(outputGrad, 1);
  filter = expandDims(filter, 0);
  strides = [strides[0], 1, strides[1], strides[2]];

  return squeeze(
    conv2DBackpropInput(
      TensorShape([inputSize[0], 1, inputSize[1], inputSize[2]]),
      filter,
      outputGrad,
      strides: strides,
      padding: padding,
    ),
    axis: [1],
  );
}

/// Computes the gradients of 1D convolution with respect to the filter.
Tensor conv1DBackpropFilter(
  Tensor input,
  TensorShape filterSize,
  Tensor outputGrad, {
  required List<int> strides,
  required Padding padding,
}) {
  input = expandDims(input, 1);
  outputGrad = expandDims(outputGrad, 1);
  strides = [strides[0], 1, strides[1], strides[2]];

  return squeeze(
    conv2DBackpropFilter(
      input,
      TensorShape([filterSize[0], 1, filterSize[1], filterSize[2]]),
      outputGrad,
      strides: strides,
      padding: padding,
    ),
    axis: [1],
  );
}
