// ignore_for_file: prefer_final_fields

import 'package:loredart_nn/src/utils/deserialization_utils.dart';
import 'package:loredart_tensor/loredart_tensor.dart';

import '../utils/serialization_utils.dart';
import '/src/nn_ops/other_ops.dart';
import '/src/modules/activations.dart';
import '/src/nn/initializer.dart';
import '/src/nn_ops/convolution.dart';
import '/src/nn_ops/pooling.dart';
import '/src/modules/module.dart';

/// The base class for model layer -
/// a callable module that performs a computation on a single Tensor and outputs a single Tensor.
///
/// Computation logic of a layer defined in the [Layer.call] methods.
///
/// A layer may have zero or more trainable and/or non-trainable parameters that are involved in the computations.
/// Values of trainable weights will be updated during the fitting process if layer is [trainable].
abstract class Layer implements Module {
  /// Name of the layer that is unique within one [Model].
  @override
  late final String name;

  /// Whether this is trainable.
  late bool trainable;

  Map<String, Tensor> _trainableWeights = {};
  Map<String, Tensor> _nonTrainableWeights = {};

  /// Expected shape of the input Tensor.
  late final List<int> inputShape;

  /// The shape of the output [Tensor] from this, excluding the batch size.
  late final List<int> outputShape;

  /// Whether this was built, i.e., initiated all weights.
  bool _built = false;

  @override
  List<Tensor> get trainableParams;

  /// The constructor of Layer.
  Layer(this.trainable);

  /// Constructs and returns layer from its [config].
  ///
  /// If [config] includes the layer's weights then it will set their values as well.
  @override
  factory Layer.fromJson(Map<String, dynamic> config) => deserializeLayer(config.remove('type'), config);

  /// The gradient function of this layer, constructed during [Layer.call] with `training` set to true.
  ///
  /// Call of this function computes gradients of layer w.r.t. the input and trainable weights (if those exist)
  /// with provided [upstream] gradient (the gradient from all the layers coming after this layer).
  @override
  List<Tensor> Function(Tensor upstream)? gradient;

  /// Creates and initialize layer's weights or/and variables.
  void build(List<int> inputShape);

  /// Adds new named (non)[trainable] weight as a Tensor of given [shape], [dType] and with values from [initializer].
  void addWeight({
    required String name,
    required List<int> shape,
    required Initializer initializer,
    required bool trainable,
    DType dType = DType.float32,
  }) {
    if (trainable) {
      _trainableWeights[name] = initializer(shape, dType);
    } else {
      _nonTrainableWeights[name] = initializer(shape, dType);
    }
  }

  /// Checks if [inputShape] of a layer meets the [expectedRank] and has all non-zero values and
  /// if any of the reqs aren't met will throw an [ArgumentError].
  void checkBuildingShape(List<int> inputShape, int expectedRank, {String? shapeStructure}) {
    if (inputShape.any((element) => element <= 0)) {
      throw ArgumentError(
        'Shape of input tensor must includes only positive integers, but layer $name received $inputShape as an input shape.',
        'inputShape',
      );
    }
    if (expectedRank >= 0 && inputShape.length != expectedRank) {
      shapeStructure ??= List.generate(expectedRank, (i) => 'N$i').join(', ');
      throw ArgumentError(
        'During the building layer $name expected the input shape of the length $expectedRank: [$shapeStructure]'
        ' (with excluded batch dim), but received $inputShape',
      );
    }
  }

  /// Checks if [shape] of an input batched Tensor met requirements of the layer and if not - throws an [ArgumentError].
  void checkInputShape(TensorShape shape) {
    if (shape.rank - 1 != inputShape.length || !shape.equalWithLastDims(TensorShape(inputShape))) {
      throw ArgumentError(
        "Layer $name expected ${inputShape.length + 1}-D tensor of the shape [batchSize, ${inputShape.map((e) => e.toString()).join(', ')}] as input"
        ', but received tensor with shape $inputShape.',
      );
    }
  }

  /// Checks whether [candidateShape] equal to a current shape of a [weight].
  /// If [candidateShape] fails check, will throw an [ArgumentError].
  void _checkWeightShape(String weight, bool isTrainable, TensorShape candidateShape) {
    TensorShape currentShape = isTrainable ? _trainableWeights[weight]!.shape : _nonTrainableWeights[weight]!.shape;
    if (!currentShape.equalTo(candidateShape)) {
      throw ArgumentError(
        "Cannot set weight $weight (of layer $name): expected Tensor of shape $currentShape, but received one of shape $candidateShape",
      );
    }
  }

  /// Applies layer's computation logic on the batched [input] Tensor.
  ///
  /// If layer wasn't built (i.e. the inputShape wasn't initialized) this method might invoke [Layer.build] method
  /// with shape of the [input].
  ///
  /// The [training] indicates if call meant for training or inference.
  /// Additionally, if [training] is true the layer will construct it's [gradient] function.
  ///
  /// The [input] should be a batched Tensor with correct non-batch shape dims or otherwise will throw an [ArgumentError].
  Tensor call(Tensor input, {bool training = false});

  /// Safely sets values of the layer's weights.
  ///
  /// If shapes of a provided Tensors doesn't match corresponding shapes of weights will throw an [ArgumentError].
  void setWeights({List<Tensor>? trainableWeights, List<Tensor>? nonTrainableWeights});

  @override
  void updateTrainableParams(List<Tensor> updatedParams) {
    if (trainable) {
      setWeights(trainableWeights: updatedParams);
    }
  }

  @override
  String toString() {
    return "$runtimeType(name: $name, trainable: $trainable)";
  }

  /// Returns config of this as JSON-serializable Map.
  /// Also will store all Tensor weights if [withWeights] is true.
  @override
  Map<String, dynamic> toJson({withWeights = true}) {
    if (!_built) {
      throw ModuleSerializationError.unbuildLayer(name);
    }
    return {'name': name, 'inputShape': inputShape, 'trainable': trainable};
  }
}

/// A regular fully-connected layer.
///
/// Implements classical nn operation `output = activation(input*kernel + bias)`
/// transforming batch of input-dim vectors into [units]-dim output vectors.
///
class Dense extends Layer {
  static int _denseCounter = 0;

  /// The dimensionality of the output space (number of output neurons).
  late final int units;

  /// Whether the layer uses a bias vector.
  late final bool useBias;

  /// Initializer for the `kernel` weights matrix.
  late final Initializer kernelInitializer;

  /// Initializer for the `bias` vector. Used only if [useBias] is true.
  late final Initializer biasInitializer;

  /// The activation function to use.
  /// If you don't specify anything, no activation is applied.
  late final Activation activation;

  /// Creates new Dense layer with given number of [units] and [activation] function.
  ///
  /// The [inputShape] is optional for dynamic graph construction, but can be provided
  /// to immediately initialize the layer's weights.
  Dense(
    this.units, {
    Activation? activation,
    this.useBias = true,
    Initializer? kernelInitializer,
    Initializer? biasInitializer,
    String? name,
    bool trainable = true,
    List<int>? inputShape,
  }) : super(trainable) {
    if (units <= 0) {
      throw ArgumentError('The number of units must be positive integer, but received $units', 'units');
    }

    this.activation = activation ?? Activations.linear;

    this.kernelInitializer = kernelInitializer ?? GlorotUniform();
    if (useBias) {
      this.biasInitializer = biasInitializer ?? Zeros();
    }

    if (name == null) {
      _denseCounter += 1;
    }
    this.name = name ?? 'Dense-$_denseCounter';
    if (inputShape != null) {
      build(inputShape);
    }
  }

  /// Returns the list of trainable parameters: `[kernel, bias (optional)]`.
  @override
  List<Tensor> get trainableParams => [_trainableWeights['kernel']!] + (useBias ? [_trainableWeights['bias']!] : []);

  /// The weight matrix, or "kernel", of the layer.
  Tensor? get kernel => _trainableWeights['kernel'];

  /// The bias vector of the layer. Available only if [useBias] is true.
  Tensor? get bias => _trainableWeights['bias'];

  /// Creates and initializes the layer's weights. Usually called internally by library, not by user.
  ///
  /// The expected [inputShape] must have a rank of 1 (excluding the batch dimension),
  /// corresponding to the number of input features.
  ///
  /// The weight matrix `kernel` will have shape `[inputShape[0], units]`.
  /// The bias vector `bias`, if used, will have shape `[units]`.
  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 1, shapeStructure: 'inputUnits');

      addWeight(name: 'kernel', shape: [inputShape[0], units], initializer: kernelInitializer, trainable: trainable);
      if (useBias) {
        addWeight(name: 'bias', shape: [units], initializer: biasInitializer, trainable: trainable);
      }

      this.inputShape = inputShape;
      super.outputShape = [units];
      _built = true;
    }
  }

  /// Safely sets the values of the layer's weights. Usually called internally by library, not by user.
  ///
  /// It expects one Tensor for the `kernel` in [trainableWeights], and a second
  /// optional Tensor for the `bias` if [useBias] is true.
  ///
  /// Throws an [ArgumentError] if the shapes of the provided Tensors do not match
  /// the corresponding shapes of the weights.
  @override
  void setWeights({List<Tensor>? trainableWeights, List<Tensor>? nonTrainableWeights}) {
    if (trainableWeights != null) {
      _checkWeightShape('kernel', true, trainableWeights[0].shape);
      _trainableWeights['kernel'] = trainableWeights[0];
      if (useBias && trainableWeights.length == 2) {
        _checkWeightShape('bias', true, trainableWeights[1].shape);
        _trainableWeights['bias'] = trainableWeights[1];
      }
    }
  }

  /// Applies the dense layer's computation logic on the batched [input] Tensor.
  ///
  /// The computation is: `output = activation(input * kernel + bias)`.
  ///
  /// If [training] is true, the [gradient] function is constructed for backpropagation.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        final Tensor dUpstreamDActivation = activation.gradient!(upstream)[0];
        return <Tensor>[matmul(dUpstreamDActivation, kernel!, transposeB: true)] + //dInput
            [matmul(input, dUpstreamDActivation, transposeA: true)] + // dW
            (useBias
                ? [
                  reduceSum(dUpstreamDActivation, axis: [0]),
                ]
                : []); //dB
      };
    }

    // if (training && trainable) {
    //   gradient = (Tensor upstream) {
    //     final Tensor dUpstreamDActivation = activation.gradient!(upstream)[0];
    //     return matmulGrad([input, kernel], [], [upstream]) +
    //       (useBias ? [dUpstreamDActivation * Tensor.ones(_trainableWeights['bias']!.shape.list)] : []);
    //   };
    // }
    if (useBias) {
      return activation(matmul(input, _trainableWeights['kernel']!) + _trainableWeights['bias']!, training: training);
    } else {
      return activation(matmul(input, _trainableWeights['kernel']!), training: training);
    }
  }

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'Dense',
        'units': units,
        'useBias': useBias,
        'activation': activation.toJson(),
        'kernelInitializer': kernelInitializer.toJson(),
        'biasInitializer': biasInitializer.toJson(),
        'weights':
            withWeights
                ? [
                  tensorToJson(kernel!),
                  if (useBias) tensorToJson(bias!),
                ]
                : null,
      });

  /// Constructs a [Dense] layer from its JSON-serializable [config].
  @override
  factory Dense.fromJson(Map<String, dynamic> config) {
    return Dense(
      config['units'],
      activation: Activation.fromJson(config['activation']),
      useBias: config['useBias'],
      kernelInitializer: Initializer.fromJson(config['kernelInitializer']),
      biasInitializer: Initializer.fromJson(config['biasInitializer']),
      name: config['name'],
      trainable: config['trainable'],
      inputShape: config['inputShape'].cast<int>(),
    )..setWeights(trainableWeights: deserializeWeights(config['weights']));
  }
}

/// A layer that applies an element-wise activation function.
///
/// This layer is always non-[trainable] and has no weights.
class ActivationLayer extends Layer {
  static int _actCounter = 0;
  late final Activation activation;

  /// Creates a new Activation layer with the given [activation] function.
  ActivationLayer(this.activation, {String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _actCounter += 1;
    }
    this.name = name ?? '${activation.runtimeType}-$_actCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  /// Builds the layer by setting the [inputShape] and [outputShape].
  ///
  /// The layer accepts any input rank and the output shape is identical
  /// to the input shape since the operation is element-wise.
  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, -1);
      this.inputShape = List.from(inputShape);
      outputShape = List.from(inputShape);
      _built = true;
    }
  }

  /// Applies the activation function element-wise to the [input] Tensor.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) => activation.gradient!(upstream);
    }

    return activation(input, training: training);
  }

  @override
  List<Tensor> get trainableParams => [];

  /// This layer has no weights, so this method does nothing.
  @override
  void setWeights({List<Tensor>? trainableWeights, List<Tensor>? nonTrainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'ActivationLayer',
        'activation': activation.toJson(),
      });

  @override
  factory ActivationLayer.fromJson(Map<String, dynamic> config) => ActivationLayer(
    Activation.fromJson(config['activation']),
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// Layer normalization layer [*](https://arxiv.org/abs/1607.06450).
///
/// Normalizes the activations of the previous layer for each given [axis] independently
///
/// The operation is:
/// `output = gamma * (input - mean) / sqrt(variance + epsilon) + beta`
/// where mean and variance are calculated across the specified [axis].
class LayerNormalization extends Layer {
  static int _lnCounter = 0;

  /// The axis/axes that should be normalized.
  ///
  /// Typically, this is the last dimension in the inputs (default: `[-1]`).
  late final List<int> axis;

  /// Whether to add offset to the normalized tensor (using trainable `beta`).
  late final bool center;

  /// Whether to multiply the normalized tensor by a scale factor (using trainable `gamma`).
  late final bool scale;

  /// Initializer for the `beta` offset factor. Used only if [center] is true.
  late final Initializer betaInitializer;

  /// Initializer for the `gamma` scale factor. Used only if [scale] is true.
  late final Initializer gammaInitializer;

  /// Small constant added to variance to avoid division by zero.
  late final double epsilon;

  /// Creates a new LayerNormalization layer.
  LayerNormalization({
    this.axis = const [-1],
    this.center = true,
    this.scale = true,
    this.betaInitializer = const Zeros(),
    this.gammaInitializer = const Ones(),
    this.epsilon = 1e-4,
    String? name,
    List<int>? inputShape,
    bool trainable = true,
  }) : super((center || scale) && trainable) {
    if (name == null) {
      _lnCounter += 1;
    }
    this.name = name ?? 'LayerNormalization-$_lnCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  /// The scale parameter. Available only if [scale] and [trainable] are true.
  Tensor? get gamma => _trainableWeights['gamma'];

  /// The offset parameter. Available only if [center] and [trainable] are true.
  Tensor? get beta => _trainableWeights['beta'];

  /// Creates and initializes the layer's weights (`gamma` and `beta`).
  ///
  /// The weights are added only if [scale]/[center] are true and the layer is [trainable].
  /// The shape of `gamma` and `beta` is determined by the input shape and the [axis] of normalization.
  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, -1);
      this.inputShape = List.from(inputShape);
      outputShape = List.from(inputShape);

      if (trainable) {
        final axis_ = axis.map((e) => e % (inputShape.length + 1)).toList();
        List<int> paramShape =
            [1] + [for (int i = 1; i <= inputShape.length; i += 1) axis_.contains(i) ? inputShape[i - 1] : 1];

        if (scale) {
          addWeight(name: 'gamma', shape: paramShape, initializer: gammaInitializer, trainable: true);
        }
        if (center) {
          addWeight(name: 'beta', shape: paramShape, initializer: betaInitializer, trainable: true);
        }
      }

      _built = true;
    }
  }

  /// Applies the layer normalization to the [input] Tensor.
  ///
  /// Calculates mean and variance across the [axis], normalizes the input,
  /// and applies `gamma` and `beta` if configured.
  /// If [training] is true, the [gradient] function is constructed.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    final mean = reduceMean(input, axis: List.from(axis), keepDims: true);
    final variance = reduceVariance(input, axis: List.from(axis), keepDims: true);
    final sqrtVar = sqrt(variance + epsilon);
    Tensor normalized = (input - mean) / sqrtVar;

    if (training) {
      gradient = (Tensor upstream) {
        final int size = [for (int i in axis) input.shape[i]].reduce((e2, e1) => e1 * e2);
        final dNorm = scale ? upstream * gamma! : upstream;
        final dVar = reduceSum(
          dNorm * (mean - input) / (sqrtVar * (variance + epsilon) * 2),
          axis: List.from(axis),
          keepDims: true,
        );
        final dMean = reduceSum(
          -dNorm / sqrtVar,
          axis: List.from(axis),
          keepDims: true,
        ); // + dVar * (dMean/dVariance) == 0
        final dx = dNorm / sqrtVar + ((input - mean) * 2) * dVar / size + dMean / size;

        List<Tensor> grads = [dx];
        if (scale) {
          grads.add(
            reduceSum(
              upstream * normalized,
              axis: [
                for (int i = 0; i < gamma!.shape.rank; i += 1)
                  if (gamma!.shape[i] == 1) i,
              ],
              keepDims: true,
            ),
          );
        }
        if (center) {
          grads.add(
            reduceSum(
              upstream,
              axis: [
                for (int i = 0; i < beta!.shape.rank; i += 1)
                  if (beta!.shape[i] == 1) i,
              ],
              keepDims: true,
            ),
          );
        }
        return grads;
      };
    }

    if (scale) {
      normalized = normalized * gamma!;
    }
    if (center) {
      normalized = normalized + beta!;
    }

    return normalized;
  }

  @override
  List<Tensor> get trainableParams =>
      trainable
          ? ([
            ...(scale ? [gamma!] : []),
            ...(center ? [beta!] : []),
          ])
          : [];

  @override
  void setWeights({List<Tensor>? trainableWeights, List<Tensor>? nonTrainableWeights}) {
    if (trainableWeights != null) {
      int k = 0;
      if (scale) {
        _checkWeightShape('gamma', true, trainableWeights[k].shape);
        _trainableWeights['gamma'] = trainableWeights[k];
        k += 1;
      }
      if (center) {
        _checkWeightShape('beta', true, trainableWeights[k].shape);
        _trainableWeights['beta'] = trainableWeights[k];
      }
    }
  }

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'LayerNormalization',
        'axis': axis,
        'scale': scale,
        'center': center,
        'betaInitializer': betaInitializer.toJson(),
        'gammaInitializer': gammaInitializer.toJson(),
        'epsilon': epsilon,
        'weights': withWeights ? [if (scale) tensorToJson(gamma!), if (center) tensorToJson(beta!)] : null,
      });

  factory LayerNormalization.fromJson(Map<String, dynamic> config) {
    return LayerNormalization(
      axis: config['axis'],
      scale: config['scale'],
      center: config['center'],
      betaInitializer: Initializer.fromJson(config['betaInitializer']),
      gammaInitializer: Initializer.fromJson(config['gammaInitializer']),
      name: config['name'],
      inputShape: config['inputShape'].cast<int>(),
      trainable: config['trainable'],
    )..setWeights(trainableWeights: deserializeWeights(config['weights']));
  }
}

/// A simple layer that rescales and offsets the input tensor by a constant amount.
///
/// Implements the element-wise operation: `output = input * scale + offset`.
///
/// This layer is always non-[trainable].
class Rescale extends Layer {
  static int _rescaleCounter = 0;
  late final double scale;
  late final double offset;

  Rescale({required this.scale, this.offset = 0.0, String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _rescaleCounter += 1;
    }
    this.name = name ?? 'Rescale-$_rescaleCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, -1);
      this.inputShape = List.from(inputShape);
      outputShape = List.from(inputShape);
      _built = true;
    }
  }

  /// Applies the element-wise rescaling and offsetting operation to the [input] Tensor.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [upstream * scale];
      };
    }

    return (input * scale + offset);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({trainableWeights, nonTrainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'Rescale',
        'scale': scale,
        'offset': offset,
      });

  factory Rescale.fromJson(Map<String, dynamic> config) => Rescale(
    scale: config['scale'],
    offset: config['offset'],
    inputShape: config['inputShape'].cast<int>(),
    name: config['name'],
  );
}

/// 2D Convolution layer
///
/// This layer creates a convolution kernel that is convolved with the layer input
/// to produce a tensor of outputs.
///
/// Layer works only with `channels_last` format, so expected shapes:
/// input - `[batch, height, width, channels]`.
/// output - `[batch, new_height, new_width, filters]`.
class Conv2D extends Layer {
  static int _convCounter = 0;

  /// The number of output filters (dimensionality of the output space).
  late final int filters;

  /// The height and width of the 2D convolution window.
  late final List<int> kernelSize;

  /// The strides of the convolution along the height and width.
  late final List<int> strides;

  /// The padding scheme to use (e.g., 'valid' or 'same').
  late final Padding padding;

  /// The activation function to use.
  /// If you don't specify anything, no activation is applied.
  late final Activation activation;

  /// Whether the layer uses a bias vector.
  late final bool useBias;

  /// Initializer for the `kernel` weights tensor.
  late final Initializer kernelInitializer;

  /// Initializer for the `bias` vector.
  late final Initializer biasInitializer;

  /// Creates a new 2D Convolutional layer
  Conv2D(
    this.filters,
    List<int> kernelSize, {
    List<int> strides = const [1, 1],
    this.padding = Padding.valid,
    this.useBias = true,
    Activation? activation,
    Initializer? kernelInitializer,
    Initializer? biasInitializer,
    String? name,
    bool trainable = true,
    List<int>? inputShape,
  }) : super(trainable) {
    if (filters <= 0) {
      throw ArgumentError('Units must be positive integer, but received $filters', 'filters');
    }

    if (kernelSize.length == 1) {
      this.kernelSize = [kernelSize[0], kernelSize[0]];
    } else if (kernelSize.length == 2) {
      this.kernelSize = List.from(kernelSize, growable: false);
    } else {
      throw ArgumentError('Kernel size must be of a length 1 or 2, but layer $name received $kernelSize', 'kernelSize');
    }

    if (strides.length == 1) {
      this.strides = [1, strides[0], strides[0], 1];
    } else if (strides.length == 2) {
      this.strides = [1, ...strides, 1];
    } else {
      throw ArgumentError('Strides must be of a length 1 or 2, but layer $name received $strides', 'strides');
    }

    this.activation = activation ?? Activations.linear;

    this.kernelInitializer = kernelInitializer ?? GlorotUniform();
    if (useBias) {
      this.biasInitializer = biasInitializer ?? Zeros();
    }

    if (name == null) {
      _convCounter += 1;
    }
    this.name = name ?? 'Conv2D-$_convCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  List<Tensor> get trainableParams => [_trainableWeights['kernel']!] + (useBias ? [_trainableWeights['bias']!] : []);

  /// The convolution kernel weights of the layer.
  Tensor? get kernel => _trainableWeights['kernel'];

  /// The bias vector of the layer. Available only if [useBias] is true, otherwise is null.
  Tensor? get bias => _trainableWeights['bias'];

  /// Creates and initializes the layer's weights based on the input shape.
  ///
  /// The expected [inputShape] must have a rank of 3 (excluding the batch dimension):
  /// `[height, width, in_channels]`.
  ///
  /// The `kernel` weight tensor will have shape: `[kernel_h, kernel_w, in_channels, filters]`.
  /// The `bias` vector (if used) will have shape: `[filters]`.
  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 3, shapeStructure: 'height, width, channels');

      addWeight(
        name: 'kernel',
        shape: [...kernelSize, inputShape[2], filters],
        initializer: kernelInitializer,
        trainable: trainable,
      );
      if (useBias) {
        addWeight(name: 'bias', shape: [filters], initializer: biasInitializer, trainable: trainable);
      }

      this.inputShape = inputShape;
      outputShape = computeBatchlessOutputShapeConv2D(
        inputShape: inputShape,
        filterShape: kernel!.shape.list,
        strides: strides,
        padding: padding,
      );
      _built = true;
    }
  }

  /// Applies the 2D convolution and activation operation on the batched [input] Tensor.
  ///
  /// The operation is: `output = activation(conv2D(input, kernel) + bias)`.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        final Tensor outputGrad = activation.gradient!(upstream)[0];
        return [
              conv2DBackpropInput(input.shape, kernel!, outputGrad, strides: strides, padding: padding),
              conv2DBackpropFilter(input, kernel!.shape, outputGrad, strides: strides, padding: padding),
            ] +
            (useBias
                ? [
                  reduceSum(outputGrad, axis: [0, 1, 2]),
                ]
                : []);
      };
    }

    return activation(conv2D(input, kernel!, strides: strides, padding: padding) + bias!, training: training);
  }

  @override
  void setWeights({List<Tensor>? trainableWeights, List<Tensor>? nonTrainableWeights}) {
    if (trainableWeights != null) {
      _checkWeightShape('kernel', true, trainableWeights[0].shape);
      _trainableWeights['kernel'] = trainableWeights[0];
      if (useBias && trainableWeights.length == 2) {
        _checkWeightShape('bias', true, trainableWeights[1].shape);
        _trainableWeights['bias'] = trainableWeights[1];
      }
    }
  }

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'Conv2D',
        'filters': filters,
        'kernelSize': kernelSize,
        'strides': strides.sublist(1, 3),
        'padding': padding.index,
        'useBias': useBias,
        'activation': activation.toJson(),
        'biasInitializer': biasInitializer.toJson(),
        'kernelInitializer': kernelInitializer.toJson(),
        'weights': withWeights ? [tensorToJson(kernel!), if (useBias) tensorToJson(bias!)] : null,
      });

  @override
  factory Conv2D.fromJson(Map<String, dynamic> config) => Conv2D(
    config['filters'],
    config['kernelSize'].cast<int>(),
    strides: config['strides'].cast<int>(),
    padding: Padding.values[config['padding']],
    useBias: config['useBias'],
    activation: Activation.fromJson(config['activation']),
    biasInitializer: Initializer.fromJson(config['biasInitializer']),
    kernelInitializer: Initializer.fromJson(config['kernelInitializer']),
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
    trainable: config['trainable'],
  )..setWeights(trainableWeights: deserializeWeights(config['weights']));
}

/// 1D Convolution layer
///
/// This layer creates a convolution kernel that is convolved with the layer input
/// to produce a tensor of outputs.
///
/// Layer works only with `channels_last` format, so expected shapes:
/// input - `[batch, steps, channels]`.
/// output - `[batch, new_steps, filters]`.
class Conv1D extends Layer {
  static int _convCounter = 0;

  /// The number of output filters (dimensionality of the output space).
  late final int filters;

  /// The size of the 1D convolution window.
  late final int kernelSize;

  /// The strides of the convolution along the steps dimension.
  late final List<int> strides;

  /// The padding scheme to use (e.g., 'valid' or 'same').
  late final Padding padding;

  /// The activation function to use.
  /// If you don't specify anything, no activation is applied.
  late final Activation activation;

  /// Whether the layer uses a bias vector.
  late final bool useBias;

  /// Initializer for the `kernel` weights tensor.
  late final Initializer kernelInitializer;

  /// Initializer for the `bias` vector.
  late final Initializer biasInitializer;

  /// Creates a new 1D Convolutional layer
  Conv1D(
    this.filters,
    this.kernelSize, {
    int strides = 1,
    this.padding = Padding.valid,
    this.useBias = true,
    Activation? activation,
    Initializer? kernelInitializer,
    Initializer? biasInitializer,
    String? name,
    bool trainable = true,
    List<int>? inputShape,
  }) : super(trainable) {
    if (filters <= 0) {
      throw ArgumentError('Units must be positive integer, but received $filters', 'filters');
    }
    this.strides = [1, strides, 1];

    this.activation = activation ?? Activations.linear;

    this.kernelInitializer = kernelInitializer ?? GlorotUniform();
    if (useBias) {
      this.biasInitializer = biasInitializer ?? Zeros();
    }

    if (name == null) {
      _convCounter += 1;
    }
    this.name = name ?? 'Conv1D-$_convCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  List<Tensor> get trainableParams => [_trainableWeights['kernel']!] + (useBias ? [_trainableWeights['bias']!] : []);

  Tensor? get kernel => _trainableWeights['kernel'];
  Tensor? get bias => _trainableWeights['bias'];

  /// Creates and initializes the layer's weights based on the input shape.
  ///
  /// The expected [inputShape] must have a rank of 2 (excluding the batch dimension):
  /// `[steps, in_channels]`.
  ///
  /// The `kernel` weight tensor will have shape: `[kernel_size, in_channels, filters]`.
  /// The `bias` vector (if used) will have shape: `[filters]`.
  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 2, shapeStructure: 'width, channels');

      addWeight(
        name: 'kernel',
        shape: [kernelSize, inputShape[1], filters],
        initializer: kernelInitializer,
        trainable: trainable,
      );
      if (useBias) {
        addWeight(name: 'bias', shape: [filters], initializer: biasInitializer, trainable: trainable);
      }

      this.inputShape = inputShape;
      outputShape = computeBatchlessOutputShapeConv1D(
        inputShape: inputShape,
        filterShape: kernel!.shape.list,
        strides: strides,
        padding: padding,
      );
      _built = true;
    }
  }

  /// Applies the 1D convolution and activation operation on the batched [input] Tensor.
  ///
  /// The operation is: `output = activation(conv1D(input, kernel) + bias)`.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        final Tensor outputGrad = activation.gradient!(upstream)[0];
        return [
              conv1DBackpropInput(input.shape, kernel!, outputGrad, strides: strides, padding: padding),
              conv1DBackpropFilter(input, kernel!.shape, outputGrad, strides: strides, padding: padding),
            ] +
            (useBias
                ? [
                  reduceSum(outputGrad, axis: [0, 1]),
                ]
                : []);
      };
    }

    return activation(conv1D(input, kernel!, strides: strides, padding: padding) + bias!, training: training);
  }

  @override
  void setWeights({List<Tensor>? trainableWeights, List<Tensor>? nonTrainableWeights}) {
    if (trainableWeights != null) {
      _checkWeightShape('kernel', true, trainableWeights[0].shape);
      _trainableWeights['kernel'] = trainableWeights[0];
      if (useBias && trainableWeights.length == 2) {
        _checkWeightShape('bias', true, trainableWeights[1].shape);
        _trainableWeights['bias'] = trainableWeights[1];
      }
    }
  }

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'Conv1D',
        'filters': filters,
        'kernelSize': kernelSize,
        'strides': strides[1],
        'padding': padding.index,
        'useBias': useBias,
        'activation': activation.toJson(),
        'biasInitializer': biasInitializer.toJson(),
        'kernelInitializer': kernelInitializer.toJson(),
        'weights': withWeights ? [tensorToJson(kernel!), if (useBias) tensorToJson(bias!)] : null,
      });

  @override
  factory Conv1D.fromJson(Map<String, dynamic> config) => Conv1D(
    config['filters'],
    config['kernelSize'],
    strides: config['strides'],
    padding: Padding.values[config['padding']],
    useBias: config['useBias'],
    activation: Activation.fromJson(config['activation']),
    biasInitializer: Initializer.fromJson(config['biasInitializer']),
    kernelInitializer: Initializer.fromJson(config['kernelInitializer']),
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
    trainable: config['trainable'],
  )..setWeights(trainableWeights: deserializeWeights(config['weights']));
}

/// 2D Max Pooling layer.
///
/// Downsamples the input by taking the maximum value over a spatial pooling window defined by [kernelSize].
/// This layer is always non-[trainable].
class MaxPool2D extends Layer {
  /// The size of the pooling window.
  late final List<int> kernelSize;

  /// The strides of the pooling window.
  late final List<int> strides;

  /// The padding scheme to use (e.g., 'valid' or 'same').
  late final Padding padding;

  static int _maxPoolCounter = 0;

  /// Creates a 2D Max Pooling layer.
  MaxPool2D(
    List<int> kernelSize, {
    List<int> strides = const [1, 1],
    this.padding = Padding.valid,
    String? name,
    List<int>? inputShape,
  }) : super(false) {
    if (kernelSize.length == 1) {
      this.kernelSize = [kernelSize[0], kernelSize[0]];
    } else if (kernelSize.length == 2) {
      this.kernelSize = List.from(kernelSize, growable: false);
    } else {
      throw ArgumentError('Kernel size must be of a length 1 or 2, but layer $name received $kernelSize', 'kernelSize');
    }

    if (strides.length == 1) {
      this.strides = [1, strides[0], strides[0], 1];
    } else if (strides.length == 2) {
      this.strides = [1, ...strides, 1];
    } else {
      throw ArgumentError('Strides must be of a length 1 or 2, but layer $name received $strides', 'strides');
    }

    if (name == null) {
      _maxPoolCounter += 1;
    }
    this.name = name ?? 'MaxPool2D-$_maxPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 3, shapeStructure: 'height, width, channels');

      this.inputShape = inputShape;
      outputShape = computeBatchlessOutputShapeConv2D(
        inputShape: inputShape,
        filterShape: [...kernelSize, inputShape[2], inputShape[2]],
        strides: strides,
        padding: padding,
      );
      _built = true;
    }
  }

  /// Applies the 2D max pooling operation on the batched [input] Tensor.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [maxPool2DBackprop(input, kernelSize, upstream, strides: strides, padding: padding)];
      };
    }

    return maxPool2D(input, kernelSize, strides: strides, padding: padding);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'MaxPool2D',
        'kernelSize': kernelSize,
        'strides': strides.sublist(1, 3),
        'padding': padding.index,
        'weights': null,
      });

  @override
  factory MaxPool2D.fromJson(Map<String, dynamic> config) => MaxPool2D(
    config['kernelSize'].cast<int>(),
    strides: config['strides'].cast<int>(),
    padding: Padding.values[config['padding']],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// 1D Max Pooling layer.
///
/// Downsamples the input by taking the maximum value over a spatial pooling window defined by [kernelSize].
/// This layer is always non-[trainable].
class MaxPool1D extends Layer {
  /// The size of the pooling window.
  late final int kernelSize;

  /// The strides of the pooling window.
  late final List<int> strides;

  /// The padding scheme to use (e.g., 'valid' or 'same').
  late final Padding padding;

  static int _maxPoolCounter = 0;

  /// Creates a 1D Max Pooling layer.
  MaxPool1D(this.kernelSize, {int strides = 1, this.padding = Padding.valid, String? name, List<int>? inputShape})
    : super(false) {
    this.strides = [1, strides, 1];

    if (name == null) {
      _maxPoolCounter += 1;
    }
    this.name = name ?? 'MaxPool1D-$_maxPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 2, shapeStructure: 'width, channels');

      this.inputShape = inputShape;
      outputShape = computeBatchlessOutputShapeConv1D(
        inputShape: inputShape,
        filterShape: [kernelSize, inputShape[1], inputShape[1]],
        strides: strides,
        padding: padding,
      );
      _built = true;
    }
  }

  /// Applies the 1D max pooling operation on the batched [input] Tensor.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [maxPool1DBackprop(input, kernelSize, upstream, strides: strides, padding: padding)];
      };
    }

    return maxPool1D(input, kernelSize, strides: strides, padding: padding);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'MaxPool1D',
        'kernelSize': kernelSize,
        'strides': strides[1],
        'padding': padding.index,
        'weights': null,
      });

  @override
  factory MaxPool1D.fromJson(Map<String, dynamic> config) => MaxPool1D(
    config['kernelSize'],
    strides: config['strides'],
    padding: Padding.values[config['padding']],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// 2D Average Pooling layer.
///
/// Downsamples the input by taking the average over a spatial pooling window defined by [kernelSize].
/// This layer is always non-[trainable].
class AveragePool2D extends Layer {
  /// The size of the pooling window.
  late final List<int> kernelSize;

  /// The strides of the pooling window.
  late final List<int> strides;

  /// The padding scheme to use (e.g., 'valid' or 'same').
  late final Padding padding;

  static int _avgPoolCounter = 0;

  /// Creates a 2D Average Pooling layer.
  AveragePool2D(
    List<int> kernelSize, {
    List<int> strides = const [1, 1],
    this.padding = Padding.valid,
    String? name,
    List<int>? inputShape,
  }) : super(false) {
    if (kernelSize.length == 1) {
      this.kernelSize = [kernelSize[0], kernelSize[0]];
    } else if (kernelSize.length == 2) {
      this.kernelSize = List.from(kernelSize, growable: false);
    } else {
      throw ArgumentError('Kernel size must be of a length 1 or 2, but layer $name received $kernelSize', 'kernelSize');
    }

    if (strides.length == 1) {
      this.strides = [1, strides[0], strides[0], 1];
    } else if (strides.length == 2) {
      this.strides = [1, ...strides, 1];
    } else {
      throw ArgumentError('Strides must be of a length 1 or 2, but layer $name received $strides', 'strides');
    }

    if (name == null) {
      _avgPoolCounter += 1;
    }
    this.name = name ?? 'AveragePool2D-$_avgPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 3, shapeStructure: 'height, width, channels');

      this.inputShape = inputShape;
      outputShape = computeBatchlessOutputShapeConv2D(
        inputShape: inputShape,
        filterShape: [...kernelSize, inputShape[2], inputShape[2]],
        strides: strides,
        padding: padding,
      );
      _built = true;
    }
  }

  /// Applies the 2D avg pooling operation on the batched [input] Tensor.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [avgPool2DBackprop(input.shape, kernelSize, upstream, strides: strides, padding: padding)];
      };
    }

    return avgPool2D(input, kernelSize, strides: strides, padding: padding);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'AveragePool2D',
        'kernelSize': kernelSize,
        'strides': strides.sublist(1, 3),
        'padding': padding.index,
        'weights': null,
      });

  @override
  factory AveragePool2D.fromJson(Map<String, dynamic> config) => AveragePool2D(
    config['kernelSize'].cast<int>(),
    strides: config['strides'].cast<int>(),
    padding: Padding.values[config['padding']],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// 1D Average Pooling layer.
///
/// Downsamples the input by taking the average over a spatial pooling window defined by [kernelSize].
/// This layer is always non-[trainable].
class AveragePool1D extends Layer {
  /// The size of the pooling window.
  late final int kernelSize;

  /// The strides of the pooling window.
  late final List<int> strides;

  /// The padding scheme to use (e.g., 'valid' or 'same').
  late final Padding padding;

  static int _maxPoolCounter = 0;

  /// Creates a 1D Average Pooling layer.
  AveragePool1D(this.kernelSize, {int strides = 1, this.padding = Padding.valid, String? name, List<int>? inputShape})
    : super(false) {
    this.strides = [1, strides, 1];

    if (name == null) {
      _maxPoolCounter += 1;
    }
    this.name = name ?? 'AveragePool1D-$_maxPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 2, shapeStructure: 'width, channels');

      this.inputShape = inputShape;
      outputShape = computeBatchlessOutputShapeConv1D(
        inputShape: inputShape,
        filterShape: [kernelSize, inputShape[1], inputShape[1]],
        strides: strides,
        padding: padding,
      );
      _built = true;
    }
  }

  /// Applies the 1D avg pooling operation on the batched [input] Tensor.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [avgPool1DBackprop(input.shape, kernelSize, upstream, strides: strides, padding: padding)];
      };
    }

    return avgPool1D(input, kernelSize, strides: strides, padding: padding);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({
        'type': 'AveragePool1D',
        'kernelSize': kernelSize,
        'strides': strides[1],
        'padding': padding.index,
        'weights': null,
      });

  @override
  factory AveragePool1D.fromJson(Map<String, dynamic> config) => AveragePool1D(
    config['kernelSize'],
    strides: config['strides'],
    padding: Padding.values[config['padding']],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// 2D Global Max Pooling layer
///
/// Computes the maximum value of the entire input tensor
/// spatial dimensions (height and width), resulting in a single feature map
/// value for each channel.
/// This layer is always non-[trainable].
class GlobalMaxPool2D extends Layer {
  static int _glMaxPoolCounter = 0;

  /// Whether to keep the spatial dimensions (height and width) with size 1
  late final bool keepDims;

  /// Creates a GlobalMaxPool2D layer
  GlobalMaxPool2D({this.keepDims = false, String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _glMaxPoolCounter++;
    }
    this.name = name ?? 'GlobalMaxPool2D-$_glMaxPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 3, shapeStructure: 'height, width, channels');

      this.inputShape = inputShape;
      if (keepDims) {
        outputShape = [1, 1, inputShape[2]];
      } else {
        outputShape = [inputShape[2]];
      }
      _built = true;
    }
  }

  /// Applies the 2d global max pooling operation on the batched [input] Tensor.
  ///
  /// The maximum value across the height (axis 1) and width (axis 2) dimensions is computed for each channel.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [
          maxPool2DBackprop(
            input,
            [inputShape[0], inputShape[1]],
            keepDims ? reshape(upstream, [upstream.shape[0], 1, 1, upstream.shape[1]]) : upstream,
            strides: [1, 1, 1, 1],
            padding: Padding.valid,
          ),
        ];
      };
    }

    return reduceMax(input, axis: [1, 2], keepDims: keepDims);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({'type': 'GlobalMaxPool2D', 'keepDims': keepDims, 'weights': null});

  @override
  factory GlobalMaxPool2D.fromJson(Map<String, dynamic> config) => GlobalMaxPool2D(
    keepDims: config['keepDims'],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// 1D Global Max Pooling layer
///
/// Computes the maximum value of the entire input tensor
/// spatial dimension (steps), resulting in a single feature map value for each channel.
/// This layer is always non-[trainable].
class GlobalMaxPool1D extends Layer {
  static int _glMaxPoolCounter = 0;

  /// Whether to keep the spatial dimension with size 1
  late final bool keepDims;

  /// Creates a GlobalMaxPool1D layer
  GlobalMaxPool1D({this.keepDims = false, String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _glMaxPoolCounter++;
    }
    this.name = name ?? 'GlobalMaxPool1D-$_glMaxPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 2, shapeStructure: 'width, channels');

      this.inputShape = inputShape;
      if (keepDims) {
        outputShape = [1, inputShape[1]];
      } else {
        outputShape = [inputShape[1]];
      }
      _built = true;
    }
  }

  /// Applies the 1d global max pooling operation on the batched [input] Tensor.
  ///
  /// The maximum value across the steps dimension (axis 1) is computed for each channel.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [
          maxPool1DBackprop(
            input,
            inputShape[1],
            keepDims ? expandDims(upstream, 1) : upstream,
            strides: [1, 1, 1],
            padding: Padding.valid,
          ),
        ];
      };
    }

    return reduceMax(input, axis: [1], keepDims: keepDims);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({'type': 'GlobalMaxPool1D', 'keepDims': keepDims, 'weights': null});

  @override
  factory GlobalMaxPool1D.fromJson(Map<String, dynamic> config) => GlobalMaxPool1D(
    keepDims: config['keepDims'],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// 2D Global Average Pooling layer
///
/// Computes the average value of the entire input tensor
/// spatial dimensions (height and width), resulting in a single feature map
/// value for each channel.
/// This layer is always non-[trainable].
class GlobalAveragePool2D extends Layer {
  static int _glMaxPoolCounter = 0;
  late final bool keepDims;

  /// Creates a GlobalAveragePool2D layer
  GlobalAveragePool2D({this.keepDims = false, String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _glMaxPoolCounter++;
    }
    this.name = name ?? 'GlobalAveragePool2D-$_glMaxPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 3, shapeStructure: 'height, width, channels');

      this.inputShape = inputShape;
      if (keepDims) {
        outputShape = [1, 1, inputShape[2]];
      } else {
        outputShape = [inputShape[2]];
      }
      _built = true;
    }
  }

  /// Applies the 2d global avg pooling operation on the batched [input] Tensor.
  ///
  /// The avg value across the height (axis 1) and width (axis 2) dimensions is computed for each channel.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [
          avgPool2DBackprop(
            input.shape,
            [inputShape[0], inputShape[1]],
            keepDims ? reshape(upstream, [upstream.shape[0], 1, 1, upstream.shape[1]]) : upstream,
            strides: [1, 1, 1, 1],
            padding: Padding.valid,
          ),
        ];
      };
    }

    return reduceMean(input, axis: [1, 2], keepDims: keepDims);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({'type': 'GlobalAveragePool2D', 'keepDims': keepDims, 'weights': null});

  @override
  factory GlobalAveragePool2D.fromJson(Map<String, dynamic> config) => GlobalAveragePool2D(
    keepDims: config['keepDims'],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// 1D Global Average Pooling layer
///
/// Computes the average value of the entire input tensor
/// spatial dimension (steps), resulting in a single feature map value for each channel.
/// This layer is always non-[trainable].
class GlobalAveragePool1D extends Layer {
  static int _glMaxPoolCounter = 0;

  /// Whether to keep the spatial dimension with size 1
  late final bool keepDims;

  /// Creates a GlobalAveragePool1D layer
  GlobalAveragePool1D({this.keepDims = false, String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _glMaxPoolCounter++;
    }
    this.name = name ?? 'GlobalAveragePool1D-$_glMaxPoolCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, 2, shapeStructure: 'width, channels');

      this.inputShape = inputShape;
      if (keepDims) {
        outputShape = [1, inputShape[1]];
      } else {
        outputShape = [inputShape[1]];
      }
      _built = true;
    }
  }

  /// Applies the 1d global avg pooling operation on the batched [input] Tensor.
  ///
  /// The avg value across the steps dimension (axis 1) is computed for each channel.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      gradient = (Tensor upstream) {
        return [
          avgPool1DBackprop(
            input.shape,
            inputShape[1],
            keepDims ? expandDims(upstream, 1) : upstream,
            strides: [1, 1, 1],
            padding: Padding.valid,
          ),
        ];
      };
    }

    return reduceMean(input, axis: [1], keepDims: keepDims);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({'type': 'GlobalAveragePool1D', 'keepDims': keepDims, 'weights': null});

  @override
  factory GlobalAveragePool1D.fromJson(Map<String, dynamic> config) => GlobalAveragePool1D(
    keepDims: config['keepDims'],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// Layer that reshapes an output to a new shape.
///
/// Primarily used to change the shape of the tensor, maintaining the total number of elements.
/// Reshape always preserves the batch size of input and output tensors.
/// This layer is always non-[trainable].
class Reshape extends Layer {
  static int _reshapeCounter = 0;

  /// Creates a new Reshape layer that transforms the input to the [targetShape].
  Reshape(List<int> targetShape, {String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _reshapeCounter++;
    }
    this.name = name ?? 'Reshape-$_reshapeCounter';

    if (targetShape.any((e) => e <= 0)) {
      throw ArgumentError(
        'The target shape must contain only positive integers, but layer $name received: $targetShape',
        'targetShape',
      );
    }

    outputShape = List.from(targetShape, growable: false);
    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, -1);
      this.inputShape = inputShape;
      _built = true;
    }
  }

  /// Reshapes the batched [input] Tensor to the layer's [outputShape].
  ///
  /// The batch dimension is preserved, so the output shape will be `[batchSize, ...outputShape]`.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    List<int> inputShapeList = input.shape.list;
    if (training) {
      gradient = (Tensor upstream) => [reshape(upstream, inputShapeList)];
    }
    return reshape(input, [inputShapeList[0], ...outputShape]);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({'type': 'Reshape', 'targetShape': outputShape, 'weights': null});

  @override
  factory Reshape.fromJson(Map<String, dynamic> config) => Reshape(
    config['targetShape'],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// A layer that flattens the input tensor into a 1-dimensional output.
///
/// The dimensions excluding the batch dimension are collapsed into a single
/// dimension, preserving the batch size.
/// This layer is always non-[trainable].
class Flatten extends Layer {
  static int _flattenCounter = 0;

  /// Creates a new Flatten layer.
  Flatten({String? name, List<int>? inputShape}) : super(false) {
    if (name == null) {
      _flattenCounter += 1;
    }
    this.name = name ?? 'Flatten-$_flattenCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, -1);
      outputShape = [inputShape.reduce((e1, e2) => e1 * e2)];
      this.inputShape = inputShape;
      _built = true;
    }
  }

  /// Flattens the batched [input] Tensor.
  ///
  /// The batch dimension is preserved, and all other dimensions are merged
  /// into a single dimension of size `input.elements / batchSize`.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    List<int> inputShapeList = input.shape.list;
    if (training) {
      gradient = (Tensor upstream) => [reshape(upstream, inputShapeList)];
    }
    return reshape(input, [inputShapeList[0], outputShape[0]]);
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()..addAll({'type': 'Flatten', 'weights': null});

  @override
  factory Flatten.fromJson(Map<String, dynamic> config) => Flatten(
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}

/// Implements the Dropout regularization technique.
///
/// Dropout randomly sets a fraction of input units to 0 at each update during
/// training time, which helps prevent overfitting.
/// Does nothing during inference (call with `training` set to false).
///
/// This layer is always non-[trainable].
class Dropout extends Layer {
  static int _dropCounter = 0;

  /// "Probability" of the input units to be dropped
  late final double rate;

  /// An integer vector, representing the shape for randomly generated input masking tensor
  late final List<int>? noiseShape;

  /// Seed for the random number generator
  late final int? seed;

  /// Creates a new Dropout layer with a given dropout [rate].
  ///
  /// [rate] must be in the range $[0, 1)$.
  Dropout(this.rate, {this.noiseShape, this.seed, List<int>? inputShape, String? name}) : super(false) {
    if (rate >= 1 || rate < 0) {
      throw ArgumentError('The rate value must be between [0, 1), but received rate: $rate', 'rate');
    }
    if (name == null) {
      _dropCounter += 1;
    }
    this.name = name ?? 'Dropout-$_dropCounter';

    if (inputShape != null) {
      build(inputShape);
    }
  }

  @override
  void build(List<int> inputShape) {
    if (!_built) {
      super.checkBuildingShape(inputShape, -1);
      this.inputShape = inputShape;
      outputShape = inputShape;
      _built = true;
    }
  }

  /// Applies dropout to the [input] Tensor if [training] is true.
  ///
  /// When dropout applied the input is scaled by `1 / (1 - rate)` to preserve the mean value.
  /// In inference mode the input is passed through unchanged.
  @override
  Tensor call(Tensor input, {bool training = false}) {
    if (!_built) {
      build(input.shape.list.sublist(1));
    } else {
      super.checkInputShape(input.shape);
    }

    if (training) {
      final dropResult = dropoutWithMask(input, rate, noiseShape: noiseShape, seed: seed);
      final mask = dropResult[1];
      gradient = (Tensor upstream) => [upstream * mask * (1 / (1 - rate))];
      return dropResult[0];
    }
    return input;
  }

  @override
  List<Tensor> get trainableParams => [];

  @override
  void setWeights({List<Tensor>? nonTrainableWeights, List<Tensor>? trainableWeights}) {}

  @override
  Map<String, dynamic> toJson({bool withWeights = true}) =>
      super.toJson()
        ..addAll({'type': 'Dropout', 'rate': rate, 'noiseShape': noiseShape, 'seed': seed, 'weights': null});

  @override
  factory Dropout.fromJson(Map<String, dynamic> config) => Dropout(
    config['rate'],
    noiseShape: config['noiseShape'],
    seed: config['seed'],
    name: config['name'],
    inputShape: config['inputShape'].cast<int>(),
  );
}
