import 'dart:math' as math;
import 'package:loredart_nn/src/utils/deserialization_utils.dart';
import 'package:loredart_tensor/loredart_tensor.dart';

enum FanMode { fanIn, fanOut, fanAvg }

enum Distribution { truncatedNormal, unTruncatedNormal, uniform }

/// The base class for initializers.
abstract class Initializer {
  /// Creates and returns Tensor of given [shape] and [dType] according to initializer logic.
  Tensor call(List<int> shape, DType dType);

  factory Initializer.fromJson(Map<String, dynamic> config) => deserializeInitializer(config.remove('type'), config);
  Map<String, dynamic> toJson();
}

/// Initializer that generates tensors with 1
///
/// Example:
/// ```dart
/// final init = Ones();
/// final values = init([2, 2], DType.int32);
/// print(values);
/// // Tensor(shape: [2, 2], values:
/// //  [[1, 1]
/// //  [1, 1]], dType: int32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: Ones()),
///   // or with Initializers.ones
///   ...
/// ], ...);
/// ```
class Ones implements Initializer {
  const Ones();
  @override
  Tensor call(shape, dType) {
    return Tensor.ones(shape, dType: dType);
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'Ones'};
}

/// Initializer that generates tensors with 0.
///
/// Example:
/// ```dart
/// final init = Zeros();
/// final values = init([2, 2], DType.int32);
/// print(values);
/// // Tensor(shape: [2, 2], values:
/// //  [[0, 0]
/// //  [0, 0]], dType: int32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: Zeros()),
///   // or with Initializers.zeros
///   ...
/// ], ...);
/// ```
class Zeros implements Initializer {
  const Zeros();
  @override
  Tensor call(shape, dType) {
    return Tensor.zeros(shape, dType: dType);
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'Zeros'};
}

/// Initializer that generates identity matrices.
///
/// This implementation can work with high-dimensional and tensors,
/// but it's not recommended
///
/// Example:
/// ```dart
/// final init = Identity();
/// final values = init([1, 2, 3], DType.int32);
/// print(values);
/// // Tensor(shape: [1, 2, 3], values:
/// //  [[[1 0 0]
/// //  [0 1 0]]], dType: int32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: Identity()),
///   // or with Initializers.identity
///   ...
/// ], ...);
/// ```
class Identity implements Initializer {
  const Identity();
  @override
  Tensor call(shape, dType) {
    int len = shape.length;
    return Tensor.eye(shape[len - 2], numCols: shape[len - 1], batchShape: shape.sublist(0, len - 2), dType: dType);
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'Identity'};
}

/// Initializer that generates tensors with constant value.
///
/// Example:
/// ```dart
/// final init = Constant(value: 0.13);
/// final values = init([2, 2], DType.float32);
/// print(values);
/// // Tensor(shape: [2, 2], values:
/// //  [[0.13, 0.13]
/// //  [0.13, 0.13]], dType: float32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: Constant(value: 1e-2)),
///   ...
/// ], ...);
/// ```
class Constant implements Initializer {
  /// Value to fill the tensors
  late final num value;

  Constant({this.value = 0.0});

  @override
  Tensor call(shape, dType) {
    return Tensor.fill(shape, value, dType: dType);
  }

  @override
  Map<String, dynamic> toJson() => {'type': 'Constant', 'value': value};
}

/// Initializer that adapts its scale to tensors' shapes.
///
/// Generation depends on the [distribution]:
///
/// - with [Distribution.truncatedNormal] or [Distribution.unTruncatedNormal]:
/// tensor is sampled from corresponding distr with zero mean and `std = sqrt([scale] / n)`,
/// where `n` computed according to the [mode].
///
/// - with [Distribution.uniform] tensors sampled from uniform distr within [-k, k],
/// where `k = sqrt(3 * [scale] / n)`
class VarianceScaler implements Initializer {
  late final double scale;
  late final FanMode mode;
  late final Distribution distribution;
  int? seed;

  VarianceScaler({required this.scale, required this.mode, required this.distribution, this.seed});

  @override
  Tensor call(List<int> shape, DType dType) {
    final List<int> fans = computeFans(shape);
    double scalingFactor = scale;
    if (mode == FanMode.fanIn) {
      scalingFactor /= math.max(1, fans[0]);
    } else if (mode == FanMode.fanOut) {
      scalingFactor /= math.max(1, fans[1]);
    } else {
      scalingFactor /= math.max(1, (fans[0] + fans[1]) / 2);
    }

    if (distribution == Distribution.truncatedNormal) {
      final double std = math.sqrt(scalingFactor);
      return truncatedNormal(shape, std: std, dType: dType);
    } else if (distribution == Distribution.unTruncatedNormal) {
      final double std = math.sqrt(scalingFactor);
      return normal(shape, std: std, dType: dType);
    } else {
      final double limit = math.sqrt(3 * scalingFactor);
      return uniform(shape, min: -limit, max: limit, dType: dType);
    }
  }

  /// Computes the number of input and output units for a weight shape.
  ///
  /// Returns [List<int>] where 1st element is number of input and 2nd is number of output units.
  static List<int> computeFans(List<int> shape) {
    if (shape.isEmpty) {
      return [1, 1];
    } else if (shape.length == 1) {
      return [shape[0], shape[0]];
    } else if (shape.length == 2) {
      return List.from(shape, growable: false);
    } else {
      return [shape[shape.length - 2], shape[shape.length - 1]];
    }
  }

  @override
  Map<String, dynamic> toJson() => {
    'type': 'VarianceScaler',
    'scale': scale,
    'mode': mode.index,
    'distribution': distribution.index,
  };
}

/// The He uniform variance scaler initializer.
///
/// This samples tensor values from a uniform distribution within [-k, k],
/// where `k = sqrt(6 / fan_in)` (fan_in is the number of input units in the weight tensor).
///
/// Example:
/// ```dart
/// final init = HeUniform();
/// final values = init([2, 2], DType.float32);
/// print(values);
/// // Tensor(shape: [2, 2], values:
/// //  [[..., ...]
/// //   [..., ...]], dType: float32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: HeUniform()),
///   // or with Initializers.heUniform
///   ...
/// ], ...);
/// ```
class HeUniform extends VarianceScaler {
  HeUniform({int? seed}) : super(scale: 2.0, mode: FanMode.fanIn, distribution: Distribution.uniform, seed: seed);

  @override
  Map<String, dynamic> toJson() => {'type': 'HeUniform', 'seed': seed};
}

/// The He normal variance scaler initializer.
///
/// This samples tensor values from a truncated normal distribution
/// with `std = sqrt(2 / fan_in)` (fan_in is the number of input units in the weight tensor).
///
/// Example:
/// ```dart
/// final init = HeNomral();
/// final values = init([2, 2], DType.float32);
/// print(values);
/// // Tensor(shape: [2, 2], values:
/// //  [[..., ...]
/// //   [..., ...]], dType: float32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: HeNomral()),
///   // or with Initializers.heNormal
///   ...
/// ], ...);
/// ```
class HeNormal extends VarianceScaler {
  HeNormal({int? seed})
    : super(scale: 2.0, mode: FanMode.fanIn, distribution: Distribution.truncatedNormal, seed: seed);

  @override
  Map<String, dynamic> toJson() => {'type': 'HeNormal', 'seed': seed};
}

/// The Glorot uniform variance scaler initializer.
///
/// This samples tensor values from a uniform distribution within [-k, k],
/// where `k = sqrt(6 / (fan_in + fan_out))`
/// (fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor).
///
/// Example:
/// ```dart
/// final init = GlorotUniform();
/// final values = init([2, 2], DType.float32);
/// print(values);
/// // Tensor(shape: [2, 2], values:
/// //  [[..., ...]
/// //   [..., ...]], dType: float32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: GlorotUniform()),
///   // or with Initializers.glorotUniform
///   ...
/// ], ...);
/// ```
class GlorotUniform extends VarianceScaler {
  GlorotUniform({int? seed}) : super(scale: 1.0, mode: FanMode.fanAvg, distribution: Distribution.uniform, seed: seed);

  @override
  Map<String, dynamic> toJson() => {'type': 'GlorotUniform', 'seed': seed};
}

/// The Glorot normal variance scaler initializer.
///
/// This samples tensor values from a truncated normal distribution
/// with `std = sqrt(2 / (fan_in + fan_out))`
/// (fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor).
///
/// Example:
/// ```dart
/// final init = GlorotNomral();
/// final values = init([2, 2], DType.float32);
/// print(values);
/// // Tensor(shape: [2, 2], values:
/// //  [[..., ...]
/// //   [..., ...]], dType: float32)
/// ```
/// As an argument for layer:
/// ```dart
/// Model model = Model([
///   Dense(13, kernelInitializer: GlorotNomral()),
///   // or with Initializers.glorotNormal
///   ...
/// ], ...);
/// ```
class GlorotNormal extends VarianceScaler {
  GlorotNormal({int? seed})
    : super(scale: 1.0, mode: FanMode.fanAvg, distribution: Distribution.truncatedNormal, seed: seed);

  @override
  Map<String, dynamic> toJson() => {'type': 'GlorotNormal', 'seed': seed};
}

/// Collection of initializers.
class Initializers {
  /// The ones initializer.
  static Initializer get ones => Ones();

  /// The identity initializer.
  static Initializer get identity => Identity();

  /// The zero initializer.
  static Initializer get zeros => Zeros();

  /// The He normal variance scaler initializer.
  static Initializer get heNormal => HeNormal();

  /// The He uniform variance scaler initializer.
  static Initializer get heUniform => HeUniform();

  /// The Glorot normal variance scaler initializer.
  static Initializer get glorotNormal => GlorotNormal();

  /// The Glorot uniform variance scaler initializer.
  static Initializer get glorotUniform => GlorotUniform();
}
