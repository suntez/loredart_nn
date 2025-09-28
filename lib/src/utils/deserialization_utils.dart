import 'package:loredart_tensor/loredart_tensor.dart';

import '../modules/activations.dart';
import '../modules/layers.dart';
import '../nn/initializer.dart';
import '../metrics_and_losses/losses.dart';
import '../metrics_and_losses/metrics.dart';
import '../optimizers/optimizers.dart';

List<Tensor>? deserializeWeights(dynamic weights) {
  if (weights is List) {
    return [for (Map<String, dynamic> data in weights) tensorFromJson(data)];
  }
  return null;
}

Metric deserializeMetric(String type, Map<String, dynamic> parameters) {
  if (type == "MeanAbsoluteErrorMetric") {
    return MeanAbsoluteErrorMetric(name: parameters['name']);
  } else if (type == "MeanSquaredErrorMetric") {
    return MeanSquaredErrorMetric(name: parameters['name']);
  } else if (type == "RootMeanSquaredErrorMetric") {
    return RootMeanSquaredErrorMetric(name: parameters['name']);
  } else if (type == "CategoricalAccuracyMetric") {
    return CategoricalAccuracyMetric(name: parameters['name']);
  } else if (type == "SparseCategoricalAccuracyMetric") {
    return SparseCategoricalAccuracyMetric(name: parameters['name']);
  } else if (type == "BinaryAccuracyMetric") {
    return BinaryAccuracyMetric(name: parameters['name'], threshold: parameters['threshold']);
  } else if (type == "CategoricalCrossentropyMetric") {
    return CategoricalCrossentropyMetric(name: parameters['name'], fromLogits: parameters['fromLogits']);
  } else if (type == "SparseCategoricalCrossentropyMetric") {
    return SparseCategoricalCrossentropyMetric(name: parameters['name'], fromLogits: parameters['fromLogits']);
  } else if (type == "BinaryCrossentropyMetric") {
    return BinaryCrossentropyMetric(name: parameters['name'], fromLogits: parameters['fromLogits']);
  } else if (type == "LogCoshErrorMetric") {
    return LogCoshErrorMetric(name: parameters['name']);
  } else if (type == "KLDivergenceMetric") {
    return KLDivergenceMetric(name: parameters['name']);
  } else {
    throw ArgumentError("Unknown metric $type cannot be deserialized");
  }
}

Loss deserializeLoss(String type, Map<String, dynamic> parameters) {
  if (type == "MeanAbsoluteError") {
    return MeanAbsoluteError(reduction: Reduction.values[parameters['reduction']]);
  } else if (type == "MeanSquaredError") {
    return MeanSquaredError(reduction: Reduction.values[parameters['reduction']]);
  } else if (type == "CategoricalCrossentropy") {
    return CategoricalCrossentropy(
      reduction: Reduction.values[parameters['reduction']],
      fromLogits: parameters['fromLogits'],
    );
  } else if (type == "SparseCategoricalCrossentropy") {
    return SparseCategoricalCrossentropy(
      reduction: Reduction.values[parameters['reduction']],
      fromLogits: parameters['fromLogits'],
    );
  } else if (type == "BinaryCrossentropy") {
    return BinaryCrossentropy(
      reduction: Reduction.values[parameters['reduction']],
      fromLogits: parameters['fromLogits'],
    );
  } else if (type == "LogCoshError") {
    return LogCoshError(reduction: Reduction.values[parameters['reduction']]);
  } else {
    throw ArgumentError("Unknown loss $type cannot be deserialized");
  }
}

Initializer deserializeInitializer(String type, Map<String, dynamic> parameters) {
  if (type == "Ones") {
    return Ones();
  } else if (type == "Zeros") {
    return Zeros();
  } else if (type == "HeNormal") {
    return HeNormal(seed: parameters['seed']);
  } else if (type == "HeUniform") {
    return HeUniform(seed: parameters['seed']);
  } else if (type == "GlorotNormal") {
    return GlorotNormal(seed: parameters['seed']);
  } else if (type == "GlorotUniform") {
    return GlorotUniform(seed: parameters['seed']);
  } else if (type == "VarianceScaler") {
    return VarianceScaler(
      scale: parameters['scale'],
      mode: FanMode.values[parameters['mode']],
      distribution: Distribution.values[parameters['distribution']],
      seed: parameters['seed'],
    );
  } else {
    throw ArgumentError("Unknown initializer $type cannot be deserialized");
  }
}

Activation deserializeActivation(String type, Map<String, dynamic> parameters) {
  if (type == "LeakyReLU") {
    return LeakyReLU(alpha: parameters['alpha']);
  } else if (type == "ELU") {
    return ELU(alpha: parameters['alpha']);
  } else if (type == "Linear") {
    return Linear();
  } else if (type == "Softplus") {
    return Softplus();
  } else if (type == "Softminus") {
    return Softminus();
  } else if (type == "Softmax") {
    return Softmax();
  } else if (type == "Sigmoid") {
    return Sigmoid();
  } else if (type == "ReLU") {
    return ReLU();
  } else if (type == "Tanh") {
    return Tanh();
  } else if (type == "Swish") {
    return Swish();
  } else {
    throw ArgumentError("Unknown activation $type cannot be deserialized");
  }
}

Layer deserializeLayer(String type, Map<String, dynamic> parameters) {
  if (type == "Dense") {
    return Dense.fromJson(parameters);
  } else if (type == "ActivationLayer") {
    return ActivationLayer.fromJson(parameters);
  } else if (type == "LayerNormalization") {
    return LayerNormalization.fromJson(parameters);
  } else if (type == "Rescale") {
    return Rescale.fromJson(parameters);
  } else if (type == "Reshape") {
    return Reshape.fromJson(parameters);
  } else if (type == "Flatten") {
    return Flatten.fromJson(parameters);
  } else if (type == "Conv2D") {
    return Conv2D.fromJson(parameters);
  } else if (type == "Conv1D") {
    return Conv1D.fromJson(parameters);
  } else if (type == "MaxPool2D") {
    return MaxPool2D.fromJson(parameters);
  } else if (type == "MaxPool1D") {
    return MaxPool1D.fromJson(parameters);
  } else if (type == "GlobalMaxPool2D") {
    return GlobalMaxPool2D.fromJson(parameters);
  } else if (type == "GlobalMaxPool1D") {
    return GlobalMaxPool1D.fromJson(parameters);
  } else if (type == "AveragePool2D") {
    return AveragePool2D.fromJson(parameters);
  } else if (type == "AveragePool1D") {
    return AveragePool1D.fromJson(parameters);
  } else if (type == "GlobalAveragePool2D") {
    return GlobalAveragePool2D.fromJson(parameters);
  } else if (type == "GlobalAveragePool1D") {
    return GlobalAveragePool1D.fromJson(parameters);
  } else if (type == "Dropout") {
    return Dropout.fromJson(parameters);
  } else {
    throw ArgumentError("Unknown layer $type cannot be deserialized");
  }
}

Optimizer deserializeOptimizer(String type, Map<String, dynamic> parameters) {
  if (type == "SGD") {
    return SGD(
      learningRate: parameters['learningRate'],
      momentum: parameters['momentum'],
      weightDecay: parameters['weightDecay'],
      name: parameters['name'],
    );
  } else if (type == "Lion") {
    return Lion(
      learningRate: parameters['learningRate'],
      beta1: parameters['beta1'],
      beta2: parameters['beta2'],
      name: parameters['name'],
    );
  } else if (type == "Adam") {
    return Adam(
      learningRate: parameters['learningRate'],
      beta1: parameters['beta1'],
      beta2: parameters['beta2'],
      epsilon: parameters['epsilon'],
      name: parameters['name'],
    );
  } else if (type == "AdaBelief") {
    return AdaBelief(
      learningRate: parameters['learningRate'],
      beta1: parameters['beta1'],
      beta2: parameters['beta2'],
      epsilon: parameters['epsilon'],
      name: parameters['name'],
    );
  } else {
    throw ArgumentError("Unknown optimizer $type cannot be deserialized");
  }
}
