import 'layer.dart';
import 'matrix.dart';

/// Base super-class for any optimizer of [NeuralNetwork]
abstract class Optimizer {
  /// The learning rate
  late double learningRate;
  /// The learning rate for biases
  double biasLearningRate = 0.01;
  Optimizer({required this.learningRate}) : assert(learningRate < 1 && learningRate > 0);
  /// apply [gradients] to the [layers] through [Optimizer]'s logic
  void applyGradients(List<List<Matrix>> gradients, List<Layer> layers, [dynamic parametr]);
}

/// Stochatic Gradient Descent optimizer with momentum parameter
/// 
/// Gradients applying depends on momentum value:
/// - momentum = 0 then:
/// ```
/// layer.w = layer.w - gradients.scaled(learningRate)
/// ```
/// - momentum != 0 then:
/// ```
/// prev = prev.scaled(momentum) - gradients.scaled(learningRate);
/// layer.w = layer.w - prev;
/// ```
class SGD extends Optimizer {
  late double momentum;
  List<Matrix>? _previuosDelta;
  SGD({double learningRate = 0.05, this.momentum = 0}) : assert(momentum <= 1 && momentum >= 0), super(learningRate: learningRate);

  @override
  void applyGradients(gradients, layers, [parametr]) {
    if (momentum == 0) {
      for (int i = 0; i < layers.length; i += 1) {
        if (layers[i].useBiases) {
          layers[i].b = layers[i].b! - gradients[i][0].scaled(biasLearningRate);
        }
        layers[i].w = layers[i].w! - gradients[i][1].scaled(learningRate);
      }
    }
    else if (_previuosDelta == null) {
      _previuosDelta = List<Matrix>.filled(gradients.length, Matrix.zero(n: 0, m: 0));
      for (int i = 0; i < layers.length; i += 1) {
        if (layers[i].useBiases) {
          layers[i].b = layers[i].b! - gradients[i][0].scaled(biasLearningRate);
        }
        _previuosDelta![i] = gradients[i][1]..scale(learningRate);
        layers[i].w = layers[i].w! - gradients[i][1];
      }
    }
    else {
      for (int i = 0; i < layers.length; i += 1) {
        if (layers[i].useBiases) {
          layers[i].b = layers[i].b! - gradients[i][0].scaled(biasLearningRate);
        }
        _previuosDelta![i] = _previuosDelta![i].scaled(momentum) - gradients[i][1].scaled(learningRate);
        layers[i].w = layers[i].w! + _previuosDelta![i];
      }
    }
  }

  @override
  String toString() => "SGD (lr: $learningRate, mn: $momentum)";
}