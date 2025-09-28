import 'package:loredart_tensor/loredart_tensor.dart';

/// The base class for computational nodes, i.e., objects that can be backpropagated
abstract class ComputationalNode {
  List<Tensor> Function(Tensor)? gradient;
}

/// The base class for neural network modules.
///
/// Module is a named computational node that has 0 or more trainable params and can update them.
abstract class Module implements ComputationalNode {
  /// Name of the module
  late final String name;

  /// List of trainable (or updatable) parameters of module
  List<Tensor> get trainableParams;

  /// Creates instance of Module with given [name].
  Module({required this.name});

  /// Apply the logic of updating [trainableParams] of the module
  void updateTrainableParams(List<Tensor> updatedParams);

  /// Constructs and returns new module from its [config].
  Module.fromJson(Map<String, dynamic> config);

  /// Returns config of the module as JSON-serializable Map.
  Map<String, dynamic> toJson({bool withWeights = true});
}
