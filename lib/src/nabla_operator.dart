import 'layer.dart';
import 'matrix.dart';

/// Class of differential operator nabla: `∇f = grad(f)`
/// 
/// Class is used in the training process of [NeuralNetwork] for gradients calculations
class NablaOperator {
  /// Compute gradients for weigths and biases of Layers in [trainableLayers] with given Loss function derivative [dLoss]
  /// 
  /// Return [List] of `gradients` and `deltas` for each [Layer]
  static List<List<Matrix>> gradients(List<Layer> trainableLayers, Matrix dLoss) {
    final len = trainableLayers.length - 1;

    List<List<Matrix>> deltasAndGradients = List<List<Matrix>>
      .generate(len+1, (i) => List<Matrix>.generate(2, (j) => Matrix.zero(n: 0, m: 0)));

    if (trainableLayers[len].activatedDerivativeBuffer!.isVector) {
      /// Output [Layer] deltas as Hadamard product of d(loss)/d(output) and d(activation_function)/d(data)
      deltasAndGradients[0][0] = dLoss%trainableLayers[len].activatedDerivativeBuffer!;
    }
    else {
      /// Output [Layer] deltas as sum from i=0 to n-1 of Hadamard product of d(loss)/d(output_i) and d(activation_function_i)/d(data_i)
      deltasAndGradients[0][0] = Matrix.column(List<double>
        .generate(trainableLayers[len].activatedDerivativeBuffer!.n,
        (index) => (trainableLayers[len].activatedDerivativeBuffer!.getColumn(index)%dLoss).reduceSum())
      );
    }

    /// Output [Layer] gradients as matrix multiplication of delta and input data for the [Layer]
      deltasAndGradients[0][1] = deltasAndGradients[0][0]*trainableLayers[len].inputDataBuffer!.T;

    /// For other [Layer]s gradients is computed as matrix multipliction of deltas and input data of the current [Layer]
    for (int i = len-1; i >= 0; i -= 1) {
      var tempSumPrev = trainableLayers[i+1].w!.T*deltasAndGradients[len-i-1][0]; // sum of previous deltas * weights
      if (trainableLayers[i].activatedDerivativeBuffer!.isVector) {
        deltasAndGradients[len-i][0] = 
          tempSumPrev%trainableLayers[i].activatedDerivativeBuffer!; // d(activation_function)/d(data)
      }
      else {
        deltasAndGradients[len-i][0] = Matrix.column(List<double>
        .generate(trainableLayers[i].activatedDerivativeBuffer!.n,
          (index) => (tempSumPrev%trainableLayers[i].activatedDerivativeBuffer!.getColumn(index)).reduceSum())
        );
      }                  
      deltasAndGradients[len-i][1] = 
      deltasAndGradients[len-i][0]*trainableLayers[i].inputDataBuffer!.T;  // delta_i * input_i
    }
    return deltasAndGradients.reversed.toList();
  }
}