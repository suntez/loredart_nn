import 'package:loredart_nn/loredart_nn.dart';

/// Class of differential operator nabla: `∇f = grad(f)`
///
/// Class is used in the training process of [NeuralNetwork] for calculations of mean gradients per (mini)batch
class NablaOperator {
  /// Compute gradients for weigths and biases of Layers in [trainableLayers] with given Loss function derivative [dLoss]
  ///
  /// Return [List] of mean `gradients` and `deltas` for each [Layer] based on the (mini)batch buffered data
  static List<List<Matrix>> gradients(
      List<Layer> trainableLayers, Matrix dLoss) {
    final len = trainableLayers.length - 1;

    List<List<Matrix>> meanDeltasAndGradients = List<List<Matrix>>.generate(
        len + 1,
        (i) => List<Matrix>.generate(2, (j) => Matrix.zero(n: 0, m: 0)));

    if (trainableLayers[len].activatedDerivativeBuffer!.length == 1) {
      /// Output [Layer] deltas is a Hadamard product of d(loss)/d(activation_function) and d(activation_function)/d(data)
      meanDeltasAndGradients[0][0] =
          (dLoss % trainableLayers[len].activatedDerivativeBuffer![0]);
    } else {
      /// With Jacobian:
      /// Output [Layer] deltas as sum from i=0 to n-1 of Hadamard product of d(loss)/d(activation_function_i) and d(activation_function_i)/d(data_i)
      Matrix tempEvalMatrix = Matrix.zero(n: 0, m: 0);
      for (int i = 0;
          i < trainableLayers[len].activatedDerivativeBuffer!.length;
          i += 1) {
        if (tempEvalMatrix.m == 0 && tempEvalMatrix.n == 0) {
          tempEvalMatrix = (MatrixOperation.hadamardProductToEachColumn(
                  trainableLayers[len].activatedDerivativeBuffer![i],
                  dLoss.getColumn(i)))
              .reduceSumByAxis(1);
        } else {
          tempEvalMatrix = MatrixOperation.columnBind(
              tempEvalMatrix,
              (MatrixOperation.hadamardProductToEachColumn(
                      trainableLayers[len].activatedDerivativeBuffer![i],
                      dLoss.getColumn(i)))
                  .reduceSumByAxis(1));
        }
      }
      meanDeltasAndGradients[0][0] = tempEvalMatrix;
    }

    /// Output [Layer] gradients is a matrix multiplication of delta and input^T data for the [Layer] (mean for a (mini)batch)
    meanDeltasAndGradients[0][1] =
        (meanDeltasAndGradients[0][0] * trainableLayers[len].inputDataBuffer!.T)
            .scaled(1 / trainableLayers[len].inputDataBuffer!.m);

    /// For other (hidden) [Layer]s gradients are computed as matrix multipliction of deltas and input^T data of the current [Layer]
    for (int i = len - 1; i >= 0; i -= 1) {
      // sum of previous deltas * weights
      var tempSumPrev =
          trainableLayers[i + 1].w!.T * meanDeltasAndGradients[len - i - 1][0];

      if (trainableLayers[i].activatedDerivativeBuffer!.length == 1) {
        // d(activation_function)/d(data)
        meanDeltasAndGradients[len - i][0] =
            tempSumPrev % trainableLayers[i].activatedDerivativeBuffer![0];
      } else {
        // For Jacobian
        Matrix tempEvalMatrix = Matrix.zero(n: 0, m: 0);
        for (int i = 0;
            i < trainableLayers[i].activatedDerivativeBuffer!.length;
            i += 1) {
          if (tempEvalMatrix.m == 0 && tempEvalMatrix.n == 0) {
            tempEvalMatrix = (MatrixOperation.hadamardProductToEachColumn(
                    trainableLayers[i].activatedDerivativeBuffer![i],
                    tempSumPrev.getColumn(i)))
                .reduceSumByAxis(1);
          } else {
            tempEvalMatrix = MatrixOperation.columnBind(
                tempEvalMatrix,
                (MatrixOperation.hadamardProductToEachColumn(
                        trainableLayers[i].activatedDerivativeBuffer![i],
                        tempSumPrev.getColumn(i)))
                    .reduceSumByAxis(1));
          }
        }
        meanDeltasAndGradients[len - i][0] = tempEvalMatrix;
      }
      meanDeltasAndGradients[len - i][1] = (meanDeltasAndGradients[len - i][0] *
              trainableLayers[i].inputDataBuffer!.T)
          .scaled(1 / trainableLayers[i].inputDataBuffer!.m);
      meanDeltasAndGradients[len - i - 1][0] =
          meanDeltasAndGradients[len - i - 1][0].reduceMeanByAxis(0);
    }
    meanDeltasAndGradients[len][0] =
        meanDeltasAndGradients[len][0].reduceMeanByAxis(0);
    return meanDeltasAndGradients.reversed.toList();
  }
}
