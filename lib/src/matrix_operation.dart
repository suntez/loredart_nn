import 'matrix.dart';

/// The class of set of the static methods of different [Matrix] operations
class MatrixOperation {

  /// Return matrix multiplication of [a] and [b]
  /// 
  /// Throw an `exception` if dimension's condition not met 
  static Matrix multiplication(Matrix a, Matrix b) {
    if (a.m == b.n) {
      var resultMatrix = Matrix.zero(n: a.n, m: b.m);
      for (int i = 0; i < resultMatrix.n; i += 1) {
        for (int j = 0; j < resultMatrix.m; j += 1) {
          double sum = 0;
          for (int k = 0; k < a.m; k += 1) {
            sum += a[i][k] * b[k][j];
          }
          resultMatrix[i][j] = sum;
        }
      }
      return resultMatrix;
    }
    else {
      throw Exception('Dimensions error: A.m != B.n');
    }
  }

  /// Return Hadamard product of [a] and [b],
  /// 
  /// Example:
  /// ```dart
  /// final a = Matrix.fromLists([[1,2], [2,1]]); // [[1, 2]
  ///                                             // [2, 1]]
  /// final b = Matrix.fromLists([[1,2], [3,4]]); // [[1, 2]
  ///                                             // [3, 4]]
  /// final matrix = MatrixOperation.hadamardProduct(a, b); // or a%b
  /// print(matrix);
  /// // Output:
  /// // matrix 2⨯2
  /// // [[1.0, 4.0]
  /// // [6.0, 4.0]]
  /// ```
  ///
  /// Throw an `exception` if dimension's condition not met
  static Matrix hadamardProduct(Matrix a, Matrix b) {
    if (a.n == b.n && a.m == b.m) {
      var resultMatrix = Matrix.zero(n: a.n, m: b.m);
      for (int i = 0; i < resultMatrix.n; i += 1) {
        for (int j = 0; j <  resultMatrix.m; j += 1) {
          resultMatrix[i][j] = a[i][j] * b[i][j];
        }
      }
      return resultMatrix;
    }
    else {
      throw Exception('Dimensions error: dim(A) != dim(B)');
    }
  }

  /// Return [Matrix] created as binded [a] and [b] by columns.
  /// Elements of [a] set first, and elements of [b] second
  /// 
  /// Example:
  /// ```dart
  /// final a = Matrix.column([0,0,0]);
  /// final b = Matrix.column([1,1,1]);
  /// final matrix = MatrixOperation.columnBind(a, b);
  /// print(matrix);
  /// // Output:
  /// // matrix 3⨯2
  /// // [[0.0, 1.0]
  /// // [0.0, 1.0]
  /// // [0.0, 1.0]]
  /// ``` 
  /// 
  /// Throw an `exception` if dimension's condition not met
  static Matrix columnBind(Matrix a, Matrix b) {
    if (a.n == b.n) {
      var resultMatrix = Matrix.zero(n: a.n, m:a.m + b.m);
      for (int i = 0; i < resultMatrix.n; i += 1) {
        for (int j = 0; j <  resultMatrix.m; j += 1) {
          resultMatrix[i][j] = j < a.m ? a[i][j] : b[i][j-a.m];
        }
      }
      return resultMatrix;
    }
    else {
      throw Exception('Dimensions error: A.n != B.n');
    }
  }

  /// Return [Matrix] created as binded [a] and [b] by rows.
  /// Elements of [a] set first, and elements of [b] second
  /// 
  /// Example:
  /// ```dart
  /// final a = Matrix.row([0,0,0]);
  /// final b = Matrix.row([1,1,1]);
  /// final matrix = MatrixOperation.rowBind(a, b);
  /// print(matrix);
  /// // Output:
  /// // matrix 2⨯3
  /// // [[0.0, 0.0, 0.0]
  /// // [1.0, 1.0, 1.0]]
  /// ```
  ///
  /// Throw an `exception` if dimension's condition not met
  static Matrix rowBind(Matrix a, Matrix b) {
    if (a.m == b.m) {
      var resultMatrix = Matrix.zero(n: a.n + b.n, m:a.m);
      for (int i = 0; i < resultMatrix.n; i += 1) {
        for (int j = 0; j <  resultMatrix.m; j += 1) {
          resultMatrix[i][j] = i < a.n ? a[i][j] : b[i-a.n][j];
        }
      }
      return resultMatrix;
    }
    else {
      throw Exception('Dimensions error: A.n != B.n');
    }
  }

  /// Return elementwise addition of [a] and [b]
  /// 
  /// Throw an `exception` if dimension's condition not met
  static Matrix addition(Matrix a, Matrix b) {
    if (a.n == b.n && a.m == b.m) {
      var resultMatrix = Matrix.zero(n: a.n, m: b.m);
      for (int i = 0; i < resultMatrix.n; i += 1) {
        for (int j = 0; j <  resultMatrix.m; j += 1) {
          resultMatrix[i][j] = a[i][j] + b[i][j];
        }
      }
      return resultMatrix;
    }
    else {
      throw Exception('Dimensions error: A.shape != B.shape');
    }
  }

  /// Return elementwise subtraction of [a] and [b]
  /// 
  /// Throw an `exception` if dimension's condition not met
  static Matrix subtraction(Matrix a, Matrix b) {
    if (a.n == b.n && a.m == b.m) {
      var resultMatrix = Matrix.zero(n: a.n, m: b.m);
      for (int i = 0; i < resultMatrix.n; i += 1) {
        for (int j = 0; j <  resultMatrix.m; j += 1) {
          resultMatrix[i][j] = a[i][j] - b[i][j];
        }
      }
      return resultMatrix;
    }
    else {
      throw Exception('Dimensions error: A.shape != B.shape');
    }
  }

  /// Return transpose of [matrix] 
  static Matrix transposition(Matrix matrix) {
    var resultMatrix = Matrix.zero(n: matrix.m, m: matrix.n);
    for (int i = 0; i < resultMatrix.n; i += 1) {
      for (int j = 0; j <  resultMatrix.m; j += 1) {
        resultMatrix[i][j] = matrix[j][i];
      }
    }
    return resultMatrix;
  }

  /// Return copy of [matrix] where elements are scaled by [scalar]
  static Matrix scalarMultiplication(Matrix matrix, double scalar) {
    var resultMatrix = Matrix.zero(n: matrix.n, m: matrix.m);
    for (int i = 0; i < resultMatrix.n; i += 1) {
      for (int j = 0; j <  resultMatrix.m; j += 1) {
        resultMatrix[i][j] = matrix[i][j] * scalar;
      }
    }
    return resultMatrix;
  }

  /// Return copy of [matrix] where elements are added by [scalar]
  static Matrix scalarAddition(Matrix matrix, double scalar) {
    var resultMatrix = Matrix.zero(n: matrix.n, m: matrix.m);
    for (int i = 0; i < resultMatrix.n; i += 1) {
      for (int j = 0; j <  resultMatrix.m; j += 1) {
        resultMatrix[i][j] = matrix[i][j] + scalar;
      }
    }
    return resultMatrix;
  }

  /// Return a copy of [matrix] where [function] was applied to every element
  static Matrix apply(Matrix matrix, double Function(double) function) {
    var resultMatrix = Matrix.zero(n: matrix.n, m: matrix.m);
    for (int i = 0; i < resultMatrix.n; i += 1) {
      for (int j = 0; j < resultMatrix.m; j += 1) {
        resultMatrix[i][j] = function(matrix[i][j]);
      }
    }
    return resultMatrix;
  }
}