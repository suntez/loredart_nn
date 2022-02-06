import 'package:normal/normal.dart';

import 'matrix_operation.dart';

/// The class for (1,1) tensor of rank 2 (matrix)
/// 
/// ### Constructors:
/// ```dart
/// Matrix({int n, int m, int? seed}) /// create n⨯m matrix with He-Normal initialized values
/// 
/// Matrix.column(List<double> column) /// create [column.length]⨯1 column-vector matrix
/// 
/// Matrix.row(List<double> row) /// create 1⨯[column.length] row-vector matrix
/// 
/// Matrix.fromLists(List<List<double>> matrix) /// create matrix from given values in the [matrix]
///
/// Matrix.zero({int n, int m}) /// create n⨯m matrix full of zeros
///
/// Matrix.identity({int order}) /// create [order]⨯[order] identity square matrix
///
/// Matrix.diag({List<double> diag}) /// create [diag.length]⨯[diag.length] square matrix with diagonal elements from [diag]
///
/// Matrix.reshapeFromList(List<double> data, {int n, int m}) /// create n⨯m matrix from [data] filled by rows
///
/// ```
class Matrix {
  /// The elements of the matrix
  late List<List<double>> _base;
  /// The row count
  late final int n;
  /// The column count
  late final int m;
  /// The [List<List<double>>] representation of matrix
  List<List<double>> get matrix => _base;
  /// The transposed matrix of [this]
  Matrix get T => MatrixOperation.transposition(this);
  /// Returns [true] if Matrix is column-vector or row-vector
  bool get isVector => m == 1 || n == 1;
  
  Matrix({required this.n, required this.m, int? seed}) {
    final variance = 2/(n+m);
    final rnormList = Normal.generate(n*m, mean: 0, variance: variance, seed: seed);
    _base = List<List<double>>.generate(n, (_) => List<double>.generate(m, (j) => rnormList[j+m*_], growable: false), growable: false);
  }
  Matrix.column(List<double> column) {
    n = column.length;
    m = 1;
    _base = List<List<double>>.generate(n, (index) => [column[index]]);
  }
  Matrix.row(List<double> row) {
    n = 1;
    m = row.length;
    _base = [row.sublist(0)];
  }
  Matrix.fromLists(List<List<double>> matrix) {
    n = matrix.length;
    m = matrix[0].length;
    _base = List<List<double>>
    .generate(n, (i) => List<double>.generate(m, (j) => matrix[i][j]));
  }
  Matrix.zero({required this.n, required this.m}) {
    _base = List<List<double>>.generate(n, (_) => List<double>.generate(m, (_) => 0, growable: false), growable: false);
  }
  Matrix.identity({required int order}) {
    n = order;
    m = order;
    _base = Matrix.zero(n: n, m: m).matrix;
    for (int i = 0; i < n; i += 1) {
      _base[i][i] = 1;
    }
  }
  Matrix.diag({required List<double> diag}) {
    n = diag.length;
    m = n;
    _base = List<List<double>>
      .generate(n, (i) => List<double>.generate(m, (j) => i == j ? diag[i] : 0));
  }
  Matrix.reshapeFromList(List<double> data, {required this.n, required this.m}) {
    if (data.length == n*m) {
      _base = List<List<double>>
        .generate(
          n,
          (index) => data.sublist(m * index, m * (index+1))
        );
    }
    else {
      throw Exception('Dimension error: ${data.length} != $n*$m');
    }
  }

  List<double> operator[](int index) {
    return _base[index];
  }

  /// Matrices multiplication operator
  Matrix operator*(Matrix other) {
    return MatrixOperation.multiplication(this, other);
  }

  /// Matrices elementwise addition operator
  Matrix operator+(Matrix other) {
    return MatrixOperation.addition(this, other);
  }

  /// Matrices elementwise subtraction operator
  Matrix operator-(Matrix other) {
    return MatrixOperation.subtraction(this, other);
  }

  /// Matrices Hadamard product operator
  Matrix operator%(Matrix other) {
    return MatrixOperation.hadamardProduct(this, other);
  }

  /// Negative matrix operator
  Matrix operator-() {
    return MatrixOperation.scalarMultiplication(this, -1);
  }

  /// Return a copy of [this] where [function] was applied to every element
  Matrix apply(double Function(double) function) {
    return MatrixOperation.apply(this, function);
  }

  /// Add scalar to every element of [this]
  void addScalar(double scalar) {
    _base = MatrixOperation.scalarAddition(this, scalar).matrix;
  }

  /// Return copy of [this] with added [scalar] to every element
  Matrix addedScalar(double scalar) {
    return MatrixOperation.scalarAddition(this, scalar);
  }

  /// Scale every element of [this] by [scalar]
  void scale(double scalar) {
    _base = MatrixOperation.scalarMultiplication(this, scalar).matrix;
  }

  /// Return copy of [this] with scaled elements by [scalar]
  Matrix scaled(double scalar) {
    return MatrixOperation.scalarMultiplication(this, scalar);
  }
  
  /// Return [Matrix.column] of [index] column of [this]
  Matrix getColumn(int index) {
    return Matrix.column(List<double>.generate(n, (j) => _base[j][index]));
  }

  /// Return [Matrix.row] of [index] row of [this]
  Matrix getRow(int index) {
    return Matrix.row(_base[index]);
  }

  /// Return [List] with all elements of [this] taken by rows
  List<double> flattenList() {
    List<double> flatten = [];
    for (var row in _base) {
      flatten.addAll(row);
    }
    return flatten;
  }

  /// Return [Matrix.column] with all elements of [this] taken by rows
  Matrix flattenColumn() {
    return Matrix.column(flattenList());
  }

  /// Return [Matrix.row] with all elements of [this] taken by rows
  Matrix flattenRow() {
    return Matrix.row(flattenList());
  }

  /// Return sum of all elements of [this]
  double reduceSum() {
    double sum = 0;
    for (int i = 0; i < n; i += 1) {
      for (int j = 0; j < m; j += 1) {
        sum += _base[i][j];
      }
    }
    return sum;
  }

  /// Return mean of all elements of [this]
  double reduceMean() {
    return reduceSum() / m / n;
  }

  /// Return [Matrix.column] of sums of the elements of [this]
  /// reduced by `rows` (if [axis] is 0) or `columns` (if [axis] is 1)
  Matrix reduceSumByAxis(int axis) {
    if (axis == 1) {
      return Matrix.column(List<double>.generate(m, (index) => getColumn(index).reduceSum()));
    }
    else if (axis == 0) {
      return Matrix.column(List<double>.generate(n, (index) => getRow(index).reduceSum()));
    }
    else {
      throw Exception('Dimensions error: axis shoud be 0 or 1');
    }
  }

  @override
  String toString() {
    String str = 'matrix $n⨯$m\n[';
    if (n < 11 && m < 11) {
      for (var row in _base) {
        str += '$row\n';
      }
      str = str.substring(0,str.length-1) + ']';
    }
    else {
      str += _base[0][0].toString() + '....' + _base[n-1][m-1].toString();
    }
    return str;
  }
}