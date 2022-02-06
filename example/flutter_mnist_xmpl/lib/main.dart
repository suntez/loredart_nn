import 'package:flutter/material.dart';
import 'package:flutter_mnist_xmpl/main_page.dart';

void main() async {
  runApp(const MnistClassApp());
}

class MnistClassApp extends StatelessWidget {
  const MnistClassApp({ Key? key }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: const MainPage(),
      theme: ThemeData.dark()
    );
  }
}
