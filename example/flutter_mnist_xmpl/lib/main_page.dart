import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_mnist_xmpl/mnist_pixel.dart';
import 'package:flutter_mnist_xmpl/probs_painter.dart';
import 'package:loredart_nn/loredart_nn.dart';

class MainPage extends StatefulWidget {
  const MainPage({Key? key}) : super(key: key);

  @override
  _MainPageState createState() => _MainPageState();
}

class _MainPageState extends State<MainPage>
    with SingleTickerProviderStateMixin {
  List<ValueNotifier<int>> pixels =
      List<ValueNotifier<int>>.generate(784, (index) => ValueNotifier<int>(0));
  List<double> probs = List<double>.filled(10, 0);
  int classIndex = 0;

  NeuralNetwork model = NeuralNetwork(
      784,
      [
        Dense(64, activation: Activation.softplus()), //or leakyReLU or elu
        Normalization(),
        Dense(10, activation: Activation.softmax())
      ],
      loss: Loss
          .crossEntropy(), // sparseCrossEntropy can be used for sparse data with the same result
      optimizer: SGD(learningRate: 0.01, momentum: 0.9),
      useAccuracyMetric: true);

  @override
  void initState() {
    rootBundle.load('assets/models/ce_mnist/model_weights.bin').then((value) {
      model.loadWeightsFromBytes(value.buffer);
    });
    super.initState();
  }

  int whichMax(List<double> data) {
    return data.indexOf(data.reduce((v, e) => max(v, e)));
  }

  void _updateProbs() {
    probs = model
        .predict([pixels.map((e) => e.value / 255).toList()])[0].flattenList();
    setState(() {
      classIndex = whichMax(probs);
    });
  }

  @override
  Widget build(BuildContext context) {
    final dt = MediaQuery.of(context).size.width / 28;
    final yBias =
        AppBar().preferredSize.height + MediaQuery.of(context).padding.top;
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'loredart_nn',
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.cleaning_services),
            onPressed: () {
              for (var element in pixels) {
                element.value = 0;
              }
            },
          )
        ],
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        mainAxisSize: MainAxisSize.max,
        children: [
          GestureDetector(
            onPanStart: (details) {
              /// global position match local
              final j = ((details.globalPosition.dx - 5) / dt).floor();
              final i = ((details.globalPosition.dy - 5 - yBias) / dt).floor();
              if (j >= 0 && j < 28 && i >= 0 && i < 28) {
                pixels[j + i * 28].value = pixels[j + i * 28].value < 200
                    ? (pixels[j + i * 28].value + 255)
                    : 255;
                if (j < 27) {
                  pixels[j + i * 28 + 1].value = pixels[j + i * 28].value < 200
                      ? (pixels[j + i * 28].value + 255)
                      : 255;
                }
                if (j > 0) {
                  pixels[j + i * 28 - 1].value = pixels[j + i * 28].value < 200
                      ? (pixels[j + i * 28].value + 255)
                      : 255;
                }
                if (i < 27) {
                  pixels[j + (i + 1) * 28].value =
                      pixels[j + i * 28].value < 200
                          ? (pixels[j + i * 28].value + 255)
                          : 255;
                }
                if (i > 0) {
                  pixels[j + (i - 1) * 28].value =
                      pixels[j + i * 28].value < 200
                          ? (pixels[j + i * 28].value + 255)
                          : 255;
                }
              }
            },
            onPanUpdate: (details) {
              final j = ((details.globalPosition.dx - 5) / dt).floor();
              final i = ((details.globalPosition.dy - 5 - yBias) / dt).floor();
              if (j >= 0 && j < 28 && i >= 0 && i < 28) {
                pixels[j + i * 28].value = pixels[j + i * 28].value < 200
                    ? (pixels[j + i * 28].value + 255)
                    : 255;
                if (j < 27) {
                  pixels[j + i * 28 + 1].value = pixels[j + i * 28].value < 200
                      ? (pixels[j + i * 28].value + 255)
                      : 255;
                }
                if (j > 0) {
                  pixels[j + i * 28 - 1].value = pixels[j + i * 28].value < 200
                      ? (pixels[j + i * 28].value + 255)
                      : 255;
                }
                if (i < 27) {
                  pixels[j + (i + 1) * 28].value =
                      pixels[j + i * 28].value < 200
                          ? (pixels[j + i * 28].value + 255)
                          : 255;
                }
                if (i > 0) {
                  pixels[j + (i - 1) * 28].value =
                      pixels[j + i * 28].value < 200
                          ? (pixels[j + i * 28].value + 255)
                          : 255;
                }
              }
            },
            onPanEnd: (details) {
              _updateProbs();
            },
            child: Container(
                height: MediaQuery.of(context).size.width,
                width: MediaQuery.of(context).size.width,
                margin: const EdgeInsets.all(5),
                decoration: BoxDecoration(
                    color: Colors.black,
                    borderRadius: BorderRadius.circular(3),
                    boxShadow: const [
                      BoxShadow(
                          offset: Offset(1, 1),
                          blurRadius: 3,
                          spreadRadius: 1,
                          color: Colors.white12)
                    ]),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    for (int i = 0; i < 28; i += 1)
                      Expanded(
                        flex: 1,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          mainAxisSize: MainAxisSize.max,
                          children: [
                            for (int j = 0; j < 28; j += 1)
                              Expanded(
                                  flex: 1,
                                  child:
                                      AnimatedPixel(scale: pixels[j + i * 28]))
                          ],
                        ),
                      )
                  ],
                )),
          ),
          Expanded(
              flex: 3,
              child: CustomPaint(
                foregroundPainter: ProbsPainter(probs),
                child: Container(),
              )),
          Expanded(
              flex: 1,
              child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  mainAxisSize: MainAxisSize.max,
                  children: List.generate(
                      10,
                      (index) => Expanded(
                            flex: 1,
                            child: AnimatedContainer(
                              margin: const EdgeInsets.symmetric(
                                  vertical: 1, horizontal: 2),
                              duration: const Duration(milliseconds: 300),
                              decoration: BoxDecoration(
                                  color: classIndex == index
                                      ? Colors.white70
                                      : Colors.black,
                                  borderRadius: BorderRadius.circular(3),
                                  boxShadow: [
                                    BoxShadow(
                                        color: classIndex == index
                                            ? Colors.black
                                            : Colors.white30,
                                        blurRadius: 3)
                                  ]),
                              alignment: const Alignment(0, 0),
                              child: Text(index.toString(),
                                  style: TextStyle(
                                      color: classIndex == index
                                          ? Colors.black
                                          : Colors.white)),
                            ),
                          )))),
        ],
      ),
    );
  }
}
