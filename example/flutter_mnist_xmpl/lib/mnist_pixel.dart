import 'package:flutter/material.dart';

// ignore: must_be_immutable
class AnimatedPixel extends StatelessWidget {
  late ValueNotifier<int> scale;
  AnimatedPixel({required this.scale, Key? key }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<int>(
      valueListenable: scale,
      builder: (context, scale, child) {
        return AnimatedContainer(
          duration: const Duration(milliseconds: 100),
          decoration: BoxDecoration(
            color: Color.fromARGB(255, scale, scale, scale),
            border: Border.all(color: Colors.white30, width: 0.5),
          ),
        );
      },
    );
  }
}