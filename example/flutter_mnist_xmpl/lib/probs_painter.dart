import 'package:flutter/material.dart';

class ProbsPainter extends CustomPainter {
  late List<double> probs;
  ProbsPainter(this.probs);

  @override
  void paint(Canvas canvas, Size size) {
    final dx = size.width / 10;
    Paint paint = Paint()..color = Colors.white70;
    const TextStyle style = TextStyle(color: Colors.white, fontSize: 12);
    for (int i = 0; i < 10; i += 1) {
      Rect rect = Rect.fromPoints(
        Offset(dx*i, size.height),
        Offset(dx*(i+1), (size.height)*(1-probs[i]))
      );
      canvas.drawRect(rect, paint);

      TextPainter(
        text: TextSpan(text: probs[i].toStringAsFixed(2), style: style),
        textAlign: TextAlign.right,
        textDirection: TextDirection.ltr
      )..layout()..paint(canvas, Offset(dx*i+8, size.height*(1-probs[i])-20));
    }
  }

  @override
  bool shouldRepaint(ProbsPainter oldDelegate) => true;

}