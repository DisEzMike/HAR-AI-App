// This is a basic Flutter widget test for HAR AI app.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:har_ai/main.dart';

void main() {
  testWidgets('HAR AI app loads without error', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that the app title is present in AppBar
    expect(find.text('HAR AI'), findsOneWidget);

    // Verify that the loading text is present initially
    expect(find.text('กำลังโหลดโมเดล...'), findsOneWidget);

    // Verify that start button is present
    expect(find.byType(ElevatedButton), findsOneWidget);
  });

  testWidgets('Start/Stop button toggles', (WidgetTester tester) async {
    // Build our app
    await tester.pumpWidget(const MyApp());

    // Wait for model loading (mock)
    await tester.pump(const Duration(seconds: 1));

    // Find the start button
    final startButton = find.byType(ElevatedButton);
    expect(startButton, findsOneWidget);

    // Initially should show "START"
    expect(find.text('START'), findsOneWidget);

    // Tap to start
    await tester.tap(startButton);
    await tester.pump();

    // Should now show "Stop" button
    expect(find.text('Stop'), findsOneWidget);
  });
}
