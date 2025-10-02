import 'dart:convert';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

class OnnxStairsClassifier {
  late final OrtSession _session;
  late final OrtSessionOptions _sessionOptions;
  late final List<String> featureOrder; // ลำดับฟีเจอร์ที่โมเดลคาดหวัง
  late final List<String> classes;
  bool _isInitialized = false;

  Future<void> init({
    String onnxAsset = 'assets/rf_4cls.onnx',
    String metaAsset = 'assets/rf_4cls.meta.json',
  }) async {
    if (_isInitialized) return;
    
    // Initialize ONNX Runtime environment
    OrtEnv.instance.init();
    
    final metaStr = await rootBundle.loadString(metaAsset);
    final meta = json.decode(metaStr) as Map<String, dynamic>;
    featureOrder = (meta['features'] as List).cast<String>();
    classes = (meta['classes'] as List).cast<String>();

    final bytes = await rootBundle.load(onnxAsset);
    final modelData = bytes.buffer.asUint8List();
    
    _sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromBuffer(modelData, _sessionOptions);
    _isInitialized = true;
  }

  /// map ฟีเจอร์ตามลำดับ featureOrder
  List<double> toFeatureVector(Map<String, double> feats) {
    print('=== FEATURE MAPPING DEBUG ===');
    print('Input features count: ${feats.length}');
    print('Expected features count: ${featureOrder.length}');
    
    final missingFeatures = <String>[];
    final vector = featureOrder.map((k) {
      final value = feats[k];
      if (value == null) {
        missingFeatures.add(k);
        return 0.0;
      }
      return value;
    }).toList(growable: false);
    
    if (missingFeatures.isNotEmpty) {
      print('Missing features (${missingFeatures.length}): $missingFeatures');
    }
    
    print('Available features: ${feats.keys.toList()}');
    print('Sample feature values:');
    feats.entries.take(5).forEach((e) {
      print('  ${e.key}: ${e.value}');
    });
    
    print('Mapped vector stats:');
    print('  Length: ${vector.length}');
    print('  Min: ${vector.isEmpty ? 0 : vector.reduce((a, b) => a < b ? a : b)}');
    print('  Max: ${vector.isEmpty ? 0 : vector.reduce((a, b) => a > b ? a : b)}');
    print('  Non-zero: ${vector.where((x) => x != 0.0).length}');
    print('============================');
    
    return vector;
  }

  /// คืน {"label":..., "conf":..., "probs":[...]}
  Future<Map<String, dynamic>> predictVector(List<double> fv) async {
    if (!_isInitialized) {
      throw StateError('Classifier not initialized. Call init() first.');
    }
    
    // Convert to float32 format that the model expects
    final floatData = Float32List.fromList(fv);
    
    // Debug: ตรวจสอบ input features
    // print('=== INPUT FEATURES DEBUG ===');
    // print('Feature vector length: ${fv.length}');
    // print('Expected features: ${featureOrder.length}');
    // print('Features: $fv');
    // print('Min value: ${fv.isEmpty ? 0 : fv.reduce((a, b) => a < b ? a : b)}');
    // print('Max value: ${fv.isEmpty ? 0 : fv.reduce((a, b) => a > b ? a : b)}');
    // print('Sum: ${fv.fold(0.0, (a, b) => a + b)}');
    // print('Non-zero count: ${fv.where((x) => x != 0.0).length}');
    
    // ตรวจสอบว่ามี features ที่เป็น NaN หรือ Infinity หรือไม่
    final nanCount = fv.where((x) => x.isNaN).length;
    final infCount = fv.where((x) => x.isInfinite).length;
    if (nanCount > 0) print('WARNING: $nanCount NaN values found');
    if (infCount > 0) print('WARNING: $infCount Infinity values found');
    
    // ตรวจสอบช่วงค่าของ features
    if (fv.isNotEmpty) {
      final mean = fv.fold(0.0, (a, b) => a + b) / fv.length;
      final variance = fv.map((x) => (x - mean) * (x - mean)).fold(0.0, (a, b) => a + b) / fv.length;
      final stdDev = math.sqrt(variance);
      print('Statistics: mean=${mean.toStringAsFixed(4)}, std=${stdDev.toStringAsFixed(4)}');
      
      // ตรวจสอบว่าค่าส่วนใหญ่เป็น 0 หรือไม่
      final zeroCount = fv.where((x) => x == 0.0).length;
      final zeroPercent = (zeroCount / fv.length * 100);
      print('Zero values: $zeroCount/${fv.length} (${zeroPercent.toStringAsFixed(1)}%)');
      
      if (zeroPercent > 80) {
        print('WARNING: Too many zero features (${zeroPercent.toStringAsFixed(1)}%)');
      }
    }
    
    print('===========================');
    
    final input = OrtValueTensor.createTensorWithDataList(floatData, [1, fv.length]);
    final inputs = {'input': input};
    final runOptions = OrtRunOptions();
    
    try {
      final outputs = await _session.runAsync(runOptions, inputs);
      
      if (outputs == null || outputs.isEmpty) {
        throw StateError('No output from model');
      }
      
    print('=== MODEL OUTPUTS ===');
    print('Number of outputs: ${outputs.length}');
    for (int i = 0; i < outputs.length; i++) {
      final output = outputs[i];
      final outputData = output?.value;
      print('Output $i: type=${outputData.runtimeType}, shape=${outputData is List ? outputData.length : 'N/A'}');
    }
    print('==================');

    // Random Forest มักจะ return class prediction และ probabilities แยกกัน
    // ลองดึง output ที่ 2 (probabilities) แทนที่จะใช้ output แรก
    final firstOutput = outputs[0]?.value;
    
    List<double> probabilities = [];
    String? predictedClass;
    
    // ตรวจสอบ output แรก (class prediction)
    if (firstOutput is List && firstOutput.isNotEmpty) {
      if (firstOutput.first is String) {
        predictedClass = firstOutput.first as String;
        print('=== PREDICTED CLASS ===');
        print('Direct prediction: $predictedClass');
        print('=====================');
      }
    }
    
    // ตรวจสอบ output ที่สอง (probabilities)
    if (outputs.length > 1) {
      final secondOutput = outputs[1]?.value;
      print('=== PROBABILITY OUTPUT ===');
      print('Second output type: ${secondOutput.runtimeType}');
      print('Second output: $secondOutput');
      print('========================');
      
      if (secondOutput is List<List<double>>) {
        // Random Forest probabilities usually come as [[prob1, prob2, ...]]
        if (secondOutput.isNotEmpty && secondOutput.first.length == classes.length) {
          probabilities = secondOutput.first;
          print('Using probabilities from second output: $probabilities');
        }
      } else if (secondOutput is List<double> && secondOutput.length == classes.length) {
        probabilities = secondOutput;
        print('Using probabilities from second output: $probabilities');
      }
    }
    
    // ถ้าไม่มี probabilities ให้สร้าง one-hot จาก predicted class
    if (probabilities.isEmpty && predictedClass != null) {
      probabilities = List.filled(classes.length, 0.0);
      final classIndex = classes.indexOf(predictedClass);
      if (classIndex >= 0) {
        probabilities[classIndex] = 1.0;
        print('Created one-hot probabilities: $probabilities');
      }
    }
    
    // ถ้ายังไม่มี probabilities ให้ใช้ uniform distribution
    if (probabilities.isEmpty) {
      probabilities = List.filled(classes.length, 1.0 / classes.length);
      print('Using uniform probabilities: $probabilities');
    }
    
    // ตรวจสอบและประมวลผล probabilities
    print('=== PROCESSING PROBABILITIES ===');
    print('Raw probabilities: $probabilities');
    final sum = probabilities.fold(0.0, (a, b) => a + b);
    print('Sum of values: ${sum.toStringAsFixed(4)}');
    
    List<double> finalProbs;
    if (sum == 0.0) {
      print('All probabilities are zero, using uniform distribution');
      finalProbs = List.filled(classes.length, 1.0 / classes.length);
    } else if ((sum - 1.0).abs() > 0.01) {
      print('Applying softmax transformation...');
      finalProbs = _softmax(probabilities);
      print('After softmax: $finalProbs');
      print('Softmax sum: ${finalProbs.fold(0.0, (a, b) => a + b).toStringAsFixed(4)}');
    } else {
      finalProbs = probabilities;
      print('Probabilities already normalized');
    }
      
      int best = 0;
      double bp = finalProbs[0];
      for (int i = 1; i < finalProbs.length; i++) {
        if (finalProbs[i] > bp) {
          bp = finalProbs[i];
          best = i;
        }
      }
      
      final result = {"label": classes[best], "conf": bp, "probs": finalProbs};
      
      // แสดง output จากการ predict
      print('=== PREDICTION RESULT ===');
      print('Predicted Label: ${result["label"]}');
      print('Confidence: ${result["conf"]}');
      print('All Probabilities: ${result["probs"]}');
      print('========================');
      
      return result;
    } finally {
      // Clean up resources
      input.release();
      runOptions.release();
    }
  }

  /// Dispose of resources when done
  void dispose() {
    if (_isInitialized) {
      _session.release();
      _sessionOptions.release();
      OrtEnv.instance.release();
      _isInitialized = false;
    }
  }

  /// Apply softmax transformation to convert logits to probabilities
  List<double> _softmax(List<double> logits) {
    if (logits.isEmpty) return [];
    
    // ป้องกัน overflow โดยลบค่าสูงสุด
    final maxVal = logits.reduce((a, b) => a > b ? a : b);
    final expVals = logits.map((x) => math.exp(x - maxVal)).toList();
    final sumExp = expVals.fold(0.0, (a, b) => a + b);
    
    if (sumExp == 0.0) return List.filled(logits.length, 1.0 / logits.length);
    
    return expVals.map((x) => x / sumExp).toList();
  }
}
