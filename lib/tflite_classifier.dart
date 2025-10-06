import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

class TensorFlowLiteClassifier {
  static TensorFlowLiteClassifier? _instance;
  static Completer<TensorFlowLiteClassifier>? _initCompleter;
  
  Interpreter? _interpreter;
  late final List<String> classes;
  late final List<double> scalerMean;
  late final List<double> scalerScale;
  late final List<String> featureOrder;
  bool _isInitialized = false;
  bool _useMockPrediction = false;

  // Private constructor for singleton
  TensorFlowLiteClassifier._();

  /// Get singleton instance with automatic initialization
  static Future<TensorFlowLiteClassifier> getInstance({
    String modelAsset = 'assets/cnn_har.tflite',
    String configAsset = 'assets/mobile_config_corrected.json',
  }) async {
    if (_instance != null && _instance!._isInitialized) {
      return _instance!;
    }

    // If already initializing, wait for it to complete
    if (_initCompleter != null && !_initCompleter!.isCompleted) {
      return await _initCompleter!.future;
    }

    // Start new initialization
    _initCompleter = Completer<TensorFlowLiteClassifier>();
    _instance = TensorFlowLiteClassifier._();
    
    try {
      await _instance!.init(modelAsset: modelAsset, configAsset: configAsset);
      _initCompleter!.complete(_instance!);
      return _instance!;
    } catch (e) {
      _initCompleter!.completeError(e);
      rethrow;
    }
  }

  Future<void> init({
    String modelAsset = 'assets/cnn_har.tflite',
    String configAsset = 'assets/mobile_config_corrected.json', // Use corrected config
  }) async {
    if (_isInitialized) return;
    
    print('🚀 Starting HAR Model Initialization...');
    final stopwatch = Stopwatch()..start();
    
    try {
      // โหลด mobile config พร้อมกับแสดงสถานะ
      print('📋 Loading configuration file: $configAsset');
      final configStr = await rootBundle.loadString(configAsset);
      final config = json.decode(configStr) as Map<String, dynamic>;
      print('✅ Configuration loaded (${stopwatch.elapsedMilliseconds}ms)');
      
      // Parse input shape [100, 4] - accelerometer only
      final modelInfo = config['model_info'] as Map<String, dynamic>;
      final preprocessing = config['preprocessing'] as Map<String, dynamic>;
      
      final inputShape = List<int>.from(modelInfo['input_shape']);
      final windowSize = preprocessing['window_size'] as int;
      
      // สร้าง classes list (3 classes: IDLE, RUN, WALK)
      classes = List<String>.from(modelInfo['classes']);
      
      // Feature order: accelerometer_x, accelerometer_y, accelerometer_z, accelerometer_magnitude
      featureOrder = List<String>.from(preprocessing['features']);
      
      // ดึง scaler parameters จาก config file
      final normalization = preprocessing['normalization'] as Map<String, dynamic>;
      final scalerParams = normalization['scaler_parameters'] as Map<String, dynamic>;
      
      scalerMean = List<double>.from(scalerParams['mean']);
      scalerScale = List<double>.from(scalerParams['scale']);
      
      print('=== Model Configuration ===');
      print('Classes: $classes');
      print('Input shape: $inputShape');
      print('Window size: $windowSize');
      print('Features: $featureOrder');
      
      // โหลดโมเดล TensorFlow Lite พร้อมแสดงสถานะ
      print('🤖 Loading TensorFlow Lite model: $modelAsset');
      final modelLoadStart = stopwatch.elapsedMilliseconds;
      
      // ใช้ Completer เพื่อให้สามารถ cancel ได้
      final completer = Completer<Interpreter>();
      
      // Start loading model asynchronously
      Interpreter.fromAsset(modelAsset).then((interpreter) {
        if (!completer.isCompleted) {
          completer.complete(interpreter);
        }
      }).catchError((error) {
        if (!completer.isCompleted) {
          completer.completeError(error);
        }
      });
      
      // Wait with shorter timeout for better user experience
      _interpreter = await completer.future.timeout(
        Duration(seconds: 5),
        onTimeout: () {
          throw TimeoutException('Model loading timeout after 5 seconds', Duration(seconds: 5));
        },
      );
      
      final modelLoadTime = stopwatch.elapsedMilliseconds - modelLoadStart;
      print('✅ Model loaded successfully! (${modelLoadTime}ms)');
      
      // ตรวจสอบ input/output shape
      final modelInputShape = _interpreter!.getInputTensor(0).shape;
      final modelOutputShape = _interpreter!.getOutputTensor(0).shape;
      
      print('Model input shape: $modelInputShape');  
      print('Model output shape: $modelOutputShape');
      
      // ตรวจสอบว่า input shape ตรงกับที่คาดหวัง [1, 100, 4]
      if (modelInputShape.length >= 3 && 
          modelInputShape[1] == 100 && 
          modelInputShape[2] == 4) {
        print('✅ Model validation passed');
        _isInitialized = true;
        _useMockPrediction = false;
      } else {
        print('⚠️ Model input shape mismatch. Expected [1, 100, 4], got $modelInputShape');
        print('Using fallback mock prediction mode');
        _useMockPrediction = true;
        _isInitialized = true;
      }
      
      stopwatch.stop();
      print('🎉 HAR Model initialization completed! Total time: ${stopwatch.elapsedMilliseconds}ms');
      
    } catch (e) {
      stopwatch.stop();
      print('❌ Model initialization failed after ${stopwatch.elapsedMilliseconds}ms: $e');
      print('Using fallback mock prediction mode');
      _useMockPrediction = true;
      _isInitialized = true;
    }
  }

  /// Predict จาก time series data [100, 4] - Accelerometer only features
  Future<Map<String, dynamic>> predictTimeSeries(List<List<double>> timeSeriesData) async {
    if (!_isInitialized) {
      throw StateError('Classifier not initialized. Call init() first.');
    }

    if (timeSeriesData.length != 100 || timeSeriesData.first.length != 4) {
      throw ArgumentError('Expected input shape [100, 4], got [${timeSeriesData.length}, ${timeSeriesData.first.length}]');
    }

    print('=== TensorFlow Lite Prediction (Accelerometer Only - 3 Classes) ===');
    print('Input shape: [${timeSeriesData.length}, ${timeSeriesData.first.length}]');
    print('Expected features: $featureOrder');
    print('Sample input: ${timeSeriesData.first.map((x) => x.toStringAsFixed(3)).join(", ")}');

    if (_useMockPrediction) {
      print('Using mock prediction mode');
      return _mockPrediction();
    }

    try {
      // Apply StandardScaler normalization
      final normalizedData = _normalizeFeatures(timeSeriesData);
      
      // Flatten data สำหรับ TensorFlow Lite [100, 4] -> [1, 100, 4]
      final inputData = [normalizedData];
      final outputData = [List<double>.filled(3, 0.0)]; // 3 classes (IDLE, RUN, WALK)

      _interpreter!.run(inputData, outputData);
      
      final probabilities = outputData[0];
      
      // หา class ที่มี probability สูงสุด
      int maxIndex = 0;
      double maxProb = (probabilities[0] as num).toDouble();
      
      for (int i = 1; i < probabilities.length; i++) {
        final prob = (probabilities[i] as num).toDouble();
        if (prob > maxProb) {
          maxProb = prob;
          maxIndex = i;
        }
      }
      
      final predictedClass = classes[maxIndex];
      final confidence = maxProb;
      
      print('Predicted: $predictedClass (${(confidence * 100).toStringAsFixed(1)}%)');
      print('All probabilities: ${probabilities.map((p) => (p * 100).toStringAsFixed(1) + '%').join(', ')}');
      
      // Check if input data suggests IDLE (very low accelerometer magnitude)
      final avgMagnitude = timeSeriesData.map((sample) => sample[3]).reduce((a, b) => a + b) / timeSeriesData.length;
      print('Average magnitude: ${avgMagnitude.toStringAsFixed(3)}');
      
      // Calculate variance to detect motion
      final magnitudes = timeSeriesData.map((sample) => sample[3]).toList();
      final meanMag = magnitudes.reduce((a, b) => a + b) / magnitudes.length;
      final variance = magnitudes.map((m) => pow(m - meanMag, 2)).reduce((a, b) => a + b) / magnitudes.length;
      print('Magnitude variance: ${variance.toStringAsFixed(6)}');
      
      String finalPrediction = predictedClass;
      
      // If variance is very low (< 0.01), it's likely IDLE regardless of model prediction
      if (variance < 0.01 && predictedClass == 'WALK') {
        finalPrediction = 'IDLE';
        print('🔄 Overriding WALK to IDLE due to low variance (${variance.toStringAsFixed(6)})');
      }
      
      return {
        'class': finalPrediction,
        'confidence': confidence,
        'probabilities': Map.fromIterables(classes, probabilities),
        'model_type': 'cnn_tflite_3classes_accelerometer'
      };
      
    } catch (e) {
      print('Prediction error: $e');
      return _mockPrediction();
    }
  }


  /// Apply StandardScaler normalization to features
  List<List<double>> _normalizeFeatures(List<List<double>> data) {
    final normalized = <List<double>>[];
    
    print('🔧 NORMALIZATION DEBUG:');
    print('Scaler Mean: $scalerMean');
    print('Scaler Scale: $scalerScale');
    print('Raw sample: ${data.first.map((x) => x.toStringAsFixed(3)).join(", ")}');
    
    for (final timeStep in data) {
      final normalizedStep = <double>[];
      for (int i = 0; i < timeStep.length; i++) {
        final normalizedValue = (timeStep[i] - scalerMean[i]) / scalerScale[i];
        normalizedStep.add(normalizedValue);
      }
      normalized.add(normalizedStep);
    }
    
    print('Normalized sample: ${normalized.first.map((x) => x.toStringAsFixed(3)).join(", ")}');
    
    return normalized;
  }

  /// Fallback mock prediction เมื่อ TensorFlow Lite ใช้ไม่ได้ - Updated for 3 classes
  Map<String, dynamic> _mockPrediction() {
    // Updated to use 3 classes only (STAIRS removed)
    final mockClasses = ['IDLE', 'RUN', 'WALK'];
    final random = DateTime.now().millisecondsSinceEpoch % mockClasses.length;
    final selectedClass = mockClasses[random];
    
    final mockProbs = List.generate(3, (i) => 
      i == random ? 0.85 + (DateTime.now().microsecond % 100) / 1000.0 : 
      (DateTime.now().microsecond % 50) / 1000.0
    );
    
    print('🎭 Mock prediction (3 classes): $selectedClass');
    
    return {
      'class': selectedClass,
      'confidence': mockProbs[random],
      'probabilities': Map.fromIterables(mockClasses, mockProbs),
      'model_type': 'mock_cnn_3classes'
    };
  }

  /// Clean up resources
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }

  /// Status checking
  bool get isInitialized => _isInitialized;
  bool get isMockMode => _useMockPrediction;
  List<String> get supportedClasses => List.from(classes);
  
  /// Utility method สำหรับ debugging
  void printModelInfo() {
    if (!_isInitialized) {
      print('Model not initialized');
      return;
    }
    
    print('=== TensorFlow Lite Model Info ===');
    print('Mode: ${_useMockPrediction ? "Mock" : "TensorFlow Lite"}');
    print('Classes: $classes');
    print('Feature order: $featureOrder');
    print('Normalization enabled: ${scalerMean.isNotEmpty}');
    
    if (_interpreter != null) {
      try {
        final inputShape = _interpreter!.getInputTensor(0).shape;
        final outputShape = _interpreter!.getOutputTensor(0).shape;
        print('Input shape: $inputShape');
        print('Output shape: $outputShape');
      } catch (e) {
        print('Could not get tensor info: $e');
      }
    }
  }

  /// Export method สำหรับ CSV debugging
  String exportTimeSeriesToCSV(List<List<double>> timeSeriesData, {String? timestamp}) {
    final ts = timestamp ?? DateTime.now().toIso8601String();
    final csvRows = <String>[];
    
    // Header
    final header = ['timestamp', 'sample_index', ...featureOrder].join(',');
    if (csvRows.isEmpty) csvRows.add(header);
    
    // Data rows
    for (int i = 0; i < timeSeriesData.length; i++) {
      final row = [ts, i.toString(), ...timeSeriesData[i].map((v) => v.toStringAsFixed(6))].join(',');
      csvRows.add(row);
    }
    
    return csvRows.join('\n');
  }

  /// Get CSV header
  String getCSVHeader() {
    return ['timestamp', 'sample_index', ...featureOrder].join(',');
  }
}

// Extension เพื่อสร้าง time series data จาก sensor readings
extension SensorDataProcessing on List<Map<String, double>> {
  /// แปลง sensor readings เป็น time series [100, 8] format with enhanced features
  List<List<double>> toTimeSeriesDataEnhanced() {
    final result = <List<double>>[];
    
    // ถ้ามี data น้อยกว่า 100 samples ให้ pad ด้วย zeros
    final targetLength = 100;
    
    // Calculate kurtosis for the entire window
    final axValues = map((s) => s['ax'] ?? 0.0).toList();
    final ayValues = map((s) => s['ay'] ?? 0.0).toList();
    final azValues = map((s) => s['az'] ?? 0.0).toList();
    
    final axKurt = _calculateKurtosis(axValues);
    final ayKurt = _calculateKurtosis(ayValues);
    final azKurt = _calculateKurtosis(azValues);
    
    for (int i = 0; i < targetLength; i++) {
      if (i < length) {
        final sample = this[i];
        final ax = sample['ax'] ?? 0.0;
        final ay = sample['ay'] ?? 0.0;
        final az = sample['az'] ?? 0.0;
        
        // Calculate enhanced features
        final accMag = sqrt(ax * ax + ay * ay + az * az);
        final accDen = accMag / sqrt(3);
        
        result.add([
          ax,           // accelerometer_x
          ay,           // accelerometer_y  
          az,           // accelerometer_z
          accMag,       // accelerometer_magnitude
          accDen,       // accelerometer_density
          axKurt,       // accelerometer_x_kurtosis
          ayKurt,       // accelerometer_y_kurtosis
          azKurt,       // accelerometer_z_kurtosis
        ]);
      } else {
        // Pad with zeros if not enough data
        result.add([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
      }
    }
    
    return result;
  }

  /// Calculate kurtosis (Fisher's definition: kurtosis - 3)
  double _calculateKurtosis(List<double> data) {
    if (data.length < 4) return 0.0;
    
    final mean = data.reduce((a, b) => a + b) / data.length;
    final n = data.length;
    
    // Calculate moments
    double m2 = 0.0, m4 = 0.0;
    for (final val in data) {
      final diff = val - mean;
      final diff2 = diff * diff;
      m2 += diff2;
      m4 += diff2 * diff2;
    }
    
    m2 /= n;
    m4 /= n;
    
    if (m2 == 0.0) return 0.0;
    
    // Fisher's kurtosis (excess kurtosis)
    return (m4 / (m2 * m2)) - 3.0;
  }
}