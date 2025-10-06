import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'feature_extractor.dart';
import 'tflite_classifier.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

typedef PredictCb = void Function(String label, double conf);

class LiveEngine {
  final TensorFlowLiteClassifier clf;
  final extractor = FeatureExtractor();
  final Duration win = const Duration(seconds: 2);
  final Duration hop = const Duration(milliseconds: 1000);
  final int smoothK = 3;

  final List<ImuSample> _buf = [];
  final List<String> _lastLabels = [];
  DateTime? _lastInferAt;
  final PredictCb onPrediction;
  
  // สำหรับเก็บข้อมูล CSV (time series)
  final List<String> _csvData = [];

  LiveEngine(this.clf, {required this.onPrediction});

  Future<void> addSample(
    DateTime t,
    double ax,
    double ay,
    double az,
    // Removed gyroscope parameters - accelerometer only according to mobile_config
  ) async {
    // ใช้เฉพาะ accelerometer data เท่านั้น (ตาม mobile_config: accelerometer_only)
    _buf.add(ImuSample(t, ax, ay, az));

    final cutoff = t.subtract(win + const Duration(seconds: 1));
    while (_buf.isNotEmpty && _buf.first.t.isBefore(cutoff)) {
      _buf.removeAt(0);
    }

    if (_lastInferAt == null || t.difference(_lastInferAt!) >= hop) {
      _lastInferAt = t;
      await _inferIfReady();
    }
  }

  Future<void> _inferIfReady() async {
    if (_buf.isEmpty) return;
    
    final latest = _buf.last.t;
    final start = latest.subtract(win);
    final w = _buf
        .where((s) => !s.t.isBefore(start) && !s.t.isAfter(latest))
        .toList();
        
    if (w.length < 100) return; // ต้องมีอย่างน้อย 100 samples สำหรับ CNN [100, 4]

    print('=== CNN INFERENCE DEBUG (3 Classes - Accelerometer Only) ===');
    print('Window size: ${w.length} samples');
    print('Time range: ${start.millisecondsSinceEpoch} - ${latest.millisecondsSinceEpoch}');
    print('Duration: ${latest.difference(start).inMilliseconds}ms');

    // แปลง sensor data เป็น Map format สำหรับ feature extractor
    final sensorDataMaps = w.map((sample) => {
      'ax': sample.ax,
      'ay': sample.ay,
      'az': sample.az,
    }).toList();

    // เตรียม time series data สำหรับ CNN [100, 4] - Accelerometer + magnitude only
    final timeSeriesData = FeatureExtractor.prepareTimeSeriesData(sensorDataMaps);
    print('Time series shape: [${timeSeriesData.length}, ${timeSeriesData.first.length}]');
    print('Features: [ax=${timeSeriesData.first[0].toStringAsFixed(3)}, ay=${timeSeriesData.first[1].toStringAsFixed(3)}, az=${timeSeriesData.first[2].toStringAsFixed(3)}, mag=${timeSeriesData.first[3].toStringAsFixed(3)}]');
    
    final res = await clf.predictTimeSeries(timeSeriesData);
    print('Raw prediction result: $res');

    // Use correct keys from the classifier response
    final conf = (res["confidence"] as double);
    String label = (res["class"] as String);
    
    // Handle both Map<String, double> (mock) and List<double> (real model) formats
    final dynamic probsData = res["probabilities"];
    List<double> probs;
    
    if (probsData is Map<String, double>) {
      // Mock prediction format: cS removed)
      final newClasses = ['IDLE', 'RUN', 'WALK'];
      probs = newClasses.map((cls) => probsData[cls] ?? 0.0).toList();
    } else {
      // Real model format: already a List<double>
      probs = probsData as List<double>;
    }
    
    // แสดง probabilities ทุก class (ใช้ 3 classes ใหม่ - STAIRS removed)
    print('=== ALL CLASS PROBABILITIES (3 Classes) ===');
    final newClasses = ['IDLE', 'RUN', 'WALK'];
    for (int i = 0; i < newClasses.length && i < probs.length; i++) {
      print('${newClasses[i]}: ${(probs[i] * 100).toStringAsFixed(2)}%');
    }
    print('=====================================');
    
    if (conf < 0.30) label = "UNKNOWN";

    // เก็บข้อมูลลง CSV
    _saveToCSV(timeSeriesData, label);

    _lastLabels.add(label);
    if (_lastLabels.length > smoothK) _lastLabels.removeAt(0);
    final maj = _majority(_lastLabels);
    onPrediction(maj, conf);
  }

  void _saveToCSV(List<List<double>> timeSeriesData, String label) {
    final timestamp = DateTime.now().toIso8601String();
    // Simple CSV format for debugging
    final csvLine = '$timestamp,$label,${timeSeriesData.length}';
    _csvData.add(csvLine);
  }

  String getCSVData() {
    if (_csvData.isEmpty) return '';
    // เพิ่ม header เป็นบรรทัดแรก
    final header = 'timestamp,predicted_label,sample_count';
    // แสดงเฉพาะ 20 samples ล่าสุด
    final last20 = _csvData.length > 20 
        ? _csvData.sublist(_csvData.length - 20) 
        : _csvData;
    return '$header\n${last20.join('\n')}';
  }
  
  void clearCSVData() {
    _csvData.clear();
  }

  String _majority(List<String> xs) {
    final m = <String, int>{};
    for (final x in xs) {
      m[x] = (m[x] ?? 0) + 1;
    }
    String best = xs.first;
    int bc = 0;
    m.forEach((k, v) {
      if (v > bc) {
        bc = v;
        best = k;
      }
    });
    return best;
  }
}

// ===== App / UI =====
class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HAR AI',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
      home: const HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  TensorFlowLiteClassifier? _clf;
  LiveEngine? _engine;

  bool _modelReady = false;
  bool _running = false;
  String _label = "—";
  double _conf = 0.0;
  final double fs = 50.0; // Sampling rate 50Hz
  StreamSubscription<AccelerometerEvent>? _accSub;
  // Removed gyroscope subscription - accelerometer only
  Timer? _tick;
  double? _ax, _ay, _az; // Removed gyroscope variables

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    setState(() {
      _label = "🔄 Loading AI model...";
    });
    
    try {
      print('📱 Starting optimized model initialization...');
      // Use optimized singleton pattern for faster loading
      _clf = await TensorFlowLiteClassifier.getInstance(
        modelAsset: 'assets/cnn_har.tflite',
        configAsset: 'assets/mobile_config_corrected.json',
      );
      
      print('🚀 Creating inference engine...');
      _engine = LiveEngine(
        _clf!,
        onPrediction: (lab, conf) {
          if (!_running) return;
          setState(() {
            _label = lab;
            _conf = conf;
          });
        },
      );
      
      setState(() {
        _modelReady = true;
        _label = "✅ Model ready! Tap START to begin";
      });
      print('✅ Model initialization completed successfully');
      
    } catch (e) {
      print('❌ Model initialization failed: $e');
      setState(() {
        _label = "❌ Model loading failed";
        _modelReady = false;
      });
    }
  }

  void _start() {
    if (!_modelReady || _running) return;
    
    _accSub = accelerometerEventStream().listen((e) {
      _ax = e.x.toDouble();
      _ay = e.y.toDouble();
      _az = e.z.toDouble();
    });
    // Removed gyroscope stream - accelerometer only according to mobile_config
    _tick = Timer.periodic(Duration(milliseconds: (1000 / fs).toInt()), (_) async {
      final ax = _ax ?? 0.0, ay = _ay ?? 0.0, az = _az ?? 0.0;
      // Removed gyroscope data - accelerometer only
      await _engine?.addSample(DateTime.now(), ax, ay, az);
    });
    setState(() {
      _running = true;
      _label = "…";
      _conf = -1.0;
    });
  }

  void _stop() {
    _accSub?.cancel();
    _accSub = null;
    // Removed gyroscope cleanup - accelerometer only
    _tick?.cancel();
    _tick = null;
    setState(() {
      _running = false;
    });
  }

  Future<void> _exportCSV() async {
    if (_engine == null) return;
    
    final csvData = _engine!.getCSVData();
    if (csvData.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('ไม่มีข้อมูลให้ export')),
      );
      return;
    }

    // แสดง dialog พร้อม CSV data ที่สามารถ copy ได้
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Export CSV Data'),
        content: SingleChildScrollView(
          child: SelectableText(
            csvData,
            style: const TextStyle(fontSize: 10, fontFamily: 'monospace'),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Clipboard.setData(ClipboardData(text: csvData));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('คัดลอกไปยังคลิปบอร์ดแล้ว')),
              );
            },
            child: const Text('คัดลอก'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('ปิด'),
          ),
        ],
      ),
    );
  }

  Future<void> _clearData() async {
    _engine?.clearCSVData();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('ล้างข้อมูลเรียบร้อย')),
    );
  }

  @override
  void dispose() {
    _stop();
    _clf?.dispose(); // Clean up ONNX resources
    super.dispose();
  }

  // ฟังก์ชันแปลงป้ายกำกับเป็นภาษาไทย (อัพเดตสำหรับ 3 classes - STAIRS removed)
  String _getDisplayLabel(String label) {
    switch (label) {
      case "WALK":
        return "เดิน";
      case "RUN":
        return "วิ่ง";
      case "IDLE":
        return "อยู่นิ่ง";
      case "UNKNOWN":
        return "ไม่ทราบ";
      default:
        return label;
    }
  }

  // ฟังก์ชันเลือกสีตามป้ายกำกับ (อัพเดตสำหรับ 3 classes - STAIRS removed)
  Color _getLabelColor(String label) {
    switch (label) {
      case "WALK":
        return Colors.orange;
      case "RUN":
        return Colors.red;
      case "IDLE":
        return Colors.grey;
      case "UNKNOWN":
        return Colors.black54;
      default:
        return Colors.black;
    }
  }

  // สร้าง chip แสดง class
  Widget _buildClassChip(String originalLabel, String displayLabel, Color color) {
    return Chip(
      label: Text(
        displayLabel,
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w500,
        ),
      ),
      backgroundColor: color.withOpacity(0.8),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
    );
  }

  @override
  Widget build(BuildContext context) {
    final canPress = _modelReady && !_running;
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('HAR AI'),
        actions: [
          if (_modelReady)
            PopupMenuButton<String>(
              onSelected: (value) {
                if (value == 'export') {
                  _exportCSV();
                } else if (value == 'clear') {
                  _clearData();
                }
              },
              itemBuilder: (context) => [
                const PopupMenuItem(
                  value: 'export',
                  child: Row(
                    children: [
                      Icon(Icons.file_download),
                      SizedBox(width: 8),
                      Text('Export CSV'),
                    ],
                  ),
                ),
                const PopupMenuItem(
                  value: 'clear',
                  child: Row(
                    children: [
                      Icon(Icons.delete),
                      SizedBox(width: 8),
                      Text('ล้างข้อมูล'),
                    ],
                  ),
                ),
              ],
            ),
        ],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                minimumSize: const Size(180, 180),
                shape: const CircleBorder(),
              ),
              onPressed: canPress ? _start : null,
              child: const Text("START", style: TextStyle(fontSize: 22)),
            ),
            const SizedBox(height: 28),
            if (_running)
              TextButton(onPressed: _stop, child: const Text("Stop")),
            const SizedBox(height: 24),
            Text(
              _running
                  ? _getDisplayLabel(_label)
                  : (_modelReady ? "กด START เพื่อเริ่ม" : "กำลังโหลดโมเดล..."),
              key: ValueKey('${_running}_${_label}_$_conf'), // Force rebuild
              style: TextStyle(
                fontSize: 28, 
                fontWeight: FontWeight.bold,
                color: _running ? _getLabelColor(_label) : Colors.black,
              ),
              textAlign: TextAlign.center,
            ),
            if (_running) ...[
              const SizedBox(height: 8),
              Text(
                _conf >= 0 
                    ? "confidence: ${_conf.toStringAsFixed(2)}"
                    : "รอการทำนาย...",
                style: const TextStyle(fontSize: 16, color: Colors.black54),
              ),
              const SizedBox(height: 16),
              Text(
                "ป้ายกำกับเดิม: $_label",
                style: const TextStyle(fontSize: 14, color: Colors.black38),
              ),
            ] else if (_modelReady) ...[
              const SizedBox(height: 20),
              const Text(
                "Class ที่รองรับ:",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 4,
                alignment: WrapAlignment.center,
                children: [
                  // Updated for 3 classes only - STAIRS removed
                  _buildClassChip("IDLE", "อยู่นิ่ง", Colors.grey),
                  _buildClassChip("RUN", "วิ่ง", Colors.red),
                  _buildClassChip("WALK", "เดิน", Colors.orange),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
