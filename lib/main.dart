import 'dart:async';
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'feature_extractor.dart';
import 'onnx_classifier.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

// ===== Engine: รับ sample → window → features → ONNX → callback =====
typedef PredictCb = void Function(String label, double conf);

class LiveEngine {
  final FeatureExtractor extractor = FeatureExtractor();
  final OnnxStairsClassifier clf;
  final Duration win = const Duration(seconds: 3);
  final Duration hop = const Duration(milliseconds: 1500);
  final int smoothK = 3;

  final List<ImuSample> _buf = [];
  final List<String> _lastLabels = [];
  DateTime? _lastInferAt;
  final PredictCb onPrediction;

  LiveEngine(this.clf, {required this.onPrediction});

  Future<void> addSample(
    DateTime t,
    double ax,
    double ay,
    double az,
    double gx,
    double gy,
    double gz,
  ) async {
    _buf.add(ImuSample(t, ax, ay, az, gx, gy, gz));

    final cutoff = t.subtract(win + const Duration(seconds: 1));
    while (_buf.isNotEmpty && _buf.first.t.isBefore(cutoff)) {
      _buf.removeAt(0);
    }

    if (_lastInferAt == null || t.difference(_lastInferAt!) >= hop) {
      // print('=== TRIGGERING INFERENCE ===');
      // print('Buffer size: ${_buf.length}');
      // print('Win: ${win.inSeconds}s, Hop: ${hop.inMilliseconds}ms');
      // print('Time since last inference: ${_lastInferAt == null ? "never" : t.difference(_lastInferAt!).inMilliseconds}ms');
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
        
    if (w.length < 16) return;

    final feats = extractor.computeFeatures(w);
    final fv = clf.toFeatureVector(feats);
    final res = await clf.predictVector(fv);
    print((fv));

    final conf = (res["conf"] as double);
    String label = (res["label"] as String);
    if (conf < 0.55) label = "UNKNOWN";

    _lastLabels.add(label);
    if (_lastLabels.length > smoothK) _lastLabels.removeAt(0);
    final maj = _majority(_lastLabels);
    onPrediction(maj, conf);
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
      title: 'IMU Stairs Classifier',
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
  final OnnxStairsClassifier _clf = OnnxStairsClassifier();
  LiveEngine? _engine;

  bool _modelReady = false;
  bool _running = false;
  String _label = "—";
  double _conf = 0.0;

  StreamSubscription<AccelerometerEvent>? _accSub;
  StreamSubscription<GyroscopeEvent>? _gyroSub;
  Timer? _tick;
  double? _ax, _ay, _az, _gx, _gy, _gz;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    try {
      await _clf.init(
        onnxAsset: 'assets/rf_4cls.onnx',
        metaAsset: 'assets/rf_4cls.meta.json',
      );
      _engine = LiveEngine(
        _clf,
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
      });
    } catch (e) {
      setState(() {
        _label = "Model load error: $e";
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
    _gyroSub = gyroscopeEventStream().listen((e) {
      _gx = e.x.toDouble();
      _gy = e.y.toDouble();
      _gz = e.z.toDouble();
    });
    _tick = Timer.periodic(const Duration(milliseconds: 20), (_) async {
      final ax = _ax ?? 0.0, ay = _ay ?? 0.0, az = _az ?? 0.0;
      final gx = _gx ?? 0.0, gy = _gy ?? 0.0, gz = _gz ?? 0.0;
      await _engine?.addSample(DateTime.now(), ax, ay, az, gx, gy, gz);
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
    _gyroSub?.cancel();
    _gyroSub = null;
    _tick?.cancel();
    _tick = null;
    setState(() {
      _running = false;
    });
  }

  @override
  void dispose() {
    _stop();
    _clf.dispose(); // Clean up ONNX resources
    super.dispose();
  }

  // ฟังก์ชันแปลงป้ายกำกับเป็นภาษาไทย
  String _getDisplayLabel(String label) {
    switch (label) {
      case "DOWNSTAIRS":
        return "ลงบันได";
      case "UPSTAIRS":
        return "ขึ้นบันได";
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

  // ฟังก์ชันเลือกสีตามป้ายกำกับ
  Color _getLabelColor(String label) {
    switch (label) {
      case "DOWNSTAIRS":
        return Colors.blue;
      case "UPSTAIRS":
        return Colors.green;
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
      appBar: AppBar(title: const Text('Stairs Classifier')),
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
              key: ValueKey('${_running}_${_label}_${_conf}'), // Force rebuild
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
                  _buildClassChip("DOWNSTAIRS", "ลงบันได", Colors.blue),
                  _buildClassChip("UPSTAIRS", "ขึ้นบันได", Colors.green),
                  _buildClassChip("WALK", "เดิน", Colors.orange),
                  _buildClassChip("RUN", "วิ่ง", Colors.red),
                  _buildClassChip("IDLE", "อยู่นิ่ง", Colors.grey),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
