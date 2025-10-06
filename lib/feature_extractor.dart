import 'dart:math' as math;

class ImuSample {
  final DateTime t;
  final double ax, ay, az;
  // Removed gyroscope data - accelerometer only according to mobile_config.json
  ImuSample(this.t, this.ax, this.ay, this.az);
}

class FeatureExtractor {
  /// สร้าง time series data [100, 4] จาก sensor readings - Accelerometer only
  static List<List<double>> prepareTimeSeriesData(List<Map<String, double>> sensorData) {
    print('=== Feature Extraction for [100, 4] Input (3 Classes) ===');
    print('Raw sensor data length: ${sensorData.length}');
    print('Target: IDLE, RUN, WALK (STAIRS removed)');
    
    final timeSeriesData = <List<double>>[];
    const targetLength = 100; // จาก mobile_config.json: window_size = 100
    
    for (int i = 0; i < targetLength; i++) {
      if (i < sensorData.length) {
        final sample = sensorData[i];
        final ax = sample['ax'] ?? 0.0;
        final ay = sample['ay'] ?? 0.0;
        final az = sample['az'] ?? 0.0;
        
        // Calculate accelerometer magnitude (4th feature)
        final accMag = math.sqrt(ax * ax + ay * ay + az * az);
        
        timeSeriesData.add([
          ax,     // accelerometer_x
          ay,     // accelerometer_y  
          az,     // accelerometer_z
          accMag, // accelerometer_magnitude
        ]);
      } else {
        // Pad with zeros if insufficient data
        timeSeriesData.add([0.0, 0.0, 0.0, 0.0]);
      }
    }
    
    print('Time series shape: [${timeSeriesData.length}, ${timeSeriesData.first.length}]');
    print('Features: [ax, ay, az, acc_magnitude]');
    print('Sample data: ${timeSeriesData.take(3).toList()}');
    
    return timeSeriesData;
  }

  /// Legacy method สำหรับ CSV export (อาจจะยังใช้) - Updated for accelerometer only
  @deprecated
  Map<String, double> computeFeatures(List<ImuSample> w) {
    if (w.isEmpty) return {};

    // แยกข้อมูลแต่ละแกน - accelerometer only
    final ax = w.map((s) => s.ax).toList();
    final ay = w.map((s) => s.ay).toList();
    final az = w.map((s) => s.az).toList();

    // คำนวณ magnitude - accelerometer only
    final accMag = <double>[];
    for (int i = 0; i < w.length; i++) {
      accMag.add(math.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]));
    }

    // สร้าง basic features dictionary สำหรับ CSV - 3 classes model
    final features = <String, double>{};

    // Basic statistics - accelerometer only (removed gyroscope)
    features['ax_mean'] = _mean(ax);
    features['ay_mean'] = _mean(ay);
    features['az_mean'] = _mean(az);
    features['acc_mag_mean'] = _mean(accMag);

    features['ax_std'] = _std(ax, features['ax_mean']!);
    features['ay_std'] = _std(ay, features['ay_mean']!);
    features['az_std'] = _std(az, features['az_mean']!);
    features['acc_mag_std'] = _std(accMag, features['acc_mag_mean']!);

    return features;
  }

  /// คำนวณ mean
  double _mean(List<double> data) {
    if (data.isEmpty) return 0.0;
    return data.reduce((a, b) => a + b) / data.length;
  }

  /// คำนวณ standard deviation
  double _std(List<double> data, double mean) {
    if (data.length <= 1) return 0.0;
    double variance = 0.0;
    for (final val in data) {
      variance += (val - mean) * (val - mean);
    }
    variance /= data.length;
    return math.sqrt(variance);
  }
}
