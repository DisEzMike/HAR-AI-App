import 'signal_utils.dart';

class ImuSample {
  final DateTime t;
  final double ax, ay, az, gx, gy, gz;
  ImuSample(this.t, this.ax, this.ay, this.az, this.gx, this.gy, this.gz);
}

class FeatureExtractor {
  final double gravityCutHz;
  FeatureExtractor({this.gravityCutHz = 0.5});

  /// ฟีเจอร์ 8 ตัวให้ตรงกับฝั่งเทรน
  Map<String, double> computeFeatures(List<ImuSample> w) {
    if (w.isEmpty) return _zeroFeatures();

    final tSec = w.map((s) => s.t.millisecondsSinceEpoch / 1000.0).toList();
    final fs = estimateFs(tSec);
    final ax = w.map((s) => s.ax).toList();
    final ay = w.map((s) => s.ay).toList();
    final az = w.map((s) => s.az).toList();
    final gx = w.map((s) => s.gx).toList();
    final gy = w.map((s) => s.gy).toList();
    final gz = w.map((s) => s.gz).toList();

    // auto-detect g→m/s^2
    final amag = mag3(ax, ay, az);
    final medMag = _median(amag);
    if (medMag > 0.5 && medMag < 2.0) {
      for (int i = 0; i < ax.length; i++) {
        ax[i] *= 9.81;
        ay[i] *= 9.81;
        az[i] *= 9.81;
      }
    }

    // gravity (EMA) + linear acc
    final gxLp = emaLowpass(ax, fs, fc: gravityCutHz);
    final gyLp = emaLowpass(ay, fs, fc: gravityCutHz);
    final gzLp = emaLowpass(az, fs, fc: gravityCutHz);
    final axLin = minus(ax, gxLp);
    final ayLin = minus(ay, gyLp);
    final azLin = minus(az, gzLp);
    final lam = mag3(axLin, ayLin, azLin); // lin_acc_mag
    final gyroM = mag3(gx, gy, gz); // gyro_mag

    // jerk magnitude
    final dt = 1.0 / (fs > 0 ? fs : 50.0);
    final jx = derivative(axLin, dt);
    final jy = derivative(ayLin, dt);
    final jz = derivative(azLin, dt);
    final jerk = mag3(jx, jy, jz);

    // vertical / horizontal
    final gMean = norm3(
      gxLp.reduce((a, b) => a + b) / gxLp.length,
      gyLp.reduce((a, b) => a + b) / gyLp.length,
      gzLp.reduce((a, b) => a + b) / gzLp.length,
    );
    final vSig = projectOnAxis(axLin, ayLin, azLin, gMean); // vertical (signed)
    final hSig = horizFromVert(axLin, ayLin, azLin, vSig);
    final vertRms = rms(vSig);
    final horizRms = rms(hSig);
    final ratioVH = vertRms / (horizRms + 1e-9);

    // spectrum features
    final domF = dominantFreqGoertzel(lam, fs, fmin: 0.5, fmax: 6.0, bins: 64);
    final bandE = bandEnergyGoertzel(lam, fs, 0.5, 5.0, bins: 64);
    final specEn = spectralEntropyGoertzel(lam, fs, bins: 64, fmin: 0.2);

    // Additional missing features
    final vertMean = vSig.fold(0.0, (a, b) => a + b) / vSig.length;
    final vertSorted = List<double>.from(vSig)..sort();
    final vertP05 = _percentile(vertSorted, 0.05);
    final vertP95 = _percentile(vertSorted, 0.95);
    
    // Impulse calculations
    final vertImpulse = vSig.where((x) => x.abs() > vertRms).length.toDouble();
    final vertPosImpulse = vSig.where((x) => x > vertRms).length.toDouble();
    final vertNegImpulse = vSig.where((x) => x < -vertRms).length.toDouble();
    final vertImpulseRatio = vertPosImpulse / (vertNegImpulse + 1e-9);
    final vertPeakRatio = (vertP95 - vertP05) / (vertRms + 1e-9);
    
    // Jerk features
    final vjerkProjected = projectOnAxis(jx, jy, jz, gMean);
    final vjerkRms = rms(vjerkProjected);
    
    // Balance features (placeholder - need more context for proper calculation)
    final vertBalanceBp = vertRms * 0.1; // Simplified
    final peakBalanceBp = (vertP95 + vertP05.abs()) * 0.1; // Simplified

    return {
      "lin_acc_mag_dom_freq": domF,
      "lin_acc_mag_rms": rms(lam),
      "vert_horiz_ratio": ratioVH,
      "vert_lin_rms": vertRms,
      "horiz_lin_rms": horizRms,
      "lin_acc_mag_band05_5Hz": bandE,
      "gyro_mag_rms": rms(gyroM),
      "lin_acc_mag_spec_entropy": specEn,
      // Missing features added
      "vert_lin_mean": vertMean,
      "vert_impulse": vertImpulse,
      "vert_pos_impulse": vertPosImpulse,
      "vert_neg_impulse": vertNegImpulse,
      "vert_impulse_ratio": vertImpulseRatio,
      "vert_p95": vertP95,
      "vert_p05": vertP05,
      "vert_peak_ratio": vertPeakRatio,
      "vjerk_rms": vjerkRms,
      "vert_balance_bp": vertBalanceBp,
      "peak_balance_bp": peakBalanceBp,
      "file_key": 0.0, // Placeholder
    };
  }

  Map<String, double> _zeroFeatures() => {
    "lin_acc_mag_dom_freq": 0.0,
    "lin_acc_mag_rms": 0.0,
    "vert_horiz_ratio": 0.0,
    "vert_lin_rms": 0.0,
    "horiz_lin_rms": 0.0,
    "lin_acc_mag_band05_5Hz": 0.0,
    "gyro_mag_rms": 0.0,
    "lin_acc_mag_spec_entropy": 0.0,
    "vert_lin_mean": 0.0,
    "vert_impulse": 0.0,
    "vert_pos_impulse": 0.0,
    "vert_neg_impulse": 0.0,
    "vert_impulse_ratio": 0.0,
    "vert_p95": 0.0,
    "vert_p05": 0.0,
    "vert_peak_ratio": 0.0,
    "vjerk_rms": 0.0,
    "vert_balance_bp": 0.0,
    "peak_balance_bp": 0.0,
    "file_key": 0.0,
  };

  double _median(List<double> x) {
    if (x.isEmpty) return 0.0;
    final y = List<double>.from(x)..sort();
    final m = y.length ~/ 2;
    return y.length.isOdd ? y[m] : 0.5 * (y[m - 1] + y[m]);
  }

  double _percentile(List<double> sortedList, double p) {
    if (sortedList.isEmpty) return 0.0;
    final index = (sortedList.length - 1) * p;
    final lower = index.floor();
    final upper = index.ceil();
    
    if (lower == upper) {
      return sortedList[lower];
    }
    
    final weight = index - lower;
    return sortedList[lower] * (1 - weight) + sortedList[upper] * weight;
  }
}
