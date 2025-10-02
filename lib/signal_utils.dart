import 'dart:math' as math;

double mean(List<double> x) =>
    x.isEmpty ? 0.0 : x.reduce((a, b) => a + b) / x.length;

double rms(List<double> x) {
  if (x.isEmpty) return 0.0;
  double s = 0;
  for (final v in x) {
    s += v * v;
  }
  return math.sqrt(s / x.length);
}

List<double> minus(List<double> a, List<double> b) {
  final n = math.min(a.length, b.length);
  return List<double>.generate(n, (i) => a[i] - b[i]);
}

List<double> mag3(List<double> ax, List<double> ay, List<double> az) {
  final n = math.min(ax.length, math.min(ay.length, az.length));
  return List<double>.generate(
    n,
    (i) => math.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]),
  );
}

List<double> derivative(List<double> x, double dt) {
  if (x.isEmpty) return <double>[];
  final out = List<double>.filled(x.length, 0.0);
  for (int i = 1; i < x.length; i++) {
    out[i] = (x[i] - x[i - 1]) / dt;
  }
  if (x.length > 1) out[0] = out[1];
  return out;
}

/// EMA low-pass (ประมาณ gravity), fc ~ 0.5 Hz
List<double> emaLowpass(List<double> x, double fs, {double fc = 0.5}) {
  if (x.isEmpty || fs <= 0) return List<double>.filled(x.length, 0.0);
  final rc = 1.0 / (2 * math.pi * fc);
  final dt = 1.0 / fs;
  final alpha = dt / (rc + dt);
  final y = List<double>.filled(x.length, 0.0);
  y[0] = x[0];
  for (int i = 1; i < x.length; i++) {
    y[i] = y[i - 1] + alpha * (x[i] - y[i - 1]);
  }
  return y;
}

List<double> norm3(double x, double y, double z, {double eps = 1e-9}) {
  final n = math.sqrt(x * x + y * y + z * z);
  if (n <= eps || n.isNaN) return [0.0, 0.0, 0.0];
  return [x / n, y / n, z / n];
}

/// โปรเจ็กต์ (ax,ay,az) ไปตามแกน u (unit) → สัญญาณแนวดิ่ง (signed)
List<double> projectOnAxis(
  List<double> ax,
  List<double> ay,
  List<double> az,
  List<double> u,
) {
  final n = math.min(ax.length, math.min(ay.length, az.length));
  final out = List<double>.filled(n, 0.0);
  for (int i = 0; i < n; i++) {
    out[i] = ax[i] * u[0] + ay[i] * u[1] + az[i] * u[2];
  }
  return out;
}

/// แนวนอน RMS = sqrt(|lin|^2 - |v|^2)
List<double> horizFromVert(
  List<double> axLin,
  List<double> ayLin,
  List<double> azLin,
  List<double> v,
) {
  final magLin = mag3(axLin, ayLin, azLin);
  final n = math.min(magLin.length, v.length);
  final out = List<double>.filled(n, 0.0);
  for (int i = 0; i < n; i++) {
    final hori2 = math.max(
      0.0,
      magLin[i] * magLin[i] - (v[i].abs() * v[i].abs()),
    );
    out[i] = math.sqrt(hori2);
  }
  return out;
}

/// ประเมิน fs จาก timestamp (วินาที)
double estimateFs(List<double> tSec) {
  if (tSec.length < 2) return 50.0;
  final dt = <double>[];
  for (int i = 1; i < tSec.length; i++) {
    final d = tSec[i] - tSec[i - 1];
    if (d > 0) dt.add(d);
  }
  if (dt.isEmpty) return 50.0;
  dt.sort();
  final med = dt[dt.length ~/ 2];
  return med > 0 ? 1.0 / med : 50.0;
}

/// Goertzel power ที่ความถี่ f
double goertzelPower(List<double> x, double fs, double f) {
  if (x.length < 8 || fs <= 0 || f <= 0) return 0.0;
  final n = x.length;
  final k = (0.5 + (n * f / fs)).floor();
  final w = 2 * math.pi * k / n;
  final cw = math.cos(w), sw = math.sin(w);
  final coeff = 2 * cw;
  double s0 = 0, s1 = 0, s2 = 0;
  final meanX = mean(x);
  for (int i = 0; i < n; i++) {
    s0 = x[i] - meanX + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  final real = s1 - s2 * cw;
  final imag = s2 * sw;
  return real * real + imag * imag;
}

/// dominant frequency ในช่วง [fmin,fmax]
double dominantFreqGoertzel(
  List<double> x,
  double fs, {
  double fmin = 0.5,
  double fmax = 6.0,
  int bins = 64,
}) {
  if (x.length < 8 || fs <= 0) return 0.0;
  fmax = math.min(fmax, fs / 2 - 1e-6);
  double bestF = 0.0, bestP = -1.0;
  for (int i = 0; i < bins; i++) {
    final f = fmin + (fmax - fmin) * i / (bins - 1);
    final p = goertzelPower(x, fs, f);
    if (p > bestP) {
      bestP = p;
      bestF = f;
    }
  }
  return bestF;
}

/// พลังงานรวมในย่าน [fmin,fmax]
double bandEnergyGoertzel(
  List<double> x,
  double fs,
  double fmin,
  double fmax, {
  int bins = 64,
}) {
  if (x.length < 8 || fs <= 0) return 0.0;
  fmax = math.min(fmax, fs / 2 - 1e-6);
  double s = 0.0;
  for (int i = 0; i < bins; i++) {
    final f = fmin + (fmax - fmin) * i / (bins - 1);
    s += goertzelPower(x, fs, f);
  }
  return s / x.length;
}

/// Spectral entropy
double spectralEntropyGoertzel(
  List<double> x,
  double fs, {
  int bins = 64,
  double fmin = 0.2,
}) {
  if (x.length < 8 || fs <= 0) return 0.0;
  final fmax = fs / 2 - 1e-6;
  final P = List<double>.filled(bins, 0.0);
  double sumP = 0.0;
  for (int i = 0; i < bins; i++) {
    final f = fmin + (fmax - fmin) * i / (bins - 1);
    final p = goertzelPower(x, fs, f);
    P[i] = p;
    sumP += p;
  }
  if (sumP <= 0) return 0.0;
  double H = 0.0;
  for (final p in P) {
    final q = p / sumP;
    if (q > 0) H += -q * math.log(q);
  }
  final hnorm = H / math.log(bins);
  return hnorm.isFinite ? hnorm : 0.0;
}
