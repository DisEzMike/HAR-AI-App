# TensorFlow Lite rules
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }
-keep class org.tensorflow.lite.delegates.** { *; }

# Keep GPU delegate related classes
-keep class org.tensorflow.lite.gpu.GpuDelegate { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegateFactory { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegateFactory$Options { *; }

# Suppress warnings for missing GPU delegate classes
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep TensorFlow Lite model related classes
-keep class org.tensorflow.lite.Interpreter { *; }
-keep class org.tensorflow.lite.Tensor { *; }
-keep class org.tensorflow.lite.DataType { *; }

# Flutter TensorFlow Lite plugin
-keep class io.flutter.plugins.** { *; }

# Sensor plugin classes
-keep class dev.fluttercommunity.plus.sensors.** { *; }