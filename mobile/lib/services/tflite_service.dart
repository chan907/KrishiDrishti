import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/scan_result.dart';
import 'kb_service.dart';

/// On-device inference. Preprocessing mirrors krishi_core.preprocess:
/// EXIF-rotate -> RGB -> resize (bilinear) -> float32 0..255 (normalisation is INSIDE the model).
/// NOTE: the Dart `image` package resize differs very slightly from tf.image.resize. Validate on
/// device (see docs) before claiming identical accuracy to the server model.
class TfliteService {
  Interpreter? _interp;
  Map<String, dynamic> _meta = {};
  List<String> _classes = [];
  int _size = 224;

  bool get available => _interp != null;

  Future<void> init() async {
    try {
      _meta = jsonDecode(await rootBundle.loadString('assets/model/model_meta.json'));
      _classes = (_meta['class_names'] as List).cast<String>();
      _size = _meta['input_size'] ?? 224;
      _interp = await Interpreter.fromAsset('assets/model/model.tflite');
    } catch (_) {
      _interp = null; // model not bundled - app still works online
    }
  }

  Future<ScanResult> predict(File file, KbService kb, String lang, String? selectedCrop) async {
    final bytes = await file.readAsBytes();
    var image = img.decodeImage(bytes);
    if (image == null) throw Exception('Unreadable image');
    image = img.bakeOrientation(image);
    final resized = img.copyResize(image, width: _size, height: _size, interpolation: img.Interpolation.linear);

    final input = Float32List(_size * _size * 3);
    var i = 0;
    for (var y = 0; y < _size; y++) {
      for (var x = 0; x < _size; x++) {
        final p = resized.getPixel(x, y);
        input[i++] = p.r.toDouble();
        input[i++] = p.g.toDouble();
        input[i++] = p.b.toDouble();
      }
    }
    final output = List.filled(_classes.length, 0.0).reshape([1, _classes.length]);
    _interp!.run(input.reshape([1, _size, _size, 3]), output);
    final probs = (output[0] as List).map((e) => (e as num).toDouble()).toList();
    return _analyze(probs, kb, lang, selectedCrop, file.path);
  }

  // Dart port of krishi_core.uncertainty.analyze_probs
  ScanResult _analyze(List<double> p, KbService kb, String lang, String? selectedCrop, String path) {
    final u = (_meta['uncertainty'] as Map?) ?? {};
    final tConf = (u['confidence'] as num?)?.toDouble() ?? 0.80;
    final tMargin = (u['margin'] as num?)?.toDouble() ?? 0.20;
    final tEnt = (u['entropy'] as num?)?.toDouble() ?? 0.60;
    final order = List<int>.generate(p.length, (k) => k)..sort((a, b) => p[b].compareTo(p[a]));
    final top = order.first;
    final conf = p[top];
    final margin = p.length > 1 ? conf - p[order[1]] : conf;
    var ent = 0.0;
    for (final v in p) { ent -= v * math.log(v + 1e-12); }
    ent = p.length > 1 ? ent / math.log(p.length) : 0.0;
    final classId = _classes[top];
    final crop = classId.split('___').first;
    final warnings = <String>[];
    var reliable = conf >= tConf && margin >= tMargin && ent <= tEnt;
    if (selectedCrop != null && crop != selectedCrop) {
      reliable = false;
      warnings.add('predicted crop differs from selected crop');
    }
    final topK = order.take(3).map((k) => {
          'class_id': _classes[k],
          'label': '${kb.cropName(_classes[k].split('___').first, lang)} - ${kb.diseaseName(_classes[k], lang)}',
          'probability': p[k],
        }).toList();
    return ScanResult(
      source: 'offline',
      reliable: reliable,
      crop: reliable ? kb.cropName(crop, lang) : null,
      disease: reliable ? kb.diseaseName(classId, lang) : null,
      classId: reliable ? classId : null,
      confidence: conf,
      advisory: reliable ? kb.advisory(classId, null, lang) : null,
      topK: topK,
      warnings: warnings,
      modelVersion: '${_meta['model_version'] ?? 'unknown'} (tflite)',
      imagePath: path,
    );
  }
}
