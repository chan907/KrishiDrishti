import 'package:flutter_test/flutter_test.dart';
import 'package:krishidrishti/models/scan_result.dart';

void main() {
  test('ScanResult survives JSON round trip (history storage)', () {
    final r = ScanResult(source: 'offline', reliable: true, crop: 'Tomato', disease: 'Early Blight',
        classId: 'Tomato___Early_blight', confidence: 0.92, modelVersion: '1.0.0');
    final back = ScanResult.fromJson(r.toJson());
    expect(back.disease, 'Early Blight');
    expect(back.confidence, 0.92);
    expect(back.source, 'offline');
  });

  test('server JSON maps to ScanResult with absolute media URLs', () {
    final r = ScanResult.fromServer({
      'reliable': true, 'crop': 'Potato', 'disease': 'Late Blight', 'class_id': 'Potato___Late_blight',
      'confidence': 0.95, 'severity': {'percent': 12.5, 'level': 'Moderate', 'note': null},
      'gradcam_url': '/media/a.png', 'severity_url': null, 'top_k': [], 'warnings': [], 'model_version': '1.0.0',
    }, 'http://host:8000', null);
    expect(r.gradcamUrl, 'http://host:8000/media/a.png');
    expect(r.severityLevel, 'Moderate');
  });
}
