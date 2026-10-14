/// One analysed scan (from the server OR from the on-device model).
class ScanResult {
  final String source; // 'server' | 'offline'
  final bool reliable;
  final String? message;
  final String? crop;
  final String? disease;
  final String? classId;
  final double confidence;
  final double? severityPercent;
  final String? severityLevel;
  final String? severityNote;
  final Map<String, dynamic>? advisory;
  final String? gradcamUrl;
  final String? severityUrl;
  final List<Map<String, dynamic>> topK;
  final List<String> warnings;
  final String modelVersion;
  final String? imagePath; // local copy of the photo
  final DateTime createdAt;

  ScanResult({
    required this.source,
    required this.reliable,
    this.message,
    this.crop,
    this.disease,
    this.classId,
    required this.confidence,
    this.severityPercent,
    this.severityLevel,
    this.severityNote,
    this.advisory,
    this.gradcamUrl,
    this.severityUrl,
    this.topK = const [],
    this.warnings = const [],
    required this.modelVersion,
    this.imagePath,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  factory ScanResult.fromServer(Map<String, dynamic> j, String baseUrl, String? imagePath) {
    final sev = j['severity'] as Map<String, dynamic>?;
    String? abs(String? p) => p == null ? null : '$baseUrl$p';
    return ScanResult(
      source: 'server',
      reliable: j['reliable'] == true,
      message: j['message'],
      crop: j['crop'],
      disease: j['disease'],
      classId: j['class_id'],
      confidence: (j['confidence'] as num).toDouble(),
      severityPercent: (sev?['percent'] as num?)?.toDouble(),
      severityLevel: sev?['level'],
      severityNote: sev?['note'],
      advisory: j['advisory'] as Map<String, dynamic>?,
      gradcamUrl: abs(j['gradcam_url']),
      severityUrl: abs(j['severity_url']),
      topK: (j['top_k'] as List).cast<Map<String, dynamic>>(),
      warnings: (j['warnings'] as List).cast<String>(),
      modelVersion: j['model_version'],
      imagePath: imagePath,
    );
  }

  Map<String, dynamic> toJson() => {
        'source': source, 'reliable': reliable, 'message': message, 'crop': crop, 'disease': disease,
        'classId': classId, 'confidence': confidence, 'severityPercent': severityPercent,
        'severityLevel': severityLevel, 'severityNote': severityNote, 'advisory': advisory,
        'gradcamUrl': gradcamUrl, 'severityUrl': severityUrl, 'topK': topK, 'warnings': warnings,
        'modelVersion': modelVersion, 'imagePath': imagePath, 'createdAt': createdAt.toIso8601String(),
      };

  factory ScanResult.fromJson(Map<String, dynamic> j) => ScanResult(
        source: j['source'], reliable: j['reliable'], message: j['message'], crop: j['crop'],
        disease: j['disease'], classId: j['classId'], confidence: (j['confidence'] as num).toDouble(),
        severityPercent: (j['severityPercent'] as num?)?.toDouble(), severityLevel: j['severityLevel'],
        severityNote: j['severityNote'], advisory: (j['advisory'] as Map?)?.cast<String, dynamic>(),
        gradcamUrl: j['gradcamUrl'], severityUrl: j['severityUrl'],
        topK: (j['topK'] as List).map((e) => (e as Map).cast<String, dynamic>()).toList(),
        warnings: (j['warnings'] as List).cast<String>(), modelVersion: j['modelVersion'],
        imagePath: j['imagePath'], createdAt: DateTime.parse(j['createdAt']),
      );
}
