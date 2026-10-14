import 'dart:convert';
import 'package:flutter/services.dart';

/// Local copy of the curated knowledge base (same JSON the backend uses) -> advisory works offline.
class KbService {
  late Map<String, dynamic> _kb;

  Future<void> init() async =>
      _kb = jsonDecode(await rootBundle.loadString('assets/kb/knowledge_base.json'));

  dynamic _pick(dynamic d, String lang) => d is Map ? (d[lang] ?? d['en']) : d;

  List<Map<String, dynamic>> crops() => (_kb['crops'] as Map<String, dynamic>)
      .entries.map((e) => {'id': e.key, 'name': e.value['name']}).toList();

  String cropName(String id, String lang) {
    final c = (_kb['crops'] as Map)[id];
    return c == null ? id.replaceAll('_', ' ') : _pick(c['name'], lang);
  }

  String diseaseName(String classId, String lang) {
    final d = (_kb['diseases'] as Map)[classId];
    return d == null ? classId.split('___').last.replaceAll('_', ' ') : _pick(d['name'], lang);
  }

  Map<String, dynamic>? advisory(String classId, String? level, String lang) {
    final d = (_kb['diseases'] as Map)[classId];
    if (d == null) return null;
    final cat = (_kb['categories'] as Map)[d['category']];
    final out = <String, dynamic>{
      'description': _pick(d['description'], lang),
      'symptoms': _pick(d['symptoms'], lang),
      'cause': d['cause'],
      'prevention': _pick(cat['prevention'], lang),
      'general_actions': _pick(cat['general_actions'], lang),
      'monitoring': _pick(_kb['monitoring'], lang),
      'consult_expert': _pick(_kb['consult_expert'], lang),
    };
    if (level == 'High' || level == 'Severe') out['severity_note'] = _pick(_kb['high_severity_note'], lang);
    return out;
  }
}
