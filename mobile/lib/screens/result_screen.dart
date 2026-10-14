import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../main.dart';
import '../models/scan_result.dart';

class ResultScreen extends StatelessWidget {
  final ScanResult result;
  final String? bannerKey;
  const ResultScreen({super.key, required this.result, this.bannerKey});

  String t(String k) => Services.app.t(k);

  Future<void> _speak() async {
    final a = result.advisory;
    if (a == null) return;
    final tts = FlutterTts();
    final locale = {'en': 'en-IN', 'hi': 'hi-IN', 'mr': 'mr-IN'}[Services.app.lang] ?? 'en-IN';
    await tts.setLanguage(locale);
    final text = [result.disease, a['description'], ...(a['general_actions'] as List? ?? [])].join('. ');
    await tts.speak(text);
  }

  Widget _section(String title, dynamic body) {
    if (body == null) return const SizedBox.shrink();
    final lines = body is List ? body.cast<String>() : [body.toString()];
    return Padding(
      padding: const EdgeInsets.only(top: 14),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
        const SizedBox(height: 4),
        for (final l in lines) Padding(padding: const EdgeInsets.symmetric(vertical: 2), child: Text(body is List ? '• $l' : l)),
      ]),
    );
  }

  Widget _image(String? url, String? localPath) {
    if (url != null) return Image.network(url, fit: BoxFit.cover, errorBuilder: (_, __, ___) => const SizedBox());
    if (localPath != null && File(localPath).existsSync()) return Image.file(File(localPath), fit: BoxFit.cover);
    return const SizedBox();
  }

  @override
  Widget build(BuildContext context) {
    final r = result;
    final a = r.advisory;
    final pct = (r.confidence * 100).toStringAsFixed(1);
    return Scaffold(
      appBar: AppBar(title: Text(t('app_title'))),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        if (bannerKey != null) Card(color: Colors.amber.shade100, child: Padding(padding: const EdgeInsets.all(12), child: Text(t(bannerKey!)))),
        if (r.imagePath != null) ClipRRect(borderRadius: BorderRadius.circular(12), child: SizedBox(height: 220, width: double.infinity, child: _image(null, r.imagePath))),
        const SizedBox(height: 12),
        if (r.reliable) ...[
          Text('${r.crop} — ${r.disease}', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 6),
          Text('${t('confidence')}: $pct%'),
        ] else ...[
          Card(color: Colors.orange.shade50, child: Padding(padding: const EdgeInsets.all(14), child: Text(t('uncertain_msg')))),
          const SizedBox(height: 8),
          Text(t('possibilities'), style: const TextStyle(fontWeight: FontWeight.bold)),
          for (final k in r.topK) Text('• ${k['label']} (${((k['probability'] as num) * 100).toStringAsFixed(1)}%)'),
        ],
        Text(r.source == 'server' ? t('source_server') : t('source_offline'), style: const TextStyle(color: Colors.grey, fontSize: 12)),
        if (r.reliable) ...[
          const Divider(height: 28),
          if (r.severityPercent != null) ...[
            Text('${t('severity')}: ${r.severityLevel}', style: const TextStyle(fontWeight: FontWeight.bold)),
            if (r.severityLevel != 'None') Text('${t('affected_area')}: ${r.severityPercent!.toStringAsFixed(0)}%'),
          ] else
            Text(r.source == 'offline' ? t('severity_offline') : (r.severityNote != null ? t('severity_unavailable') : '')),
          if (r.severityUrl != null) ...[
            const SizedBox(height: 8),
            ClipRRect(borderRadius: BorderRadius.circular(12), child: SizedBox(height: 200, width: double.infinity, child: _image(r.severityUrl, null))),
            Text(t('lesion_view'), style: const TextStyle(fontSize: 12, color: Colors.grey)),
          ],
          if (r.gradcamUrl != null) ...[
            const SizedBox(height: 12),
            ClipRRect(borderRadius: BorderRadius.circular(12), child: SizedBox(height: 200, width: double.infinity, child: _image(r.gradcamUrl, null))),
            Text('${t('heatmap')}. ${t('heatmap_note')}', style: const TextStyle(fontSize: 12, color: Colors.grey)),
          ],
          if (a != null) ...[
            _section(r.disease ?? '', a['description']),
            _section(t('symptoms'), a['symptoms']),
            _section(t('actions'), a['general_actions']),
            _section(t('prevention'), a['prevention']),
            _section(t('monitoring'), a['monitoring']),
            if (a['severity_note'] != null) _section('!', a['severity_note']),
            _section(t('consult'), a['consult_expert']),
            if (a['translation_note'] != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(t('translation_review'), style: const TextStyle(fontSize: 12, color: Colors.grey))),
            const SizedBox(height: 8),
            TextButton.icon(onPressed: _speak, icon: const Icon(Icons.volume_up), label: Text(t('speak'))),
          ],
        ],
        for (final w in r.warnings) Text('⚠ $w', style: const TextStyle(fontSize: 12, color: Colors.grey)),
        const Divider(height: 28),
        Text(t('disclaimer'), style: const TextStyle(fontSize: 12)),
        Text('model: ${r.modelVersion}', style: const TextStyle(fontSize: 11, color: Colors.grey)),
      ]),
    );
  }
}
