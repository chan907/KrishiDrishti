import 'dart:io';
import 'package:flutter/material.dart';
import '../main.dart';
import '../models/scan_result.dart';
import 'result_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  late Future<List<ScanResult>> _f = Services.history.all();
  String t(String k) => Services.app.t(k);

  void _reload() => setState(() => _f = Services.history.all());

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(t('history'))),
      body: FutureBuilder<List<ScanResult>>(
        future: _f,
        builder: (_, snap) {
          final items = snap.data ?? [];
          if (snap.connectionState != ConnectionState.done) return const Center(child: CircularProgressIndicator());
          if (items.isEmpty) return Center(child: Text(t('no_history')));
          return ListView.separated(
            itemCount: items.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (_, i) {
              final r = items[i];
              final d = r.createdAt;
              return ListTile(
                leading: r.imagePath != null && File(r.imagePath!).existsSync()
                    ? ClipRRect(borderRadius: BorderRadius.circular(8), child: Image.file(File(r.imagePath!), width: 52, height: 52, fit: BoxFit.cover))
                    : const Icon(Icons.eco),
                title: Text(r.reliable ? '${r.crop} — ${r.disease}' : t('uncertain_msg').split('—').first.trim()),
                subtitle: Text('${d.day}/${d.month}/${d.year}  •  ${(r.confidence * 100).toStringAsFixed(0)}%'
                    '${r.severityLevel != null ? '  •  ${r.severityLevel}' : ''}'),
                onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => ResultScreen(result: r))),
                trailing: IconButton(
                  icon: const Icon(Icons.delete_outline),
                  tooltip: t('delete'),
                  onPressed: () async { await Services.history.removeAt(i); _reload(); },
                ),
              );
            },
          );
        },
      ),
    );
  }
}
