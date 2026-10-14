import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../main.dart';
import '../models/scan_result.dart';
import '../services/api_service.dart';
import 'history_screen.dart';
import 'result_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  String? _crop; // null = let model infer
  bool _busy = false;

  String t(String k) => Services.app.t(k);

  Future<void> _pick(ImageSource src) async {
    final x = await _picker.pickImage(source: src, maxWidth: 1600, imageQuality: 92);
    if (x == null) return;
    await _analyze(File(x.path));
  }

  Future<void> _analyze(File file) async {
    setState(() => _busy = true);
    final app = Services.app;
    ScanResult? result;
    String? bannerKey;
    try {
      if (app.offlineMode) {
        result = await _offline(file);
      } else {
        try {
          result = await ApiService(app.serverUrl).predict(file, crop: _crop, language: app.lang);
        } on ApiException catch (e) {
          if (e.network && Services.tflite.available) {
            result = await _offline(file); // graceful fallback: no internet / server down
            bannerKey = 'offline_used';
          } else {
            rethrow;
          }
        }
      }
      await Services.history.add(result);
      if (!mounted) return;
      await Navigator.push(context, MaterialPageRoute(builder: (_) => ResultScreen(result: result!, bannerKey: bannerKey)));
    } on ApiException catch (e) {
      _snack(e.message);
    } catch (_) {
      _snack(t('error_generic'));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<ScanResult> _offline(File f) async {
    if (!Services.tflite.available) throw ApiException(t('model_missing'));
    return Services.tflite.predict(f, Services.kb, Services.app.lang, _crop);
  }

  void _snack(String m) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(m), duration: const Duration(seconds: 6)));

  @override
  Widget build(BuildContext context) {
    final lang = Services.app.lang;
    final crops = Services.kb.crops();
    return Scaffold(
      appBar: AppBar(
        title: Text(t('app_title')),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: t('settings'),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const SettingsScreen())),
          ),
        ],
      ),
      body: _busy
          ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [const CircularProgressIndicator(), const SizedBox(height: 16), Text(t('analyzing'))]))
          : ListView(padding: const EdgeInsets.all(20), children: [
              Text(t('tagline'), style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 16),
              DropdownButtonFormField<String?>(
                value: _crop,
                decoration: InputDecoration(labelText: t('select_crop'), border: const OutlineInputBorder()),
                items: [
                  DropdownMenuItem(value: null, child: Text(t('any_crop'))),
                  for (final c in crops) DropdownMenuItem(value: c['id'] as String, child: Text((c['name'] as Map)[lang] ?? (c['name'] as Map)['en'])),
                ],
                onChanged: (v) => setState(() => _crop = v),
              ),
              const SizedBox(height: 20),
              FilledButton.icon(onPressed: () => _pick(ImageSource.camera), icon: const Icon(Icons.photo_camera), label: Padding(padding: const EdgeInsets.all(12), child: Text(t('take_photo')))),
              const SizedBox(height: 12),
              OutlinedButton.icon(onPressed: () => _pick(ImageSource.gallery), icon: const Icon(Icons.photo_library), label: Padding(padding: const EdgeInsets.all(12), child: Text(t('choose_gallery')))),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const HistoryScreen())),
                icon: const Icon(Icons.history),
                label: Padding(padding: const EdgeInsets.all(12), child: Text(t('history'))),
              ),
              const SizedBox(height: 24),
              Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(t('tips_title'), style: const TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Text(t('tips')),
              ]))),
            ]),
    );
  }
}
