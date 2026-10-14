import 'package:flutter/material.dart';
import '../main.dart';
import '../services/app_state.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final _url = TextEditingController(text: Services.app.serverUrl);
  AppState get app => Services.app;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: app,
      builder: (_, __) => Scaffold(
        appBar: AppBar(title: Text(app.t('settings'))),
        body: ListView(padding: const EdgeInsets.all(16), children: [
          Text(app.t('language'), style: const TextStyle(fontWeight: FontWeight.bold)),
          for (final e in AppState.supportedLangs.entries)
            RadioListTile<String>(title: Text(e.value), value: e.key, groupValue: app.lang, onChanged: (v) => app.setLang(v!)),
          const Divider(),
          SwitchListTile(
            title: Text(app.t('offline_mode')),
            subtitle: Services.tflite.available ? null : Text(app.t('model_missing')),
            value: app.offlineMode,
            onChanged: Services.tflite.available ? (v) => app.setOffline(v) : null,
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _url,
            decoration: InputDecoration(labelText: app.t('server_url'), border: const OutlineInputBorder(), hintText: 'http://192.168.1.10:8000'),
            keyboardType: TextInputType.url,
            onSubmitted: app.setServer,
            onEditingComplete: () => app.setServer(_url.text),
          ),
        ]),
      ),
    );
  }
}
