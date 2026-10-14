import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// App-wide settings + localisation (strings live in assets/i18n/*.json - no per-language screens).
class AppState extends ChangeNotifier {
  static const supportedLangs = {'en': 'English', 'hi': 'हिन्दी', 'mr': 'मराठी'};
  late SharedPreferences _p;
  final Map<String, Map<String, String>> _strings = {};

  String lang = 'en';
  // 10.0.2.2 = host PC from the Android emulator. On a real phone use your PC's LAN IP.
  String serverUrl = 'http://10.0.2.2:8000';
  bool offlineMode = false;

  Future<void> init() async {
    _p = await SharedPreferences.getInstance();
    lang = _p.getString('lang') ?? 'en';
    serverUrl = _p.getString('server') ?? serverUrl;
    offlineMode = _p.getBool('offline') ?? false;
    for (final l in supportedLangs.keys) {
      final raw = await rootBundle.loadString('assets/i18n/$l.json');
      _strings[l] = (jsonDecode(raw) as Map<String, dynamic>).map((k, v) => MapEntry(k, v.toString()));
    }
  }

  String t(String key) => _strings[lang]?[key] ?? _strings['en']?[key] ?? key;

  Future<void> setLang(String l) async { lang = l; await _p.setString('lang', l); notifyListeners(); }
  Future<void> setServer(String s) async { serverUrl = s.trim().replaceAll(RegExp(r'/+$'), ''); await _p.setString('server', serverUrl); notifyListeners(); }
  Future<void> setOffline(bool v) async { offlineMode = v; await _p.setBool('offline', v); notifyListeners(); }
}
