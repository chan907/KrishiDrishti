import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/scan_result.dart';

/// Local prediction history (stores metadata + result snapshot only; no identity, no location).
class HistoryStore {
  static const _key = 'history_v1';

  Future<List<ScanResult>> all() async {
    final p = await SharedPreferences.getInstance();
    return (p.getStringList(_key) ?? []).map((s) => ScanResult.fromJson(jsonDecode(s))).toList();
  }

  Future<void> add(ScanResult r) async {
    final p = await SharedPreferences.getInstance();
    final list = p.getStringList(_key) ?? [];
    list.insert(0, jsonEncode(r.toJson()));
    await p.setStringList(_key, list.take(200).toList());
  }

  Future<void> removeAt(int index) async {
    final p = await SharedPreferences.getInstance();
    final list = p.getStringList(_key) ?? [];
    if (index < list.length) list.removeAt(index);
    await p.setStringList(_key, list);
  }
}
