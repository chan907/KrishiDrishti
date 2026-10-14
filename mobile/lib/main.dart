import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'services/app_state.dart';
import 'services/history_store.dart';
import 'services/kb_service.dart';
import 'services/tflite_service.dart';

/// Tiny service locator (kept simple on purpose - no state-management dependency needed).
class Services {
  static final app = AppState();
  static final kb = KbService();
  static final tflite = TfliteService();
  static final history = HistoryStore();
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Services.app.init();
  await Services.kb.init();
  await Services.tflite.init();
  runApp(const KrishiApp());
}

class KrishiApp extends StatelessWidget {
  const KrishiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Services.app,
      builder: (_, __) => MaterialApp(
        title: 'KrishiDrishti',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(colorSchemeSeed: const Color(0xFF2E7D32), useMaterial3: true),
        home: const HomeScreen(),
      ),
    );
  }
}
