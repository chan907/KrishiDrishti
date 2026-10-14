import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/scan_result.dart';

class ApiException implements Exception {
  final String message;
  final bool network; // true -> no internet / server down -> offline fallback allowed
  ApiException(this.message, {this.network = false});
  @override
  String toString() => message;
}

class ApiService {
  final String baseUrl;
  ApiService(this.baseUrl);

  Future<ScanResult> predict(File image, {String? crop, required String language}) async {
    try {
      final req = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/predict'))
        ..fields['language'] = language
        ..fields['explain'] = 'true'
        ..files.add(await http.MultipartFile.fromPath('image', image.path));
      if (crop != null) req.fields['crop'] = crop;
      final streamed = await req.send().timeout(const Duration(seconds: 60));
      final res = await http.Response.fromStream(streamed);
      final body = jsonDecode(utf8.decode(res.bodyBytes));
      if (res.statusCode == 200) return ScanResult.fromServer(body, baseUrl, image.path);
      final detail = body['detail'];
      final msg = detail is Map ? (detail['message'] ?? 'Request failed') : (detail ?? 'Request failed').toString();
      // 503 = model unavailable on server -> treat like a network problem so offline fallback can kick in
      throw ApiException(msg.toString(), network: res.statusCode == 503);
    } on SocketException {
      throw ApiException('Cannot reach the server.', network: true);
    } on TimeoutException {
      throw ApiException('Server did not respond in time.', network: true);
    } on http.ClientException {
      throw ApiException('Cannot reach the server.', network: true);
    }
  }
}
