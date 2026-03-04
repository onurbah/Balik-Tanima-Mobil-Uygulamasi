import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart' show compute;
import 'package:video_player/video_player.dart';

void main() {
  runApp(const MaterialApp(home: HomePage()));
}

// RGBA → normalize edilmiş Float32
Float32List rgbBytesToFloat32(List<int> rgbBytes) {
  final len = rgbBytes.length;
  final out = Float32List(len);
  for (int i = 0; i < len; i++) {
    out[i] = rgbBytes[i] / 255.0;
  }
  return out;
}

class Detection {
  final Rect box; // orijinal piksel koordinatı
  final String label;
  final double score;
  Detection({required this.box, required this.label, required this.score});
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  bool _showSplash = true;
  static const int kInput = 640;
  static const double kConfThresh = 0.15;   // modelin kabul edeceği minimum
  static const double kDisplayThresh = 0.50; // ekranda kutu çizilecek minimum

  File? _image;
  late Interpreter _interpreter;
  late Map<String, dynamic> _speciesInfo;
  bool _isModelReady = false;
  List<Detection> _detections = [];
  List<Detection> _lowConf = []; // düşük güvenli tahminler için
  Detection? _selectedDetection;

  // Loading state for model inference
  bool _isLoading = false;

  int? _origW, _origH;
  img.Image? _modelInput;
  double _scale = 1.0;
  double _padX = 0.0;
  double _padY = 0.0;

  // 🔍 buton nefes animasyonu
  late AnimationController _animController;
  late Animation<double> _scaleAnim;

  // Splash transition animation controller and animations
  late AnimationController _splashController;
  late Animation<Alignment> _textAlignment;
  late Animation<double> _textFontSize;
  late Animation<double> _fishOpacity;
  late Animation<double> _fishScale;

  final picker = ImagePicker();

  // Video background controller
  late VideoPlayerController _bgController;

  @override
  void initState() {
    super.initState();
    loadModelAndData();

    _animController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);

    _scaleAnim = Tween<double>(begin: 0.90, end: 1.18).animate(
      CurvedAnimation(parent: _animController, curve: Curves.easeInOut),
    );

    _bgController = VideoPlayerController.asset("assets/videos/background_fixed.mp4")
      ..setLooping(true)
      ..setVolume(0);
    print("▶️ Video controller init çağrıldı");
    _bgController.initialize().then((_) {
      print("✅ Video initialize oldu, play ediliyor");
      _bgController.play();
      setState(() {});
    });

    // Splash transition animations
    _splashController = AnimationController(vsync: this, duration: const Duration(seconds: 2));

    // Yazı ana ekrandaki boyutu ve konumuna uygun olacak şekilde tweener ayarları
    _textAlignment = AlignmentTween(
      begin: Alignment.center,
      end: const Alignment(0, -0.85), // AppBar başlığına ortalı hizalama
    ).animate(CurvedAnimation(parent: _splashController, curve: Curves.easeInOut));

    // Splash yazısı başlangıçta 28pt → 18pt küçülecek
    _textFontSize = Tween<double>(begin: 28, end: 18)
        .animate(CurvedAnimation(parent: _splashController, curve: Curves.easeInOut));

    _fishOpacity = Tween<double>(begin: 1.0, end: 0.0)
        .animate(CurvedAnimation(parent: _splashController, curve: Curves.easeInOut));
    _fishScale = Tween<double>(begin: 1.0, end: 5.0)
        .animate(CurvedAnimation(parent: _splashController, curve: Curves.easeInOut));

    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) _splashController.forward();
    });

    _splashController.addStatusListener((status) {
      if (status == AnimationStatus.completed && mounted) {
        setState(() {
          _showSplash = false;
        });
      }
    });
  }

  @override
  void dispose() {
    _animController.dispose();
    _splashController.dispose();
    if (_isModelReady) {
      _interpreter.close();
    }
    _bgController.dispose();
    super.dispose();
  }

  Future<void> loadModelAndData() async {
    final opts = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(
      'assets/model/best_yolov8n_float32.tflite',
      options: opts,
    );

    final jsonStr = await rootBundle.loadString('assets/data/species_info.json');
    final jsonData = json.decode(jsonStr);
    _speciesInfo = {for (var item in jsonData) item['label'].toLowerCase(): item};

    setState(() {
      _isModelReady = true;
    });
  }

  img.Image _letterbox(img.Image src) {
    final sw = src.width.toDouble();
    final sh = src.height.toDouble();
    final r = min(kInput / sw, kInput / sh);
    final nw = (sw * r).round();
    final nh = (sh * r).round();

    final resized = img.copyResize(src, width: nw, height: nh);
    final padX = ((kInput - nw) / 2).floor();
    final padY = ((kInput - nh) / 2).floor();

    final canvas = img.Image(kInput, kInput);
    img.fill(canvas, img.getColor(0, 0, 0));
    img.copyInto(canvas, resized, dstX: padX, dstY: padY);

    _scale = r;
    _padX = padX.toDouble();
    _padY = padY.toDouble();
    return canvas;
  }

  Future<void> pickImageFromSource(ImageSource source) async {
    final picked = await picker.pickImage(source: source);
    if (picked == null) return;

    final original = File(picked.path);
    final decoded = img.decodeImage(await original.readAsBytes());
    if (decoded == null) return;

    final modelInput = _letterbox(decoded);

    setState(() {
      _image = original;
      _origW = decoded.width;
      _origH = decoded.height;
      _modelInput = modelInput;
      _detections.clear();
      _lowConf.clear();
      _selectedDetection = null; // yeni görselde seçili balığı sıfırla
    });
  }

  Future<void> runModel() async {
    print("▶️ runModel başladı");
    if (!_isModelReady || _modelInput == null || _origW == null || _origH == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      // inputu hazırla
      final image = _modelInput!;
      final rgbaBytes = image.getBytes();
      final rgbBytes = <int>[];
      for (int i = 0; i < rgbaBytes.length; i += 4) {
        rgbBytes.add(rgbaBytes[i]);
        rgbBytes.add(rgbaBytes[i + 1]);
        rgbBytes.add(rgbaBytes[i + 2]);
      }
      final input = await compute(rgbBytesToFloat32, rgbBytes);
      final inputTensor = input.reshape([1, kInput, kInput, 3]);

      // çıktı shape [1,300,6] (x1,y1,x2,y2,score,class)
      final output = List.generate(1, (_) => List.generate(300, (_) => List.filled(6, 0.0)));
      _interpreter.run(inputTensor, output);
      print("✅ İnference bitti");

      final raw = output[0]; // [300,6]

      final shiftedLabels = <String>[
        'angelfish', 'bluetang', 'butterflyfish', 'clownfish',
        'goldfish', 'gourami', 'morishidol', 'platyfish',
        'ribbonedsweetlips', 'threestrippeddamselfish',
        'yellowcichlid', 'yellowtang', 'zebrafish'
      ];

      final List<Detection> dets = [];

      for (final pred in raw) {
        final x1 = pred[0] * kInput;
        final y1 = pred[1] * kInput;
        final x2 = pred[2] * kInput;
        final y2 = pred[3] * kInput;
        final score = pred[4];
        final clsIdx = pred[5].round();

        if (score < kConfThresh) continue;

        // Kutuları orijinale çevir
        double ox1 = (x1 - _padX) / _scale;
        double oy1 = (y1 - _padY) / _scale;
        double ox2 = (x2 - _padX) / _scale;
        double oy2 = (y2 - _padY) / _scale;

        // Kutuyu balığa daha yakın hale getir
        final boxW = ox2 - ox1;
        final boxH = oy2 - oy1;
        const tightenFactor = 0.02; // kutuyu içeri doğru orantılı sıkılaştır
        ox1 += boxW * tightenFactor;
        oy1 += boxH * tightenFactor;
        ox2 -= boxW * tightenFactor;
        oy2 -= boxH * tightenFactor;

        ox1 = ox1.clamp(0.0, _origW!.toDouble());
        oy1 = oy1.clamp(0.0, _origH!.toDouble());
        ox2 = ox2.clamp(0.0, _origW!.toDouble());
        oy2 = oy2.clamp(0.0, _origH!.toDouble());

        final label = (clsIdx >= 0 && clsIdx < shiftedLabels.length)
            ? shiftedLabels[clsIdx]
            : 'unknown';

        final rect = Rect.fromLTRB(ox1, oy1, ox2, oy2);
        if (!rect.isFinite || rect.width <= 1 || rect.height <= 1) continue;

        dets.add(Detection(
          box: rect,
          label: label,
          score: score,
        ));
      }

      // Listeyi ayır: yüksek güven kutular vs düşük güven yazıları
      final List<Detection> shownDets = [];
      final List<Detection> lowDets = [];
      for (final d in dets) {
        if (d.score >= kDisplayThresh) {
          shownDets.add(d);
        } else {
          lowDets.add(d);
        }
      }

      setState(() {
        _detections = shownDets;
        _lowConf = lowDets;
        _selectedDetection = null; // yeni sonuçta açık info varsa kapat
      });

      print("🎯 ${shownDets.length} kutu çizildi, ${lowDets.length} düşük güven bulundu");
    } finally {
      // Show loading overlay at least 1 second
      await Future.delayed(const Duration(seconds: 1));
      setState(() {
        _isLoading = false;
      });
    }
  }

  // 🔎 Seçili balık için JSON bilgisini getiren yardımcı
  Map<String, dynamic>? _infoFor(Detection d) {
    final key = d.label.toLowerCase();
    final v = _speciesInfo[key];
    return (v is Map<String, dynamic>) ? v : null;
  }

  // Show info dialog for a detection
  void _showInfoDialog(Detection d) {
    final info = _infoFor(d);
    showDialog(
      context: context,
      builder: (context) {
        return Dialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(24, 20, 24, 12),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        info != null
                            ? "${(info['name_tr'] ?? d.label)}"
                              "${info['name_en'] != null ? " (${info['name_en']})" : ""}"
                            : d.label,
                        style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text("🎯 Güven: ${(d.score * 100).toStringAsFixed(1)}%"),
                if (info != null && info['latin_name'] != null)
                  Text("🔬 ${info['latin_name']}"),
                if (info != null && info['habitat'] != null)
                  Text("🌍 Habitat: ${info['habitat']}"),
                if (info != null && info['size'] != null)
                  Text("📏 Boyut: ${info['size']}"),
                if (info != null && info['status'] != null)
                  Text("📊 Durum: ${info['status']}"),
                const SizedBox(height: 8),
                if (info != null && info['info_tr'] != null)
                  Text("📘 Açıklama: ${info['info_tr']}"),
                if (info != null && info['info_tr'] == null && info['info_en'] != null)
                  Text("📘 Info: ${info['info_en']}"),
                const SizedBox(height: 8),
              ],
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Video background layer
        _bgController.value.isInitialized
            ? (() {
                return SizedBox.expand(
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: _bgController.value.size.width,
                      height: _bgController.value.size.height,
                      child: VideoPlayer(_bgController),
                    ),
                  ),
                );
              })()
            : (() {
                return Container(color: Colors.black);
              })(),
        if (_showSplash)
          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(0.5),
              child: Stack(
                children: [
                  AlignTransition(
                    alignment: _textAlignment,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        AnimatedBuilder(
                          animation: _textFontSize,
                          builder: (context, child) {
                            return Hero(
                              tag: "appTitle",
                              child: Material(
                                type: MaterialType.transparency,
                                child: Text(
                                  "Sualtı Canlı Tanıma",
                                  style: TextStyle(
                                    fontSize: _textFontSize.value,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white,
                                    decoration: TextDecoration.none,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                              ),
                            );
                          },
                        ),
                        const SizedBox(height: 100),
                      ],
                    ),
                  ),
                  FadeTransition(
                    opacity: _fishOpacity,
                    child: ScaleTransition(
                      scale: _fishScale,
                      alignment: Alignment.center,
                      child: Align(
                        alignment: Alignment.center,
                        child: Image.asset("assets/images/fish.png", width: 80, height: 80),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        // Main content
        if (!_showSplash)
          Scaffold(
            backgroundColor: Colors.transparent,
            appBar: AppBar(
              backgroundColor: Colors.transparent,
              elevation: 0,
              bottomOpacity: 0,
              shadowColor: Colors.transparent,
              //automaticallyImplyLeading: false,
              actions: (_image != null)
                  ? [
                      IconButton(
                        icon: const Text(
                          "✕",
                          style: TextStyle(
                            fontSize: 26,
                            color: Colors.black,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        onPressed: () {
                          setState(() {
                            _image = null;
                            _detections.clear();
                            _lowConf.clear();
                            _selectedDetection = null;
                          });
                        },
                      ),
                    ]
                  : [],
              titleSpacing: 0,
              toolbarHeight: 60,
              title: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Hero(
                    tag: "appTitle",
                    child: Material(
                      type: MaterialType.transparency,
                      child: Text(
                        "Sualtı Canlı Tanıma",
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          decoration: TextDecoration.none,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 2),
                  const Text("Fotoğrafını çek, türünü öğren", style: TextStyle(fontSize: 12, color: Colors.white70)),
                ],
              ),
            ),
            body: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Column(
                children: [
                  // Görsel ve kutular
                  if (_image != null && _origW != null && _origH != null)
                    Expanded(
                      child: Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            AnimatedContainer(
                              duration: const Duration(milliseconds: 400),
                              curve: Curves.easeInOut,
                              width: MediaQuery.of(context).size.width * 0.9,
                              height: min(
                                (MediaQuery.of(context).size.width * 0.9) * (_origH! / _origW!),
                                MediaQuery.of(context).size.height * 0.63,
                              ),
                              child: Card(
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                                elevation: 6,
                                margin: EdgeInsets.zero,
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(24),
                                  child: LayoutBuilder(
                                    builder: (context, constraints) {
                                      final frameW = constraints.maxWidth;
                                      final frameH = constraints.maxHeight;
                                      return Stack(
                                        children: [
                                          Image.file(
                                            _image!,
                                            fit: BoxFit.cover,
                                            width: double.infinity,
                                            height: double.infinity,
                                          ),
                                          // Draw bounding boxes only when not loading
                                          if (!_isLoading && _detections.isNotEmpty)
                                            Positioned.fill(
                                              child: CustomPaint(
                                                painter: BoundingBoxPainter(
                                                  detections: _detections,
                                                  imageRect: Rect.fromLTWH(0, 0, _origW!.toDouble(), _origH!.toDouble()),
                                                  origW: _origW!,
                                                  origH: _origH!,
                                                ),
                                              ),
                                            ),
                                          // 🔍 Büyüteç butonları overlay'i (only when not loading)
                                          if (!_isLoading)
                                            ..._detections.map((d) {
                                              // Hesaplama: BoundingBoxPainter'daki ile aynı mantık
                                              final imgRatio = _origW! / _origH!;
                                              final frameRatio = frameW / frameH;
                                              double scale, offsetX = 0, offsetY = 0;
                                              if (imgRatio > frameRatio) {
                                                scale = frameH / _origH!;
                                                offsetX = (_origW! * scale - frameW) / 2;
                                              } else {
                                                scale = frameW / _origW!;
                                                offsetY = (_origH! * scale - frameH) / 2;
                                              }
                                              final left = d.box.left * scale - offsetX;
                                              final top = d.box.top * scale - offsetY;
                                              final right = d.box.right * scale - offsetX;
                                              final buttonSize = 38.0;
                                              // Sağ üst köşe: (right, top)
                                              return Positioned(
                                                left: right - buttonSize / 2,
                                                top: top - buttonSize / 2,
                                                child: ScaleTransition(
                                                  scale: _scaleAnim,
                                                  child: Material(
                                                    color: Colors.transparent,
                                                    child: InkWell(
                                                      borderRadius: BorderRadius.circular(buttonSize / 2),
                                                      onTap: () => _showInfoDialog(d),
                                                      child: Container(
                                                        width: buttonSize,
                                                        height: buttonSize,
                                                        decoration: BoxDecoration(
                                                          color: Colors.transparent,
                                                          shape: BoxShape.circle,
                                                          boxShadow: [
                                                            BoxShadow(
                                                              color: Colors.black.withOpacity(0.11),
                                                              blurRadius: 4,
                                                              offset: Offset(1, 2),
                                                            ),
                                                          ],
                                                        ),
                                                        child: const Center(
                                                          child: Text(
                                                            "🔍",
                                                            style: TextStyle(fontSize: 22),
                                                          ),
                                                        ),
                                                      ),
                                                    ),
                                                  ),
                                                ),
                                              );
                                            }).toList(),
                                          // Loading overlay
                                          if (_isLoading)
                                            Positioned.fill(
                                              child: Container(
                                                color: Colors.black.withOpacity(0.45),
                                                child: Center(
                                                  child: Column(
                                                    mainAxisAlignment: MainAxisAlignment.center,
                                                    children: [
                                                      const CircularProgressIndicator(
                                                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                                      ),
                                                      const SizedBox(height: 18),
                                                      const Text(
                                                        "Yükleniyor...",
                                                        style: TextStyle(
                                                          color: Colors.white,
                                                          fontSize: 18,
                                                          fontWeight: FontWeight.w500,
                                                        ),
                                                      ),
                                                    ],
                                                  ),
                                                ),
                                              ),
                                            ),
                                          // Overlay: "Balık bulunamadı!" + düşük güvenli tahminler
                                          if (!_isLoading && _detections.isEmpty && _lowConf.isNotEmpty)
                                            Positioned.fill(
                                              child: Center(
                                                child: Container(
                                                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
                                                  decoration: BoxDecoration(
                                                    color: Colors.white.withOpacity(0.50),
                                                    borderRadius: BorderRadius.circular(22),
                                                  ),
                                                  child: Column(
                                                    mainAxisSize: MainAxisSize.min,
                                                    crossAxisAlignment: CrossAxisAlignment.center,
                                                    children: [
                                                      const Text(
                                                        "❗ Balık bulunamadı",
                                                        style: TextStyle(
                                                          color: Colors.red,
                                                          fontSize: 18,
                                                          fontWeight: FontWeight.w700,
                                                        ),
                                                      ),
                                                      const SizedBox(height: 10),
                                                      Column(
                                                        crossAxisAlignment: CrossAxisAlignment.center,
                                                        children: _lowConf
                                                            .map((d) => Text(
                                                                  "${d.label} (${(d.score * 100).toStringAsFixed(1)}%)",
                                                                  style: const TextStyle(
                                                                    color: Colors.black,
                                                                    fontSize: 16,
                                                                  ),
                                                                ))
                                                            .toList(),
                                                      ),
                                                    ],
                                                  ),
                                                ),
                                              ),
                                            ),
                                        ],
                                      );
                                    },
                                  ),
                                ),
                              ),
                            ),
                            // Tanı butonu doğrudan fotoğrafın altında
                            const SizedBox(height: 16),
                            SizedBox(
                              width: MediaQuery.of(context).size.width * 0.8,
                              child: ElevatedButton(
                                onPressed: () {
                                  runModel();
                                },
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.deepOrange,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(18),
                                  ),
                                  padding: const EdgeInsets.symmetric(vertical: 15),
                                  elevation: 3,
                                ),
                                child: const Text(
                                  "Tara",
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 18,
                                    color: Colors.white,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(height: 16),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            ),
            bottomNavigationBar: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 0),
              child: Transform.translate(
                offset: const Offset(0, -8),
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: Colors.white.withOpacity(0.35),
                      width: 2,
                    ),
                    borderRadius: BorderRadius.circular(16),
                    color: Colors.transparent,
                  ),
                  // Reduce vertical padding so border is just below text labels and above buttons
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      // Left: Gallery button with label
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          GestureDetector(
                            onTap: () => pickImageFromSource(ImageSource.gallery),
                            child: Container(
                              width: 40,
                              height: 40,
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.7),
                                borderRadius: BorderRadius.circular(8),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.07),
                                    blurRadius: 2,
                                    offset: Offset(1, 1),
                                  ),
                                ],
                              ),
                              child: const Center(
                                child: Text(
                                  "📁",
                                  style: TextStyle(fontSize: 28, color: Colors.deepPurple),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 2),
                          const Text(
                            "Galeri",
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                      // Center: Camera button with label
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // Move the camera button upward to overlap/cut through the border frame
                          Transform.translate(
                            offset: const Offset(0, -8),
                            child: GestureDetector(
                              onTap: () => pickImageFromSource(ImageSource.camera),
                              child: Container(
                                width: 64,
                                height: 64,
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.7),
                                  shape: BoxShape.circle,
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.12),
                                      blurRadius: 6,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: const Center(
                                  child: SizedBox(
                                    width: 36,
                                    height: 36,
                                    child: DecoratedBox(
                                      decoration: BoxDecoration(
                                        color: Colors.white,
                                        shape: BoxShape.circle,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 2),
                          const Text(
                            "Kamera",
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                      // Right: Menu button with label
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          GestureDetector(
                            onTap: () {},
                            child: Container(
                              width: 40,
                              height: 40,
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.7),
                                borderRadius: BorderRadius.circular(8),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.07),
                                    blurRadius: 2,
                                    offset: Offset(1, 1),
                                  ),
                                ],
                              ),
                              child: const Center(
                                child: Text(
                                  "☰",
                                  style: TextStyle(fontSize: 28, color: Colors.deepPurple),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 2),
                          const Text(
                            "Menü",
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class BoundingBoxPainter extends CustomPainter {
  final List<Detection> detections;
  final Rect imageRect;
  final int origW;
  final int origH;

  BoundingBoxPainter({
    required this.detections,
    required this.imageRect,
    required this.origW,
    required this.origH,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final frameW = size.width;
    final frameH = size.height;

    final imgRatio = origW / origH;
    final frameRatio = frameW / frameH;

    double scale, offsetX = 0, offsetY = 0;

    if (imgRatio > frameRatio) {
      // Resim geniş → yükseğe göre ölçeklenir, x'ten kırpılır
      scale = frameH / origH;
      offsetX = (origW * scale - frameW) / 2;
    } else {
      // Resim uzun → genişliğe göre ölçeklenir, y’den kırpılır
      scale = frameW / origW;
      offsetY = (origH * scale - frameH) / 2;
    }

    for (final det in detections) {
      final left   = det.box.left * scale - offsetX;
      final top    = det.box.top * scale - offsetY;
      final right  = det.box.right * scale - offsetX;
      final bottom = det.box.bottom * scale - offsetY;

      final rect = Rect.fromLTRB(left, top, right, bottom);
      if (rect.left < 0 || rect.top < 0 || rect.right > frameW || rect.bottom > frameH) {
        final clipped = Rect.fromLTRB(
          rect.left.clamp(0.0, frameW),
          rect.top.clamp(0.0, frameH),
          rect.right.clamp(0.0, frameW),
          rect.bottom.clamp(0.0, frameH),
        );
        canvas.drawRect(clipped, boxPaint);
      } else {
        canvas.drawRect(rect, boxPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
  // For Tanı button animation
  bool _isPressed = false;