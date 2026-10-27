// Every user-facing text of the app lives here (Turkish). An English version later = one more class.
class S {
  static const appName = 'Zoria Blur';

  // menu bar
  static const menuFile = 'Dosya';
  static const menuView = 'Görünüm';
  static const menuHelp = 'Yardım';
  static const openVideoMenu = 'Video aç…';
  static const recentMenu = 'Son projeler';
  static const recentEmpty = 'Henüz proje yok';
  static const exportMenu = 'Bulanıklaştır ve kaydet…';
  static const quit = 'Çıkış';
  static const sidebar = 'Kenar çubuğu';
  static const theme = 'Tema';
  static const themeSystem = 'Windows ayarı';
  static const themeLight = 'Açık';
  static const themeDark = 'Koyu';
  static const shortcuts = 'Klavye kısayolları';
  static const rerunCheck = 'Sistem kontrolünü tekrar çalıştır';
  static const about = 'Zoria Blur hakkında';
  static String version(String v) => 'Sürüm $v';
  static const aboutText = 'Videodaki yüzleri bulur ve seçtiğiniz kişileri bulanıklaştırır. '
      'Her şey bu bilgisayarda yapılır; video internete gönderilmez.';
  static const aboutMaker = 'ZoriaSoft';
  static const scPlay = 'Oynat / duraklat';
  static const scStep = 'Bir kare geri / ileri';
  static const scEnds = 'Başa / sona git';
  static const scOpen = 'Video aç';
  static const scExport = 'Bulanıklaştır ve kaydet';
  static const scSidebar = 'Kenar çubuğunu aç / kapat';

  // sidebar + status bar
  static const recent = 'Son projeler';
  static const recentNone = 'Açtığınız videolar burada görünür.';
  static const removeRecent = 'Listeden kaldır';
  static const collapse = 'Daralt';
  static const expand = 'Genişlet';
  static const statusReady = 'Hazır';
  static const statusPickPeople = 'Kişileri seçin, sonra kaydedin';
  static const privacy = 'Video bu bilgisayardan dışarı çıkmaz';
  static String persons(int n) => '$n kişi';
  static const today = 'Bugün';
  static const yesterday = 'Dün';
  static const months = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara'];
  static String when(DateTime t, DateTime now) {
    final hm = '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';
    final d0 = DateTime(now.year, now.month, now.day), d = DateTime(t.year, t.month, t.day);
    final days = d0.difference(d).inDays;
    if (days == 0) return '$today $hm';
    if (days == 1) return '$yesterday $hm';
    return '${t.day} ${months[t.month - 1]}${t.year == now.year ? '' : ' ${t.year}'}';
  }

  // steps (top bar)
  static const step1 = 'Video';
  static const step2 = 'Kişiler';
  static const step3 = 'Kaydet';

  // start
  static const dropTitle = 'Videoyu buraya sürükleyin';
  static const dropOr = 'ya da';
  static const pickVideo = 'Video seç';
  static const dropFoot = 'MP4, MOV, MKV, AVI, WEBM ve diğer videolar · Video bilgisayarınızdan dışarı çıkmaz';
  static const dropRelease = 'Bırakın';
  static const notAVideo = 'Bu dosya video değil. Lütfen bir video seçin (MP4, MOV, MKV…).';
  static const videoFiles = 'Video dosyaları';
  static const otherVideo = 'Başka video';
  static const analyze = 'Başlat';
  static const analyzeHint = 'Videodaki yüzler bulunur ve kişilere ayrılır. Sonra kimin bulanık olacağını seçersiniz.';
  static const analyzeWait = 'Bilgisayar hazırlanıyor, birkaç saniye sürer…';
  static const advanced = 'Gelişmiş';
  static const detMode = 'Yüz arama';
  static const detFast = 'Hızlı';
  static const detBalanced = 'Dengeli';
  static const detThorough = 'Detaylı';
  static const detFastHelp = 'Yakın çekimler için. En hızlısı.';
  static const detBalancedHelp = 'Uzaktaki küçük yüzleri gerekince kendisi arar. Önerilen.';
  static const detThoroughHelp = 'Kalabalık ve uzak çekimler için. En yavaşı.';
  static const processor = 'Donanım';
  static const procAuto = 'Otomatik';
  static const procCpu = 'İşlemci';
  static const procGpu = 'Ekran kartı';
  static const deviceChecking = 'Bilgisayar kontrol ediliyor…';
  static String deviceGpu(String name, String api) => 'Ekran kartı: $name ($api)';
  static const deviceGpuUnknown = 'Ekran kartı';
  static const deviceCpu = 'İşlemci ile çalışıyor';
  static const gpuNotUsable = 'Ekran kartı kullanılamıyor, işlemci ile çalışılacak.';
  static String duration(String d) => 'Süre $d';

  // analyze
  static const analyzing = 'Yüzler bulunuyor';
  static const phaseStart = 'Hazırlanıyor…';
  static const phaseFaces = 'Yüzler aranıyor…';
  static const phaseGrouping = 'Kişiler gruplanıyor…';
  static const phaseRender = 'Yüzler bulanıklaştırılıyor…';
  static const phaseQa = 'Son kontrol yapılıyor…';
  static String eta(String t) => 'Yaklaşık $t kaldı';
  static const etaCalc = 'Süre hesaplanıyor…';
  static const cancel = 'İptal';
  static const cancelled = 'İşlem iptal edildi.';
  static const lessThanMinute = '1 dakikadan az';
  static String minutes(int m) => '$m dakika';

  // persons
  static const whatToDo = 'Ne yapılsın?';
  static const modeSelected = 'Seçilenleri bulanıklaştır';
  static const modeExcept = 'Seçilenler hariç herkesi bulanıklaştır';
  static const strength = 'Bulanıklık';
  static const strengthLow = 'Hafif';
  static const strengthHigh = 'Güçlü';
  static const people = 'Kişiler';
  static String peopleCount(int n) => '$n kişi';
  static String selectedCount(int n) => '$n seçili';
  static const selectHint = 'Bulanık olacak kişilere tıklayın.';
  static const selectHintExcept = 'Net kalacak kişilere tıklayın. Diğer herkes bulanık olur.';
  static const merge = 'Birleştir';
  static const mergeHelp = 'Seçili kişiler aynı kişi ise birleştirin.';
  static const split = 'Ayır';
  static const splitTip = 'Bu kişinin içinde başka biri de varsa ayırın.';
  static const rename = 'Adını değiştir';
  static const renameTitle = 'Kişinin adı';
  static const save = 'Kaydet';
  static const close = 'Kapat';
  static String personDefault(int id) => 'Kişi $id';
  static String seconds(String s) => '$s sn görünüyor';
  static const lowConfidence = 'Az görünüyor';
  static const willBlur = 'Bulanık';
  static const selectAll = 'Tümünü seç';
  static const selectNone = 'Seçimi kaldır';
  static const splitTitle = 'Kişiyi ayır';
  static const splitHelp =
      'Bu kişinin içinde başka biri varsa, o parçaları seçip ayrı kişi yapın.';
  static const splitDo = 'Ayrı kişi yap';
  static const splitNothing = 'Bu kişi tek parça, ayrılacak bir şey yok.';
  static String range(String a, String b) => '$a – $b';

  // preview
  static const preview = 'Önizleme';
  static const previewExact = 'Kayıttaki hali';
  static const previewExactTip = 'Bu kare, kaydedilecek videoda nasıl görünecekse öyle gösteriliyor.';
  static const previewLiveTip = 'Oynatırken bulanıklık yaklaşık çizilir; durdurunca kaydedilecek hali gelir.';
  static const previewLive = 'Canlı önizleme';
  static const timeline = 'Zaman çizelgesi';
  static const prevFrame = 'Önceki kare';
  static const nextFrame = 'Sonraki kare';
  static const firstFrame = 'Başa git';
  static const lastFrame = 'Sona git';
  static const visible = 'Net';
  static const previewEmpty = 'Önizleme için bir an seçin';
  static const riskyMoments = 'Kontrol edilecek anlar';
  static const noRiskyMoments = 'Riskli an yok';
  static const drawBoxHint = 'Yüzün etrafına fareyle bir kutu çizin';
  static const frameLabel = 'Kare';
  static const markFace = 'Bulanık olmayan yüzü işaretle';
  static const markFaceStop = 'İşaretlemeyi bitir';
  static const whoseFace = 'Bu yüz kimin?';
  static const newPerson = 'Yeni kişi';
  static const boxAdded = 'Kutu eklendi; yüz birkaç kare boyunca takip edildi.';
  static const play = 'Oynat (bulanık önizleme)';
  static const pause = 'Duraklat';
  static const livePending = 'Önizleme güncelleniyor, şimdilik herkes bulanık…';
  static const liveLoading = 'Video açılıyor…';
  static const before = 'Önce';
  static const after = 'Sonra';

  // risk types (engine `type`) in plain words
  static const risk = <String, String>{
    'track_break': 'Yüz bir süre görünmedi',
    'occlusion': 'Biri başkasının arkasında kaldı',
    'face_merge': 'Yüzler üst üste geldi',
    'heads_overlap': 'Başlar üst üste geldi',
    'ambiguous': 'Yüzün kime ait olduğu belirsiz',
    'low_confidence': 'Kişi tanıması zayıf',
    'unknown_face': 'Tanınmayan net yüz',
    'unconfirmed_face': 'Yüze benzeyen bir şey açık kaldı',
    'leak': 'Yüz açık kalmış olabilir',
  };
  static const riskOther = 'Riskli an';

  // export
  static const export = 'Bulanıklaştır ve kaydet';
  static const exporting = 'Video kaydediliyor';
  static const needSelection = 'Önce soldan bulanık olacak kişileri seçin.';
  static const exceptNone = 'Kimse seçilmedi: herkes bulanıklaşacak.';
  static String step(int a, int b) => 'Adım $a/$b';
  static const done = 'Video hazır';
  static const openFolder = 'Klasörde göster';
  static const backToEdit = 'Düzenlemeye dön';
  static const doneTitle = 'Sonuç';
  static const resultTab = 'Bulanık hali';
  static const originalTab = 'Orijinal';
  static const originalBadge = 'Orijinal video';
  static const newVideo = 'Yeni video';
  static const openExternal = 'Başka oynatıcıda aç';
  static const resultCantPlay = 'Bu video burada oynatılamadı. Bilgisayarınızdaki oynatıcıyla açabilirsiniz.';
  static const finalCheck = 'Son kontrol';
  static const leakWatchHint = 'Bu anları buradan izleyin; gerekirse düzenlemeye dönüp düzeltin.';
  static const fixInEditor = 'Düzenlemeye dön ve düzelt';
  static const playTip = 'Oynat';
  static const pauseTip = 'Duraklat';
  static const mute = 'Sesi kapat';
  static const unmute = 'Sesi aç';
  static const qaClean = 'Kontrol temiz: açık kalan yüz bulunamadı.';
  static String qaRiskOnly(int n) =>
      'Açık yüz bulunamadı. Yine de $n riskli an var, bakmanız iyi olur.';
  static String qaLeaks(int n) =>
      'Riskli $n an var, kontrol edin: bu anlarda bir yüz net kalmış olabilir.';
  static const qaSkipped = 'Son kontrol yapılamadı.';
  static const saveTitle = 'Bulanık videoyu kaydet';
  static const outSuffix = '_bulanik';


  // first-run setup (welcome -> system check -> ready)
  static const setupTitle = 'Kurulum';
  static const setupSkip = 'Şimdilik geç';
  static const setupSkipTip = 'Kontrol arka planda sürer; sonuç hazır olunca kullanılır.';
  static const wStep1 = 'Hoş geldiniz';
  static const wStep2 = 'Sistem';
  static const wStep3 = 'Hazır';
  static const back = 'Geri';
  static const next = 'İleri';
  static const letsStart = 'Başlayalım';
  static const finish = 'Bitir';
  static const welcomeTitle = 'Zoria Blur’a hoş geldiniz';
  static const welcomeLead = 'Videolardaki yüzleri bulur, seçtiğiniz kişileri bulanıklaştırır.';
  static const welcomePrivacy = 'Videolarınız bu bilgisayardan dışarı çıkmaz';
  static const welcomePrivacySub = 'Tüm işlem bu bilgisayarda yapılır. Hiçbir video internete gönderilmez.';
  static const welcomeCheck = 'Şimdi sistem bilgilerini öğrenelim';
  static const welcomeCheckSub =
      'Bilgisayarınızın donanımına bakıp en hızlı çalışma yolunu seçeceğiz. Genellikle yarım dakika sürer.';
  static const welcomeLater = 'Seçim sonradan değiştirilebilir';
  static const welcomeLaterSub = 'Gelişmiş bölümünden istediğiniz zaman işlemci ya da ekran kartı seçebilirsiniz.';
  static const checkTitle = 'Sistem kontrolü';
  static const checkLead = 'Donanımınızı kontrol ediyoruz. Öncelik sırası: NVIDIA CUDA, sonra DirectML, sonra işlemci.';
  static const checkRunning = 'Kontrol sürüyor…';
  static String checkProgress(int a, int b) => '$a / $b tamamlandı';
  // rows
  static const rowCpu = 'İşlemci';
  static const rowMemory = 'Bellek';
  static const rowGpus = 'Ekran kartı';
  static const rowCuda = 'NVIDIA CUDA desteği';
  static const rowDml = 'DirectML desteği (AMD, Intel ve diğer kartlar)';
  static const rowTest = 'Deneme çalıştırması';
  // row states
  static const stWaiting = 'Bekliyor';
  static const stChecking = 'Kontrol ediliyor';
  static const stFound = 'Bulundu';
  static const stMissing = 'Yok';
  static const stFailed = 'Çalışmadı';
  static const stWorks = 'Çalışıyor';
  static const stPassed = 'Başarılı';
  static const stUnknown = 'Bilinmiyor';
  static const stSkipped = 'Atlandı';
  // row details
  static String cpuCores(int cores, int threads) =>
      cores > 0 && threads > cores ? '$cores çekirdek, $threads iş parçacığı' : '${cores > 0 ? cores : threads} çekirdek';
  static const cpuNoName = 'Model adı okunamadı';
  static String memoryGb(String gb) => '$gb GB';
  static const memoryUnknown = 'Bellek miktarı okunamadı';
  static const gpusNone = 'Ekran kartı bulunamadı';
  static const cudaNoNvidia = 'NVIDIA ekran kartı yok, bu adım atlandı';
  static const cudaNotShipped = 'NVIDIA kartı var, ancak bu kurulumda NVIDIA desteği yok';
  static const cudaDisabled = 'Kapatılmış';
  static const cudaFailed = 'NVIDIA kartı var, ancak denemede çalışmadı';
  static String cudaOldDriver(String have, String need) => 'Sürücü eski ($have). En az $need gerekiyor';
  static String adviceOldNvidia(String have) =>
      'NVIDIA kartınızın sürücüsü eski ($have). Güncellerseniz program kartınızı tam hızda kullanır; '
      'güncelledikten sonra Yardım menüsünden kontrolü tekrar çalıştırın.';
  static const adviceCudaFailed =
      'NVIDIA kartınız denemede çalışmadı. Çoğu zaman sürücüyü güncellemek bunu çözer; '
      'güncelledikten sonra kontrolü tekrar çalıştırın.';
  static const adviceNvidiaPack =
      'NVIDIA kartınız var ama bu kurulumda NVIDIA hız paketi yok. Tam kurulumu indirirseniz kartınız tam hızda kullanılır.';
  static const adviceDmlFailed =
      'Ekran kartınız denemede çalışmadı, işlem işlemciyle yapılacak. Sürücüyü güncellemek çoğu zaman bunu çözer.';
  static const adviceDriverButton = 'Sürücü sayfasını aç';
  static const adviceNvidiaPackButton = 'Tam kurulumu indir';
  static const adviceRecheck = 'Tekrar kontrol et';
  static const adviceTitle = 'Daha hızlı çalışması için';
  static const dmlNotNeeded = 'Gerek kalmadı: NVIDIA CUDA çalışıyor';
  static const dmlNotInstalled = 'Bu kurulumda DirectML yok';
  static const dmlSoftwareOnly = 'Yalnızca Windows’un yazılım ekran sürücüsü var';
  static const dmlFailed = 'Ekran kartı denemede çalışmadı';
  static const dmlNoGpu = 'Kullanılabilir ekran kartı yok';
  static const testOnGpu = 'Ekran kartında kısa bir yüz arama denemesi yapılıyor…';
  static const testOnCpu = 'İşlemcide kısa bir yüz arama denemesi yapılıyor…';
  static String testOk(String where, String sec) => '$where bir karede yüz arama $sec sn sürdü';
  static const whereGpu = 'Ekran kartında';
  static const whereCpu = 'İşlemcide';
  static String testCompare(String sec) => '(işlemcide $sec sn)';
  static const testFellBack = 'Ekran kartında çalışmadı, bir sonraki seçeneğe geçildi.';
  static const testCpuOnly = 'İşlemci ile çalışılacak';
  static const noAnswer = 'Yanıt alınamadı, atlandı';
  // result
  static const resultTitle = 'Bu bilgisayarda en hızlı seçenek';
  static String resultGpu(String name, String api) => '$name · $api';
  static String resultCpu(String name) => 'İşlemci · $name';
  static const resultCpuPlain = 'İşlemci';
  static const whyCuda = 'NVIDIA ekran kartınız CUDA ile denendi ve çalıştı. En hızlı yol bu.';
  static const whyDml = 'NVIDIA kartı yok; ekran kartınız DirectML ile denendi ve çalıştı.';
  static const whyDmlAfterCuda = 'NVIDIA CUDA çalışmadı; ekran kartınız DirectML ile çalışıyor.';
  static const whyCpu =
      'Kullanılabilir bir ekran kartı bulunamadı. İşlem işlemci ile yapılacak: biraz daha yavaştır, sonuç aynıdır.';
  static const whyCpuFailed =
      'Ekran kartı kontrol edilemedi. Güvenli seçenek olarak işlemci kullanılacak; sonuç aynıdır.';
  static String speedHint(String x) =>
      'Denemede yüz arama işlemciye göre yaklaşık $x kat hızlı çalıştı (tüm işlemde fark daha küçüktür).';
  static const changeLater = 'Bunu daha sonra Gelişmiş → Donanım bölümünden değiştirebilirsiniz.';
  static const outFolderTitle = 'Bulanık videolar nereye kaydedilsin?';
  static const outBesideVideo = 'Videonun bulunduğu klasöre (önerilen)';
  static const outFixed = 'Her zaman şu klasöre:';
  static const outPick = 'Klasör seç…';
  static const outNone = 'Henüz klasör seçilmedi';
  static const outAskEachTime = 'Kaydederken yine de başka bir yer seçebilirsiniz.';
  static const deviceChanged = 'Donanım değişmiş; en hızlı seçenek yeniden belirlendi.';

  // errors
  static const errEngineMissing = 'İşlem motoru bulunamadı. Uygulamayı yeniden kurun.';
  static const errFfmpeg = 'Video aracı (ffmpeg) bulunamadı. Uygulamayı yeniden kurun.';
  static const errModels = 'Yüz tanıma modelleri eksik. Uygulamayı yeniden kurun.';
  static const errOpenVideo = 'Bu video açılamadı. Farklı bir video deneyin.';
  static const videoChanged = 'Bu video, kayıtlı projedekinden farklı. Yüz araması yeniden yapılacak.';
  static const errNoFile = 'Dosya bulunamadı. Taşınmış ya da silinmiş olabilir.';
  static const errWrite = 'Dosya yazılamadı. Başka bir klasör seçin veya diskte yer açın.';
  static const errMemory = 'Bellek yetmedi. Diğer programları kapatıp tekrar deneyin.';
  static const errEncode = 'Video yazılamadı. Diskte yer olduğundan emin olun.';
  static const errFrame = 'Bu an videonun dışında.';
  static const errNeedPersons = 'Önce bulanıklaştırılacak kişileri seçin.';
  static const errGeneric = 'Beklenmeyen bir sorun oluştu.';
  static const errDetails = 'Ayrıntı';
  static const ok = 'Tamam';
}
