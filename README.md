# الكشف عن الإنسان في الوقت الحقيقي بـ YOLOv5 وكاميرا الهاتف
Real-time human (upper / full body) detection with YOLOv5 using a phone camera.

مراحل المشروع: **جمع البيانات ← وضع العلامات ← التقسيم (Train / Val / Test) ← التدريب ← التحقق ← الاختبار بالكاميرا**

```
scripts/camera.py               قراءة بث كاميرا الهاتف (أو كاميرا الحاسوب)
scripts/capture_images.py       التقاط صور التدريب من الهاتف
scripts/download_coco_person.py تنزيل صور أشخاص جاهزة بالعلامات من COCO
scripts/split_dataset.py        تقسيم البيانات train/val/test وإنشاء data.yaml
scripts/detect_realtime.py      الكشف المباشر بالنموذج best.pt مع عرض FPS
notebooks/train_yolov5_colab.ipynb  التدريب والتقييم على Google Colab (GPU مجاني)
```

---

## ✅ النموذج جاهز: النتائج

دُرّب النموذج **`models/person_yolov5n.pt`** (حجمه 3.7 MB) على **2000 صورة من COCO** فيها أشخاص (فئة واحدة `person`):

| | |
|---|---|
| البيانات | 2000 صورة (Train 1600 / Val 200 / Test 200)، فيها 4051 شخصاً عليهم علامات |
| الشبكة | YOLOv5n، انطلاقاً من أوزان مدرّبة مسبقاً (Transfer Learning) |
| التدريب | 40 جولة (epoch)، حجم الصورة 416، على CPU فقط (4 أنوية): 65 دقيقة |

**نتائج التقييم:**

| المجموعة | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Validation (أفضل جولة) | 0.909 | 0.815 | **0.899** | 0.591 |
| **Test** (صور لم يرها النموذج) | 0.870 | 0.764 | **0.865** | 0.571 |

**السرعة:** **37 FPS** على CPU بحجم 416، و 26 FPS بحجم 640، أي أنه يعمل في الوقت الحقيقي حتى بدون GPU.

الملفات في مجلد `results/`:
- `results.png`: منحنيات الـ Loss و Precision و Recall و mAP خلال التدريب
- `test_confusion_matrix.png` و `test_PR_curve.png`: تقييم مجموعة الاختبار
- `test_predictions.jpg`: أمثلة كشف على صور الاختبار
- `demo_pedestrians.mp4` و `demo_frame_*.jpg`: تجربة على فيديو مشاة مع عرض FPS

**نقطة ضعف معروفة:** الأشخاص **الصغار جداً أو البعيدين جداً** (مثل كاميرا مراقبة عالية) يُكتشفون أقل، لأن الأشخاص الأصغر من 2% من مساحة الصورة استُبعدوا من التدريب. أما الأشخاص القريبون من كاميرا الهاتف (الجزء العلوي أو الجسم كاملاً) فيُكتشفون جيداً. للأشخاص البعيدين استعمل `--img-size 640`.

**التجربة مباشرة بكاميرا الهاتف:**
```bash
git clone https://github.com/ultralytics/yolov5        # مرة واحدة، داخل مجلد المشروع
python scripts/detect_realtime.py --weights models/person_yolov5n.pt --source http://PHONE_IP:8080/video
```

---

## المرحلة الثانية: عدة فئات، إخفاء جزئي، ومقارنة خوارزميات

شرح كامل بالعربية مع الكود: `docs/project_phase2_ar.pdf`

| الملف | الوصف |
|---|---|
| `models/multi_yolov5n.pt` | نموذج واحد لـ 5 فئات: person, car, cat, dog, drone |
| `models/multi_occ_yolov5n.pt` | نفسه، مضبوط 10 جولات مع صور مخفية جزئياً |

```bash
python scripts/detect_realtime.py --weights models/multi_yolov5n.pt --source 1
```

**النتائج (مجموعة الاختبار، 310 صور):** mAP@0.5 = 0.651 (person 0.629، car 0.503، cat 0.768، dog 0.601، drone 0.755).

**الإخفاء الجزئي (mAP@0.5):** الأساسي 0.651 / 0.542 / 0.311 / 0.135 عند إخفاء 0 / 20 / 40 / 60%. التدريب على صور مخفية أعطى 0.623 / 0.530 / 0.317 / 0.145. الفروق عن المجموعة الضابطة نحو 0.02 فقط وبتجربة واحدة، فهي غير حاسمة.

**مقارنة الخوارزميات:** `results/compare/compare.json` (YOLOv5، SSD، RetinaNet، FCOS، Faster R-CNN، DETR على COCO val2017).

| السكربت | الوظيفة |
|---|---|
| `download_coco_multi.py` | صور COCO لعدة فئات |
| `add_yolo_dataset.py` | إضافة مجموعة بيانات أخرى كفئة جديدة (المسيّرات) |
| `pseudo_label.py` | ملء العلامات الناقصة بنموذج جاهز |
| `make_occlusion.py` | صناعة صور مخفية جزئياً، العلامة = الجزء الظاهر |
| `run_occlusion_experiment.py` | تجربة الإخفاء كاملة |
| `compare_detectors.py` | مقارنة الخوارزميات بنفس المقياس |

---

## 0. التثبيت على الحاسوب
```bash
git clone https://github.com/chemsoubensalah/real_time-_object_detection.git
cd real_time-_object_detection
pip install -r requirements.txt
```

## 1. ربط كاميرا الهاتف بالحاسوب
1. ثبّت على الهاتف تطبيق **IP Webcam** (أندرويد) أو **DroidCam** (أندرويد / آيفون).
2. صِل الهاتف والحاسوب **بنفس شبكة Wi-Fi**.
3. في IP Webcam اضغط **Start server**، فيظهر عنوان مثل `http://192.168.1.5:8080`.
4. افتح العنوان في متصفح الحاسوب للتأكد أن الصورة تظهر.
5. العنوان الذي نستعمله في السكربتات:
   - IP Webcam: `http://192.168.1.5:8080/video`
   - DroidCam: `http://192.168.1.5:4747/video`

> استبدل `192.168.1.5` بالعنوان الظاهر على هاتفك.

### إذا كنت تستعمل iVCam
تطبيق iVCam لا يعطيك عنواناً، بل يجعل الهاتف **كاميرا ويب افتراضية** على الحاسوب، فتستعمل **رقم كاميرا** بدل العنوان:
1. ثبّت برنامج **iVCam للحاسوب** (Windows) من موقع e2esoft.com.
2. شغّل iVCam على الهاتف وعلى الحاسوب (نفس شبكة الـ Wi-Fi، أو بكابل USB)، وانتظر حتى تظهر صورة الهاتف في برنامج الحاسوب.
3. اعرف رقم الكاميرا:
   ```bash
   python scripts/list_cameras.py
   ```
   افتح الصور `camera_0.jpg` و`camera_1.jpg`... والصورة التي تظهر فيها صورة الهاتف هي رقم iVCam (عادةً `1` إذا كان في الحاسوب كاميرا مدمجة، أو `0` إذا لم تكن).
4. استعمل هذا الرقم في `--source`:
   ```bash
   python scripts/detect_realtime.py --weights models/person_yolov5n.pt --source 1
   ```

## 2. جمع البيانات (Dataset)

**أ) صور جاهزة بالعلامات من COCO** (أسرع طريقة، بدون رسم مربعات يدوياً):
```bash
python scripts/download_coco_person.py --num 1500 --out data/raw
```

**ب) صورك الخاصة من الهاتف:**
```bash
python scripts/capture_images.py --source http://192.168.1.5:8080/video --out data/raw/images --prefix upper
```
- `SPACE` يحفظ صورة، و`a` يشغّل الالتقاط التلقائي (صورة كل ثانية)، و`q` للخروج.
- صوّر أشخاصاً مختلفين بزوايا وإضاءة وخلفيات ومسافات متنوعة (الجزء العلوي والجسم كاملاً).

## 3. وضع العلامات (Labeling) لصورك
- استعمل **Roboflow** (على الإنترنت) أو **LabelImg** (`pip install labelImg` ثم `labelImg`).
- اختر صيغة **YOLO**، وارسم مربعاً حول كل شخص.
- ضع ملفات `.txt` في `data/raw/labels/` بنفس أسماء الصور.
- فئة واحدة `person`، أو فئتان `upper_body` و`full_body` (في هذه الحالة صور COCO كلها من فئة رقم 0).

## 4. التقسيم: تدريب / تحقق / اختبار
```bash
python scripts/split_dataset.py --src data/raw --dst data/dataset --names person
# أو بفئتين:
python scripts/split_dataset.py --names upper_body full_body
```
| المجموعة | النسبة | الدور |
|---|---|---|
| Train | 80% | يتعلم منها النموذج |
| Validation | 10% | مراقبة الأداء أثناء التدريب |
| Test | 10% | التقييم النهائي على صور لم يرها النموذج |

## 5. التدريب (Training)
**على Colab (مستحسن):** افتح `notebooks/train_yolov5_colab.ipynb` في Google Colab، واختر GPU، ثم شغّل الخلايا بالترتيب. في النهاية يُنزَّل ملف `best.pt`.

**أو على حاسوب فيه GPU:**
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
python train.py --img 640 --batch 16 --epochs 100 --data ../data/dataset/data.yaml --weights yolov5s.pt --name person
```
النموذج الناتج: `yolov5/runs/train/person/weights/best.pt`

## 6. التحقق والاختبار (Validation / Test)
```bash
python val.py --weights runs/train/person/weights/best.pt --data ../data/dataset/data.yaml --task test
```
المقاييس التي تكتبها في التقرير: **Precision** و**Recall** و**mAP@0.5** و**mAP@0.5:0.95**، مع منحنيات `results.png` و`confusion_matrix.png`.

## 7. الكشف المباشر بكاميرا الهاتف
استعمل النموذج الجاهز `models/person_yolov5n.pt` أو نموذجك `best.pt`. من الأفضل نسخ مجلد `yolov5` داخل المشروع حتى يُحمَّل النموذج بدون إنترنت:
```bash
python scripts/detect_realtime.py --weights best.pt --source http://192.168.1.5:8080/video
python scripts/detect_realtime.py --weights best.pt --source video.mp4   # فيديو
python scripts/detect_realtime.py --weights best.pt --source photo.jpg   # صورة
```
- تظهر المربعات ونسبة الثقة و**FPS** على الشاشة، ويُطبع متوسط FPS عند الخروج.
- `s` يلتقط صورة شاشة (للتقرير)، و`q` للخروج.
- `--save-video out.mp4` يسجّل الفيديو مع المربعات.
- لزيادة السرعة: `--img-size 416` أو درّب `yolov5n.pt`.

## 8. مقارنة (اختياري)
درّب `yolov5n` و`yolov5s` و`yolov5m` (أو YOLOv8 / YOLO11 من مكتبة `ultralytics`) على نفس البيانات، واملأ الجدول:

| النموذج | mAP@0.5 | mAP@0.5:0.95 | FPS (كاميرا الهاتف) |
|---|---|---|---|
| yolov5n | | | |
| yolov5s | | | |
| yolov5m | | | |

## مشاكل شائعة
- **Cannot open camera source**: الهاتف والحاسوب ليسا على نفس الشبكة، أو العنوان خاطئ. جرّب فتحه في المتصفح أولاً.
- **الفيديو متأخر**: خفّض الدقة في إعدادات IP Webcam (مثلاً 640x480).
- **FPS منخفض على CPU**: استعمل `yolov5n` و`--img-size 416`.
