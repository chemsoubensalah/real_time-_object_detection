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
ضع `best.pt` في مجلد المشروع ثم:
```bash
python scripts/detect_realtime.py --weights best.pt --source http://192.168.1.5:8080/video
python scripts/detect_realtime.py --weights best.pt --source video.mp4   # فيديو
python scripts/detect_realtime.py --weights best.pt --source photo.jpg   # صورة
```
- تظهر المربعات ونسبة الثقة و**FPS** على الشاشة، ويُطبع متوسط FPS عند الخروج.
- `s` يلتقط صورة شاشة (للتقرير)، و`q` للخروج.
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
