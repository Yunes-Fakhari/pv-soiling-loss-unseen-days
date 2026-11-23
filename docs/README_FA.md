# راهنمای فارسی PBI-Loss

این پوشه‌ها و فایل‌ها را در شاخهٔ اصلی ریپوی خود قرار دهید؛ یعنی `README.md`، `training`، `data` و `src` مستقیماً در ریشهٔ ریپو باشند. فایل ZIP را ابتدا روی رایانه استخراج کنید و محتوای پوشه را به ریپو منتقل کنید.

## پوشه‌های مهم

| پوشه | کاربرد |
|---|---|
| `src/pbi_loss` | کد پردازش تصویر، شاخص ناحیه‌ای و فرمول تخمین افت توان |
| `training` | آموزش یا برازش ضرایب و اجرای پروتکل‌های مقاله |
| `evaluation` | محاسبهٔ MSE، RMSE، MAE و R² و آزمایش حذف مؤلفه‌ها |
| `inference` | پیش‌بینی برای یک تصویر جدید و زمان ثبت آن |
| `configs` | مسیر دیتاست و پارامترهای روش |
| `data/raw` | محل قرار دادن دیتاست خام |
| `data/processed` | خروجی آماده‌سازی داده و استخراج ویژگی |
| `data/reference` | ویژگی‌های آرشیوی مقاله برای بازتولید سریع اعداد |
| `models` | مدل‌های آمادهٔ کم‌حجم؛ سه ضریب و تنظیمات پردازش |
| `results/paper` | اعداد مرجع مقاله و گزارش کنترل بازتولید |
| `outputs` | نتایج آموزش و ارزیابی اجراهای جدید |

آموزش در این روش به معنی برازش سه ضریب فرمول با کمترین مربعات است؛ شبکهٔ عصبی و GPU لازم نیست.

## ۱. نصب

از Python نسخهٔ ۳٫۱۰ یا جدیدتر استفاده کنید. ترمینال را در ریشهٔ ریپو باز کنید:

```bash
python -m venv .venv
```

در ویندوز:

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

در لینوکس به جای دستور فعال‌سازی بالا از `source .venv/bin/activate` استفاده کنید. نسخه‌های دقیق محیط آزموده‌شده در `requirements-tested.txt` ثبت شده‌اند.

## ۲. مسیر دیتاست

دیتاست اصلی را از https://deep-solar-eye.github.io/ دریافت کنید. تصاویر در این بسته نیستند. ساختار پیشنهادی:

```text
data/raw/PanelImages/solar_....jpg
```

اگر دیتاست جای دیگری است، مسیر پوشه‌ای را بدهید که `PanelImages` داخل آن قرار دارد:

```bash
python data/prepare_dataset.py --data-root "D:/Datasets/DeepSolarEye" --workers 4
```

راه دیگر، تغییر `dataset_root` در `configs/paper.json` است. با مسیر پیش‌فرض:

```bash
python data/prepare_dataset.py --workers 4
```

کد، زمان و برچسب را از نام اصلی فایل‌ها می‌خواند و اولین عکس هر بازهٔ ۳۰ ثانیه‌ای را انتخاب می‌کند. خروجی دیتاست کامل باید ۱۱٬۶۳۲ تصویر از ۱۵ روز باشد. فرمت تصاویر باید همان نمای کامل دوربین با ابعاد ۱۹۲×۱۹۲ باشد. تابش در آموزش PBI-Loss استفاده نمی‌شود.

## ۳. آموزش

```bash
python training/train.py --protocol random_days --seed 42 --output outputs/random_days_42
```

این دستور ۱۲ روز را برای آموزش و سه روز را برای آزمون انتخاب می‌کند. مدل، فهرست دقیق تصاویر هر بخش، پیش‌بینی‌های آزمون و چهار معیار در پوشهٔ خروجی ذخیره می‌شوند.

برای اجرای همهٔ پروتکل‌های مقاله:

```bash
python training/run_protocols.py --output outputs/paper
python evaluation/verify_paper.py --summary outputs/paper/metrics_summary.csv --atol 0.00005
```

LODO یعنی هر بار یک روز کامل از آموزش کنار گذاشته شود. در پروتکل تصادفی تصویر، تصاویر یک روز ممکن است در هر دو بخش باشند؛ بنابراین تفسیر آن با آزمون روی روزهای ندیده متفاوت است.

## ۴. بازتولید سریع بدون تصاویر

```bash
python training/run_protocols.py --features data/reference/paper_features.csv.gz --output outputs/reference
python evaluation/verify_paper.py --summary outputs/reference/metrics_summary.csv
```

این مسیر از سهم آلودگی ۷۲ سلول که قبلاً ذخیره شده استفاده می‌کند؛ ضرایب و پیش‌بینی‌ها دوباره محاسبه می‌شوند. این آزمون جای اجرای پردازش تصویر از عکس خام را نمی‌گیرد. جزئیات در `docs/REPRODUCIBILITY.md` آمده است.

## ۵. پیش‌بینی یک عکس

```bash
python inference/predict.py --model models/random_days_seed42.json --image "D:/Images/frame.jpg" --hour 10.75 --output outputs/prediction.json --visuals outputs/example
```

عدد ۱۰٫۷۵ یعنی ساعت ۱۰:۴۵ به وقت محلی ثبت تصاویر. به جای `--hour` می‌توانید `--timestamp 2017-06-22T10:45:00` بدهید. خروجی، افت نسبی توان در مقیاس صفر تا یک و مقدار درصدی آن است. برچسب افت توان داخل نام فایل در پیش‌بینی خوانده نمی‌شود.

گوشه‌های پیش‌فرض مخصوص نمای دوربین DeepSolarEye هستند؛ برای دوربین جدید باید مختصات گوشه‌ها و آرایش سلول/زیررشته بررسی شود.

## ۶. مدل قابل استفاده پس از آموزش روی تمام داده‌ها

```bash
python training/train.py --protocol deployment --output outputs/deployment
```

در این حالت تمام تصاویر برای برازش استفاده می‌شوند و امتیاز آزمون مستقل تولید نمی‌شود. برای مقاله از نتایج پروتکل‌های دارای آزمون استفاده کنید.

## ۷. آزمون کد

```bash
python -m unittest discover -s tests -v
```

قالب نام فایل و مسیرها در `docs/DATASET.md` توضیح داده شده‌اند. زیرمجموعهٔ `clean_diagnostic` با استفاده از برچسب‌ها انتخاب می‌شود و نتیجهٔ اصلی مقاله همان `all` است.
