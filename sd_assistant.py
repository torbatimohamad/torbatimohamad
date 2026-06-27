#!/usr/bin/env python3
"""
SD Assistant - کنترل کامل Stable Diffusion با زبان طبیعی
ساخته شده برای استفاده با Claude Code
"""

import requests
import base64
import json
import os
import sys
import time
from pathlib import Path

SD_URL = "http://192.168.1.55:7860"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────
# اتصال و اطلاعات
# ─────────────────────────────────────────

def check_connection():
    """بررسی اتصال به SD"""
    try:
        r = requests.get(f"{SD_URL}/sdapi/v1/sd-models", timeout=5)
        if r.status_code == 200:
            print("✅ اتصال به Stable Diffusion برقرار است")
            return True
    except:
        pass
    print("❌ خطا: Stable Diffusion در حال اجرا نیست یا API فعال نیست")
    return False

def get_models():
    """لیست مدل‌های نصب شده"""
    r = requests.get(f"{SD_URL}/sdapi/v1/sd-models")
    models = r.json()
    print("\n📦 مدل‌های موجود:")
    for i, m in enumerate(models):
        print(f"  {i+1}. {m['model_name']}")
    return models

def get_current_model():
    """مدل فعلی"""
    r = requests.get(f"{SD_URL}/sdapi/v1/options")
    opts = r.json()
    current = opts.get("sd_model_checkpoint", "نامشخص")
    print(f"\n🎯 مدل فعلی: {current}")
    return current

def switch_model(model_name: str):
    """تعویض مدل"""
    print(f"\n🔄 در حال تعویض به مدل: {model_name}")
    payload = {"sd_model_checkpoint": model_name}
    r = requests.post(f"{SD_URL}/sdapi/v1/options", json=payload)
    if r.status_code == 200:
        print("✅ مدل با موفقیت تعویض شد")
    else:
        print(f"❌ خطا در تعویض مدل: {r.text}")

def get_samplers():
    """لیست sampler ها"""
    r = requests.get(f"{SD_URL}/sdapi/v1/samplers")
    samplers = [s["name"] for s in r.json()]
    print("\n🎲 Sampler های موجود:")
    for s in samplers:
        print(f"  - {s}")
    return samplers

def get_loras():
    """لیست LoRA های نصب شده"""
    r = requests.get(f"{SD_URL}/sdapi/v1/loras")
    loras = r.json()
    if loras:
        print("\n🎨 LoRA های موجود:")
        for l in loras:
            print(f"  - {l['name']}")
    else:
        print("\n⚠️ هیچ LoRA ای نصب نیست")
    return loras

def get_embeddings():
    """لیست Embedding های نصب شده"""
    r = requests.get(f"{SD_URL}/sdapi/v1/embeddings")
    data = r.json()
    loaded = data.get("loaded", {})
    if loaded:
        print("\n🔤 Embedding های موجود:")
        for name in loaded.keys():
            print(f"  - {name}")
    else:
        print("\n⚠️ هیچ Embedding ای نصب نیست")
    return loaded

# ─────────────────────────────────────────
# تولید تصویر
# ─────────────────────────────────────────

def generate_image(
    prompt: str,
    negative_prompt: str = "ugly, blurry, bad anatomy, watermark, text, low quality",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 7.0,
    sampler: str = "DPM++ 2M Karras",
    seed: int = -1,
    batch_size: int = 1,
    filename: str = None
):
    """
    تولید تصویر با پارامترهای کامل
    
    مثال:
    generate_image(
        prompt="a beautiful sunset over mountains, photorealistic, 8k",
        negative_prompt="ugly, blurry",
        width=768,
        height=512,
        steps=30,
        cfg_scale=7.5,
        sampler="DPM++ 2M Karras"
    )
    """
    
    print(f"\n🎨 در حال تولید تصویر...")
    print(f"   پرامپت: {prompt[:80]}...")
    print(f"   سایز: {width}x{height} | استپ: {steps} | CFG: {cfg_scale}")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler,
        "seed": seed,
        "batch_size": batch_size,
        "save_images": True
    }
    
    start = time.time()
    r = requests.post(f"{SD_URL}/sdapi/v1/txt2img", json=payload, timeout=300)
    elapsed = time.time() - start
    
    if r.status_code != 200:
        print(f"❌ خطا: {r.text}")
        return None
    
    data = r.json()
    saved_paths = []
    
    for i, img_b64 in enumerate(data["images"]):
        if filename:
            name = f"{filename}_{i+1}.png" if batch_size > 1 else f"{filename}.png"
        else:
            timestamp = int(time.time())
            name = f"output_{timestamp}_{i+1}.png"
        
        path = OUTPUT_DIR / name
        with open(path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        saved_paths.append(str(path))
        print(f"✅ ذخیره شد: {path}")
    
    # اطلاعات seed
    if "info" in data:
        info = json.loads(data["info"])
        used_seed = info.get("seed", "نامشخص")
        print(f"🌱 Seed استفاده شده: {used_seed}")
    
    print(f"⏱️ زمان: {elapsed:.1f} ثانیه")
    return saved_paths

def img2img(
    image_path: str,
    prompt: str,
    negative_prompt: str = "ugly, blurry, bad anatomy, watermark",
    denoising_strength: float = 0.7,
    steps: int = 20,
    cfg_scale: float = 7.0,
    sampler: str = "DPM++ 2M Karras",
    width: int = None,
    height: int = None,
    filename: str = None
):
    """
    تبدیل تصویر به تصویر (img2img)
    
    denoising_strength: 0.0 = تغییر ندیدن اصل | 1.0 = تغییر کامل
    """
    
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "init_images": [img_b64],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler,
    }
    
    if width: payload["width"] = width
    if height: payload["height"] = height
    
    print(f"\n🖼️ در حال پردازش img2img...")
    r = requests.post(f"{SD_URL}/sdapi/v1/img2img", json=payload, timeout=300)
    
    if r.status_code != 200:
        print(f"❌ خطا: {r.text}")
        return None
    
    data = r.json()
    timestamp = int(time.time())
    name = f"{filename}.png" if filename else f"img2img_{timestamp}.png"
    path = OUTPUT_DIR / name
    
    with open(path, "wb") as f:
        f.write(base64.b64decode(data["images"][0]))
    
    print(f"✅ ذخیره شد: {path}")
    return str(path)

# ─────────────────────────────────────────
# دانلود از CivitAI
# ─────────────────────────────────────────

def get_sd_paths():
    """پیدا کردن مسیر SD"""
    # مسیرهای رایج AUTOMATIC1111
    common_paths = [
        r"C:\stable-diffusion-webui",
        r"C:\SD\stable-diffusion-webui",
        r"D:\stable-diffusion-webui",
        r"C:\Users\Public\stable-diffusion-webui",
    ]
    
    # خواندن از API
    try:
        r = requests.get(f"{SD_URL}/sdapi/v1/cmd-flags")
        if r.status_code == 200:
            flags = r.json()
            # مسیر از config
    except:
        pass
    
    for p in common_paths:
        if os.path.exists(p):
            return p
    
    return None

def download_from_civitai(
    model_id: str,
    model_type: str = "model",
    civitai_token: str = None,
    sd_path: str = None
):
    """
    دانلود مدل، LoRA، Embedding از CivitAI
    
    model_type: "model" | "lora" | "embedding" | "vae"
    model_id: ID از URL سایت CivitAI (عدد آخر URL)
    
    مثال:
    download_from_civitai("12345", "lora")
    download_from_civitai("98765", "model")
    """
    
    # پیدا کردن مسیر SD
    if not sd_path:
        sd_path = get_sd_paths()
    
    if not sd_path:
        print("❌ مسیر Stable Diffusion پیدا نشد.")
        print("   لطفاً مسیر رو وارد کن:")
        sd_path = input("SD Path: ").strip()
    
    # تعیین پوشه مقصد
    type_dirs = {
        "model": r"models\Stable-diffusion",
        "lora": r"models\Lora",
        "embedding": r"embeddings",
        "vae": r"models\VAE",
        "controlnet": r"models\ControlNet",
        "upscaler": r"models\ESRGAN",
    }
    
    dest_dir = os.path.join(sd_path, type_dirs.get(model_type, r"models\Stable-diffusion"))
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"\n📥 دانلود از CivitAI...")
    print(f"   Model ID: {model_id}")
    print(f"   نوع: {model_type}")
    print(f"   مقصد: {dest_dir}")
    
    # گرفتن اطلاعات مدل
    headers = {}
    if civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"
    
    info_url = f"https://civitai.com/api/v1/models/{model_id}"
    r = requests.get(info_url, headers=headers)
    
    if r.status_code != 200:
        print(f"❌ خطا در دریافت اطلاعات مدل: {r.status_code}")
        return None
    
    model_info = r.json()
    model_name = model_info.get("name", f"model_{model_id}")
    print(f"   نام: {model_name}")
    
    # پیدا کردن آخرین نسخه
    versions = model_info.get("modelVersions", [])
    if not versions:
        print("❌ هیچ نسخه‌ای پیدا نشد")
        return None
    
    latest = versions[0]
    files = latest.get("files", [])
    
    if not files:
        print("❌ هیچ فایلی پیدا نشد")
        return None
    
    # انتخاب فایل اصلی
    target_file = None
    for f in files:
        if f.get("primary", False) or f["type"] == "Model":
            target_file = f
            break
    if not target_file:
        target_file = files[0]
    
    download_url = target_file["downloadUrl"]
    file_name = target_file["name"]
    file_size = target_file.get("sizeKB", 0) * 1024
    
    print(f"   فایل: {file_name}")
    print(f"   حجم: {file_size/1024/1024:.1f} MB")
    
    if civitai_token:
        download_url += f"?token={civitai_token}"
    
    # دانلود با progress bar
    dest_path = os.path.join(dest_dir, file_name)
    
    if os.path.exists(dest_path):
        print(f"⚠️ فایل قبلاً موجود است: {dest_path}")
        return dest_path
    
    print(f"\n⬇️ شروع دانلود...")
    
    r = requests.get(download_url, headers=headers, stream=True, timeout=60)
    
    if r.status_code != 200:
        print(f"❌ خطا در دانلود: {r.status_code}")
        return None
    
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if file_size > 0:
                    pct = downloaded / file_size * 100
                    print(f"\r   {pct:.1f}% ({downloaded/1024/1024:.1f} MB)", end="", flush=True)
    
    print(f"\n✅ دانلود کامل شد: {dest_path}")
    print(f"⚠️ برای استفاده از مدل جدید، SD رو refresh کن یا ری‌استارت کن")
    
    return dest_path

def refresh_models():
    """رفرش لیست مدل‌ها بدون ری‌استارت"""
    print("\n🔄 در حال رفرش مدل‌ها...")
    r = requests.post(f"{SD_URL}/sdapi/v1/refresh-checkpoints")
    r2 = requests.post(f"{SD_URL}/sdapi/v1/refresh-loras")
    print("✅ لیست مدل‌ها به‌روز شد")

# ─────────────────────────────────────────
# تنظیمات پیشرفته
# ─────────────────────────────────────────

def set_hires_fix(
    prompt: str,
    negative_prompt: str = "ugly, blurry, bad anatomy",
    width: int = 512,
    height: int = 512,
    upscale_by: float = 2.0,
    steps: int = 20,
    hires_steps: int = 15,
    denoising: float = 0.5,
    upscaler: str = "R-ESRGAN 4x+",
    filename: str = None
):
    """تولید تصویر با Hires Fix (کیفیت بالاتر)"""
    
    print(f"\n🔍 تولید با Hires Fix ({width}x{height} → {int(width*upscale_by)}x{int(height*upscale_by)})")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "enable_hr": True,
        "hr_scale": upscale_by,
        "hr_upscaler": upscaler,
        "hr_second_pass_steps": hires_steps,
        "denoising_strength": denoising,
    }
    
    r = requests.post(f"{SD_URL}/sdapi/v1/txt2img", json=payload, timeout=600)
    
    if r.status_code != 200:
        print(f"❌ خطا: {r.text}")
        return None
    
    data = r.json()
    timestamp = int(time.time())
    name = f"{filename}.png" if filename else f"hires_{timestamp}.png"
    path = OUTPUT_DIR / name
    
    with open(path, "wb") as f:
        f.write(base64.b64decode(data["images"][0]))
    
    print(f"✅ ذخیره شد: {path}")
    return str(path)

def interrogate(image_path: str, model: str = "clip"):
    """
    تحلیل تصویر و استخراج پرامپت
    model: "clip" یا "deepdanbooru"
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    payload = {"image": img_b64, "model": model}
    r = requests.post(f"{SD_URL}/sdapi/v1/interrogate", json=payload, timeout=60)
    
    if r.status_code == 200:
        caption = r.json().get("caption", "")
        print(f"\n🔍 پرامپت پیشنهادی:\n{caption}")
        return caption
    else:
        print(f"❌ خطا: {r.text}")
        return None

# ─────────────────────────────────────────
# راهنما
# ─────────────────────────────────────────

def help():
    print("""
╔════════════════════════════════════════════╗
║         SD Assistant - راهنمای استفاده        ║
╚════════════════════════════════════════════╝

📋 توابع اصلی:

  check_connection()          بررسی اتصال
  get_models()                لیست مدل‌ها
  get_current_model()         مدل فعلی
  switch_model("نام مدل")     تعویض مدل
  get_loras()                 لیست LoRA ها
  get_embeddings()            لیست Embedding ها
  refresh_models()            رفرش بدون ری‌استارت

🎨 تولید تصویر:

  generate_image(
      prompt="...",
      negative_prompt="...",
      width=512, height=512,
      steps=20,
      cfg_scale=7.0,
      sampler="DPM++ 2M Karras",
      seed=-1,
      batch_size=1
  )

  img2img(image_path="...", prompt="...")
  
  set_hires_fix(prompt="...", upscale_by=2.0)
  
  interrogate("path/to/image.png")

📥 دانلود از CivitAI:

  download_from_civitai(
      model_id="12345",        # عدد از URL سایت
      model_type="lora",       # model/lora/embedding/vae
      civitai_token="xxx"      # اختیاری - برای مدل‌های محدود
  )

💡 نکات:
  - تصاویر در پوشه outputs/ ذخیره می‌شن
  - seed=-1 یعنی تصادفی
  - از LoRA اینطوری استفاده کن: <lora:نام:0.8>
""")

# ─────────────────────────────────────────
# اجرای اولیه
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 SD Assistant آماده است")
    print("   برای راهنما: help()")
    check_connection()
