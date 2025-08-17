# CLIP Embedding API

🚀 واجهة برمجة سريعة لاستخراج التضمينات (embeddings) للصور والنصوص باستخدام نموذج CLIP.

## الميزات
- ⚡ تحميل نموذج مسبقًا لتقليل وقت البدء
- 🖼️ يدعم `image_url`
- 📝 يدعم `text`
- 📦 جاهز للنشر على RunPod

## طريقة الاستخدام

أرسل طلب POST كالتالي:

```json
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "text": "وصف الصورة"
  }
}
