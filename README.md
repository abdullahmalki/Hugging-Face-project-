# Hugging-Face-project-
# مشروع: مترجم إنجليزي-عربي باستخدام Hugging Face وGradio

## 💡 فكرة المشروع
تطبيق بسيط لترجمة الجمل من اللغة الإنجليزية إلى اللغة العربية باستخدام نموذج مدرّب مسبقاً من Hugging Face، مع واجهة تفاعلية مبنية بـ Gradio.

## 🧠 النموذج المستخدم
- الاسم: Helsinki-NLP/opus-mt-en-ar
- المصدر: Hugging Face
- المهمة: ترجمة آلية (Machine Translation)

## 🚀 تشغيل التطبيق
1. تثبيت الحزم:

pip install transformers gradio)



## 🖼️ شرح واجهة Gradio

في مشروعنا، استخدمنا مكتبة **Gradio** لإنشاء واجهة تفاعلية تسهّل على المستخدم تجربة النموذج بدون الحاجة إلى معرفة برمجية.

### ✅ مكوّن الواجهة المستخدم:
```python
interface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(label="أدخل النص باللغة الإنجليزية"),
    outputs=gr.Textbox(label="النص المترجم إلى العربية"),
    title="مترجم إنجليزي-عربي",
    description="أدخل جملة باللغة الإنجليزية وسيقوم النموذج بترجمتها إلى اللغة العربية باستخدام نموذج من Hugging Face"
)
