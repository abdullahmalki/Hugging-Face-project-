# Hugging-Face-project-
from transformers import pipeline
import gradio as gr

# تحميل النموذج الجاهز من Hugging Face
classifier = pipeline("sentiment-analysis")

# دالة تستخدم النموذج لتحليل الشعور
def analyze_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return f"المشاعر: {label} (الدقة: {score:.2f})"

# واجهة Gradio
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="أدخل جملة بالإنجليزية..."),
    outputs="text",
    title="تحليل المشاعر",
    description="أدخل جملة وسيقوم النموذج بتحليل مشاعرك (إيجابي أو سلبي)"
)

interface.launch()
