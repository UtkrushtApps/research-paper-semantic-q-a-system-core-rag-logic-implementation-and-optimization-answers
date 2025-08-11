# rag_pipeline/generation.py
from .config import GENERATION_PROMPT
try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

class SimpleGenerator:
    def __init__(self, model_name='google/flan-t5-large'):
        if hf_pipeline is None:
            raise ImportError("transformers must be installed to use generation.")
        self.summarizer = hf_pipeline('text2text-generation', model=model_name, tokenizer=model_name)

    def generate(self, context, query):
        prompt = GENERATION_PROMPT.format(context=context, query=query)
        response = self.summarizer(prompt, max_new_tokens=256, do_sample=False)[0]['generated_text']
        return response.strip()

# If transformers is not available, fallback to echo context and query.
def fallback_generate(context, query):
    return f"[SIMULATED]: Context: {context}\nQuestion: {query}\n[Install HuggingFace transformers for generation]"
