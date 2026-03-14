from __future__ import annotations

from dataclasses import dataclass
import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


@dataclass
class GenerationOutput:
    text: str
    latency_sec: float
    prompt: str


class HFGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> GenerationOutput:
        start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        end = time.perf_counter()
        return GenerationOutput(text=text, latency_sec=end - start, prompt=prompt)



def build_baseline_prompt(query: str, context: str) -> str:
    return (
        "You are a document assistant. Answer the user question using only the provided context. "
        "Be concise and grounded.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )



def build_context_aware_prompt(query: str, global_summary: str, local_context: str) -> str:
    return (
        "You are a document assistant. Use the global document summary for topic awareness and the local evidence "
        "for factual grounding. If the answer is not supported, say that the evidence is insufficient.\n\n"
        f"Question: {query}\n\n"
        f"Global document summary:\n{global_summary}\n\n"
        f"Local evidence:\n{local_context}\n\n"
        "Answer:"
    )
