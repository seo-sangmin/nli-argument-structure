"""
Text summarization module.

Uses the LongT5 model (fine-tuned for book summarization)
to generate summaries of the CRA and MBP texts.
"""

import textwrap

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "pszemraj/long-t5-tglobal-base-16384-book-summary"


def create_summarizer():
    """Load the LongT5 tokenizer and model for summarization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


def summarize_text(summarizer, text):
    """Summarize a text and return the summary string."""
    tokenizer = summarizer["tokenizer"]
    model = summarizer["model"]
    device = summarizer["device"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=16384,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=512,
            min_length=32,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def wrap_text(text, width=100):
    """Wrap text for convenient reading."""
    return textwrap.fill(text, width=width)


def summarize_and_display(summarizer, cra_text, mbp_text):
    """Summarize both texts and print wrapped results.

    Returns:
        tuple: (wrapped_cra_summary, wrapped_mbp_summary)
    """
    cra_summary = summarize_text(summarizer, cra_text)
    mbp_summary = summarize_text(summarizer, mbp_text)

    wrapped_cra = wrap_text(cra_summary)
    wrapped_mbp = wrap_text(mbp_summary)

    print("Summary of CRA:")
    print(wrapped_cra)
    print()
    print("Summary of MBP:")
    print(wrapped_mbp)
    print()

    return wrapped_cra, wrapped_mbp
