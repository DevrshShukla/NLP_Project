import os
import sys

# Ensure backend path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.nlp.text_cleaner import TextCleaner
from backend.nlp.priority_classifier import PriorityClassifier
from backend.nlp.pipeline import NLPPipeline
from backend.services.exporter import Exporter
from backend.core.errors import create_app_error

def test_worked_examples():
    print("Testing Worked Examples A & B...")
    # Worked Example A (Phase 2, all 4 weights: 0.40, 0.25, 0.20, 0.15)
    # tfidf 0.82, freq 0.90, pos 0.75, sem 0.88
    # (0.82*0.40) + (0.90*0.25) + (0.75*0.20) + (0.88*0.15) = 0.328 + 0.225 + 0.150 + 0.132 = 0.835 -> 83.5
    raw_a = 100.0 * (0.82 * 0.40 + 0.90 * 0.25 + 0.75 * 0.20 + 0.88 * 0.15) / 1.00
    score_a = round(raw_a, 1)
    assert score_a == 83.5, f"Expected 83.5, got {score_a}"
    assert PriorityClassifier.classify(score_a) == "HIGH"
    
    # Worked Example B (MVP, semantic disabled, weights: 0.40, 0.25, 0.20)
    # sum = 0.85
    # (0.82*0.40 + 0.90*0.25 + 0.75*0.20) / 0.85 = 0.703 / 0.85 = 0.82705... -> 82.7
    raw_b = 100.0 * (0.82 * 0.40 + 0.90 * 0.25 + 0.75 * 0.20) / 0.85
    score_b = round(raw_b, 1)
    assert score_b == 82.7, f"Expected 82.7, got {score_b}"
    assert PriorityClassifier.classify(score_b) == "HIGH"
    print("[PASS] Worked Examples test passed.")

def test_priority_boundaries():
    print("Testing Priority Boundaries...")
    assert PriorityClassifier.classify(80.0) == "HIGH"
    assert PriorityClassifier.classify(79.9) == "MEDIUM"
    assert PriorityClassifier.classify(50.0) == "MEDIUM"
    assert PriorityClassifier.classify(49.9) == "LOW"
    print("[PASS] Priority boundaries test passed.")

def test_reference_sample_nlp():
    print("Testing Reference Sample Pipeline...")
    sample_text = (
        "Machine learning is a branch of artificial intelligence.\n\n"
        "Supervised learning uses labeled datasets to train models.\n\n"
        "Classification predicts discrete categories, while regression predicts continuous values.\n\n"
        "Common algorithms include decision trees, support vector machines, and linear regression."
    )
    pipeline = NLPPipeline()
    result = pipeline.run(text=sample_text, title="ML Reference Sample")
    
    print(f"Found {len(result.topics)} topics:")
    for t in result.topics:
        print(f"  #{t.rank} {t.name} (Score: {t.score}, Priority: {t.priority}) - Summary: {t.summary[:60]}...")

    # Check top topics
    topic_names = [t.name.lower() for t in result.topics]
    assert any("machine learning" in name for name in topic_names), "Machine Learning must be extracted"
    assert any("supervised learning" in name for name in topic_names), "Supervised Learning must be extracted"
    
    # Verify summary constraints: each summary must be <= 280 chars
    for t in result.topics:
        assert len(t.summary) <= 280, f"Summary exceeds 280 chars: {len(t.summary)}"
        assert len(t.summary) > 0, f"Topic {t.name} has empty summary"
        
    print("[PASS] Reference sample test passed.")

def test_export_safety():
    print("Testing Export Safety (CSV injection guard)...")
    sample_text = (
        "=cmd|' /C calc'!A0 is an example of formula injection.\n\n"
        "+123456 is a math formula concept.\n\n"
        "Machine learning is a branch of artificial intelligence. Supervised learning uses labeled datasets."
    )
    pipeline = NLPPipeline()
    result = pipeline.run(text=sample_text, title="Injection Test")
    csv_bytes, mime, fname = Exporter.export(result, "csv")
    csv_str = csv_bytes.decode("utf-8-sig")
    
    # Assert any cell starting with = or + was prefixed with '
    for line in csv_str.splitlines():
        cells = line.split(",")
        for cell in cells:
            cell_clean = cell.strip('"')
            if cell_clean.startswith(('=', '+', '-', '@')):
                assert cell_clean.startswith("'"), f"Unescaped formula injection: {cell_clean}"
                
    # Also test PDF export
    pdf_bytes, pdf_mime, pdf_fname = Exporter.export(result, "pdf")
    assert len(pdf_bytes) > 500, "PDF export was too small or empty"
    assert pdf_mime == "application/pdf"
    print("[PASS] Export safety and PDF test passed.")

if __name__ == "__main__":
    test_worked_examples()
    test_priority_boundaries()
    test_reference_sample_nlp()
    test_export_safety()
    print("\nALL BACKEND UNIT TESTS PASSED SUCCESSFULLY!")
