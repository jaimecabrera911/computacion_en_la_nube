import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from src.config import settings

DOCUMENT_METADATA = {
    "DOC_01_Guia_Clinica_Enfermedad_Coronaria_MinSalud_OMS.md": {
        "title": "Guía de Práctica Clínica: Diagnóstico y Estratificación de la Enfermedad Arterial Coronaria",
        "entity": "MinSalud / OPS / OMS",
        "code": "GPC-CARDIO-2024-V2",
        "doc_type": "clinical_guideline",
        "domain": "cardiology_prevention"
    },
    "DOC_02_Protocolo_Pruebas_Esfuerzo_Ergometria_ACC_AHA.md": {
        "title": "Protocolo Clínico de Pruebas de Esfuerzo, Ergometría y Angiografía Coronaria",
        "entity": "ACC / AHA",
        "code": "PRT-DIAG-CARDIO-ACC-2024",
        "doc_type": "diagnostic_protocol",
        "domain": "ergometry_hemodynamics"
    },
    "DOC_03_Guia_Manejo_Farmacologico_Prevencion_Secundaria.md": {
        "title": "Protocolo Clínico de Manejo Terapéutico, Farmacológico y Prevención Cardiovascular Secundaria",
        "entity": "ESC / OMS",
        "code": "ESC-OMS-PREV-2024",
        "doc_type": "pharmacology_protocol",
        "domain": "secondary_prevention"
    }
}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un archivo PDF utilizando pypdf."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def load_unstructured_documents(documents_dir: str = "data/documents") -> List[Dict[str, Any]]:
    """Carga los documentos complementarios en formato Markdown o PDF."""
    docs = []
    
    for filename, meta in DOCUMENT_METADATA.items():
        md_path = os.path.join(documents_dir, filename)
        pdf_filename = filename.replace(".md", ".pdf")
        pdf_path = os.path.join(documents_dir, pdf_filename)
        
        content = ""
        used_format = "markdown"
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif os.path.exists(pdf_path):
            content = extract_text_from_pdf(pdf_path)
            used_format = "pdf"
        else:
            raise FileNotFoundError(f"No se encontró el documento {filename} ni su versión PDF.")
            
        docs.append({
            "filename": filename,
            "format": used_format,
            "content": content,
            "metadata": meta
        })
    return docs

def chunk_unstructured_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap
) -> List[Dict[str, Any]]:
    """
    Aplica fragmentación recursiva mediante RecursiveCharacterTextSplitter.
    Justificación técnica:
    - chunk_size = 512 caracteres: alineado al límite máximo de atención de modelos
      Sentence-BERT (256 wordpiece tokens) para evitar truncamiento semántico.
    - chunk_overlap = 64 caracteres (~12%): conserva continuidad de frases y preserva
      relaciones contextuales entre límites de fragmentos.
    - separators = ["\n\n", "\n", ". ", " ", ""]: jerarquía que prioriza divisiones en
      párrafos semánticos antes de separar oraciones o palabras.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunked_corpus = []
    chunk_global_id = 0
    
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        base_meta = doc["metadata"]
        
        for local_idx, split_text in enumerate(splits):
            chunk_global_id += 1
            meta = {
                **base_meta,
                "filename": doc["filename"],
                "chunk_id": chunk_global_id,
                "local_chunk_index": local_idx,
                "chunk_char_length": len(split_text)
            }
            chunked_corpus.append({
                "text": split_text,
                "metadata": meta
            })
            
    return chunked_corpus
