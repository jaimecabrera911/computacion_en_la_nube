import pandas as pd
from ucimlrepo import fetch_ucirepo
from typing import List, Dict, Any, Tuple

CP_MAP = {
    1: "Angina típica (Tipo 1, dolor retroesternal opresivo)",
    2: "Angina atípica (Tipo 2, equivalente anginoso)",
    3: "Dolor torácico no anginoso (Tipo 3)",
    4: "Asintomático (Tipo 4, ausencia de dolor espontáneo)"
}

RESTECG_MAP = {
    0: "ECG normal en reposo",
    1: "Anormalidad de repolarización del segmento ST y onda T",
    2: "Hipertrofia ventricular izquierda (HVI)"
}

SLOPE_MAP = {
    1: "Pendiente ascendente (Upsloping)",
    2: "Pendiente horizontal (Flat, alto riesgo isquémico)",
    3: "Pendiente descendente (Downsloping, isquemia severa)"
}

THAL_MAP = {
    3.0: "Perfusión miocárdica normal",
    6.0: "Defecto de perfusión fijo (necrosis / infarto antiguo)",
    7.0: "Defecto de perfusión reversible (isquemia miocárdica activa inducible)"
}

def load_heart_disease_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Descarga e inspecciona el dataset UCI Heart Disease (ID: 45) mediante ucimlrepo."""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    df = pd.concat([X, y], axis=1)
    # Renombrar target num a 'target'
    if 'num' in df.columns:
        df['target'] = df['num']
    return df, heart_disease.metadata

def compute_clinical_risk(row: pd.Series) -> str:
    """Calcula nivel de riesgo clínico basado en hallazgos cardiovasculares."""
    target = row.get('target', 0)
    ca = row.get('ca', 0)
    oldpeak = row.get('oldpeak', 0.0)
    exang = row.get('exang', 0)
    thal = row.get('thal', 3.0)
    
    if target >= 2 or (ca is not None and ca >= 2) or (oldpeak >= 2.5):
        return "Muy Alto / Crítico"
    elif target == 1 or ca == 1 or exang == 1 or thal == 7.0:
        return "Alto"
    elif row.get('trestbps', 120) >= 140 or row.get('chol', 200) >= 240 or row.get('fbs', 0) == 1:
        return "Moderado"
    else:
        return "Bajo"

def patient_row_to_clinical_narrative(row: pd.Series, patient_id: int) -> Dict[str, Any]:
    """
    Transforma un registro tabular de paciente del dataset UCI Heart Disease
    en una viñeta clínica estructurada en lenguaje natural con metadatos asociados.
    """
    age = int(row['age'])
    sex_str = "Masculino" if row['sex'] == 1 else "Femenino"
    cp_val = int(row['cp'])
    cp_desc = CP_MAP.get(cp_val, f"Dolor torácico clase {cp_val}")
    trestbps = int(row['trestbps'])
    chol = int(row['chol'])
    fbs_desc = "Glucemia elevada >120 mg/dl (diabetes/intolerancia)" if row['fbs'] == 1 else "Glucemia basal normal (<=120 mg/dl)"
    restecg_val = int(row['restecg'])
    restecg_desc = RESTECG_MAP.get(restecg_val, "ECG basal no especificado")
    thalach = int(row['thalach'])
    exang_desc = "Presente durante la prueba ergométrica" if row['exang'] == 1 else "Ausente en el esfuerzo"
    oldpeak = float(row['oldpeak'])
    slope_val = int(row['slope'])
    slope_desc = SLOPE_MAP.get(slope_val, f"Pendiente ST tipo {slope_val}")
    
    ca_val = row['ca']
    ca_str = f"{int(ca_val)} vasos mayores" if pd.notna(ca_val) else "No determinado"
    
    thal_val = row['thal']
    thal_desc = THAL_MAP.get(thal_val, "Gammagrafía no concluyente")
    
    target_val = int(row['target'])
    has_disease = target_val > 0
    risk_level = compute_clinical_risk(row)
    
    text = (
        f"HISTORIA CLÍNICA CARDIOLÓGICA - PACIENTE #{patient_id:03d}\n"
        f"Datos Demográficos: Paciente de {age} años de edad, sexo {sex_str}.\n"
        f"Motivo de Consulta y Dolor Torácico: {cp_desc}.\n"
        f"Constantes y Perfil Metabólico: Presión arterial sistólica en reposo de {trestbps} mmHg; "
        f"colesterol sérico total de {chol} mg/dL; {fbs_desc}.\n"
        f"Electrocardiograma Basal: {restecg_desc}.\n"
        f"Evaluación Funcional y Prueba de Esfuerzo (Ergometría): Frecuencia cardíaca pico alcanzada de {thalach} lpm; "
        f"Angina inducida por esfuerzo: {exang_desc}; Depresión del segmento ST de {oldpeak:.1f} mm "
        f"con morfología de {slope_desc}.\n"
        f"Estudios Invasivos y Perfusión: Angiografía fluoroscópica con calcificación/obstrucción en {ca_str}; "
        f"Estudio de perfusión SPECT: {thal_desc}.\n"
        f"Diagnóstico Clínico de Cardiopatía Isquémica: {'PRESENCIA CONFIRMADA' if has_disease else 'AUSENCIA'} "
        f"(Estadio de severidad angiográfica: {target_val}/4). Nivel de Riesgo Global: {risk_level}."
    )
    
    metadata = {
        "doc_type": "patient_record",
        "patient_id": patient_id,
        "age": age,
        "age_group": "<50" if age < 50 else ("50-64" if age < 65 else ">=65"),
        "sex": sex_str,
        "chest_pain_type": cp_val,
        "chest_pain_label": cp_desc.split(" (")[0],
        "resting_blood_pressure": trestbps,
        "cholesterol": chol,
        "fasting_blood_sugar": int(row['fbs']),
        "max_heart_rate": thalach,
        "exercise_angina": int(row['exang']),
        "st_depression": oldpeak,
        "st_slope": slope_val,
        "major_vessels": int(ca_val) if pd.notna(ca_val) else -1,
        "thal_status": int(thal_val) if pd.notna(thal_val) else -1,
        "target_disease": target_val,
        "disease_present": bool(has_disease),
        "risk_level": risk_level
    }
    
    return {"text": text, "metadata": metadata}

def generate_patient_corpus(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convierte todas las filas del dataset en una colección de documentos clínicos estructurados."""
    corpus = []
    for idx, row in df.iterrows():
        corpus.append(patient_row_to_clinical_narrative(row, patient_id=idx+1))
    return corpus
