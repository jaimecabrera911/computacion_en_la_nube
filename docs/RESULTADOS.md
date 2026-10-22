# 📊 Resultados Medidos del Benchmark Semántico

> **Este documento se genera automáticamente** con `uv run rag report`, a partir de `output/benchmark_summary.json` (el JSON que produce el propio pipeline al ejecutarse). Por lo tanto, **todos los números aquí corresponden a una ejecución real y medible** contra el clúster de Qdrant Cloud.

**Colección:** `clinical_heart_disease_rag` · **Modo:** cloud · **Modelo:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## 1. Corpus indexado

| Componente | Cantidad |
|---|---|
| Viñetas clínicas (dataset UCI Heart Disease, 303 pacientes) | 303 |
| Fragmentos documentales (3 guías clínicas) | 46 |
| **Total de vectores en la colección** | **349** |

## 2. Batería oficial de consultas (Top-1 y Top-5)

| ID | Enfoque | Filtro Qdrant | Top-1 | Media Top-5 | Mín | Máx | Fuente recuperada |
|---|---|---|---|---|---|---|---|
| **Q1** | Criterios de angina típica y estratificación de riesgo en dolor torácico | `doc_type: 'clinical_guideline'` | **0.6154** | 0.5431 | 0.4734 | 0.6154 | Guía de Práctica Clínica: Diagnóstico y Estratificación de la Enfermedad Arterial Coronaria |
| **Q2** | Significado clínico de la depresión del ST > 2 mm con pendiente descendente | `doc_type: 'diagnostic_protocol'` | **0.5733** | 0.4443 | 0.368 | 0.5733 | Protocolo Clínico de Pruebas de Esfuerzo, Ergometría y Angiografía Coronaria |
| **Q3** | Metas de c-LDL con estatinas y doble antiagregación plaquetaria (DAPT) | `doc_type: 'pharmacology_protocol'` | **0.5764** | 0.5284 | 0.4772 | 0.5764 | Protocolo Clínico de Manejo Terapéutico, Farmacológico y Prevención Cardiovascular Secundaria |
| **Q4** | Cohorte de pacientes masculinos mayores de 60 años con cardiopatía confirmada | `doc_type: 'patient_record' + sex: 'Masculino' + disease_present: true + age >= 60` | **0.7014** | 0.6863 | 0.6549 | 0.7014 | Paciente #125 |
| **Q5** | Consulta híbrida abierta: isquemia inducible y decisión terapéutica | `Sin filtro (consulta cross-modal)` | **0.5676** | 0.5286 | 0.5039 | 0.5676 | Protocolo Clínico de Pruebas de Esfuerzo, Ergometría y Angiografía Coronaria |

## 3. Métricas agregadas

| Métrica | Valor |
|---|---|
| Similitud coseno media (Top-1) | **0.6068** |
| Similitud coseno máxima (Top-1) | 0.7014 |
| Similitud coseno mínima (Top-1) | 0.5676 |
| Similitud coseno media (todos los Top-5) | 0.5461 |
| Latencia de búsqueda remota (mediana) | **102.6 ms** |
| Latencia de búsqueda remota (media) | 104.6 ms |
| Latencia mínima / máxima | 100.7 / 120.9 ms |
| Vectorización local (SBERT en CPU) | 14.5 ms |

## 4. Degradación de la similitud por rango de recuperación

Una búsqueda semántica sana **ordena por relevancia**: el score del Rank 1 debe ser el mayor y debe decrecer de forma monótona. Si no fuera así, el sistema no estaría ordenando por relevancia.

| Rank | Similitud media |
|---|---|
| 1 | 0.6068 |
| 2 | 0.5603 |
| 3 | 0.5446 |
| 4 | 0.5235 |
| 5 | 0.4955 |

**Degradación monótona decreciente:** sí ✅

## 5. Fragmentos recuperados (Top-1 de cada consulta)

### Q1 — Criterios de angina típica y estratificación de riesgo en dolor torácico

- **Pregunta:** ¿Cuáles son los criterios clínicos para diagnosticar angina típica y cuál es el algoritmo de estratificación de riesgo en dolor torácico?
- **Filtro:** `doc_type: 'clinical_guideline'`
- **Similitud coseno:** `0.6154`
- **Fuente recuperada:** Guía de Práctica Clínica: Diagnóstico y Estratificación de la Enfermedad Arterial Coronaria
- **Resultado esperado:** Guía MinSalud: angina tipo 1 e indicación de ergometría.
- **Fragmento devuelto:**

  > ## 5. Algoritmo de Decisión Clínica y Priorización 1. Todo paciente con angina típica (Tipo 1) y factores de riesgo concurrentes (edad > 55 años, HTA estadio 1-

### Q2 — Significado clínico de la depresión del ST > 2 mm con pendiente descendente

- **Pregunta:** ¿Qué significado clínico tiene una depresión del segmento ST mayor a 2 mm con pendiente plana o descendente durante la prueba de esfuerzo?
- **Filtro:** `doc_type: 'diagnostic_protocol'`
- **Similitud coseno:** `0.5733`
- **Fuente recuperada:** Protocolo Clínico de Pruebas de Esfuerzo, Ergometría y Angiografía Coronaria
- **Resultado esperado:** Protocolo ACC/AHA: isquemia severa multivaso (slope 3).
- **Fragmento devuelto:**

  > ### 3.1. Depresión del Segmento ST Inducida por Esfuerzo (Oldpeak) - **Medición:** Descenso del punto J y del segmento ST medido a 60-80 ms posterior al punto J

### Q3 — Metas de c-LDL con estatinas y doble antiagregación plaquetaria (DAPT)

- **Pregunta:** ¿Cuáles son las metas de c-LDL con estatinas de alta potencia y el esquema de doble antiagregación plaquetaria con aspirina y ticagrelor?
- **Filtro:** `doc_type: 'pharmacology_protocol'`
- **Similitud coseno:** `0.5764`
- **Fuente recuperada:** Protocolo Clínico de Manejo Terapéutico, Farmacológico y Prevención Cardiovascular Secundaria
- **Resultado esperado:** Protocolo ESC/OMS: meta c-LDL < 55 mg/dL y DAPT por 12 meses.
- **Fragmento devuelto:**

  > ### 2.2. Optimización Lipídica Agresiva - **Meta de Colesterol LDL (c-LDL):**   - Reducción $\ge 50\%$ respecto al valor basal y alcanzar un c-LDL absoluto `< 5

### Q4 — Cohorte de pacientes masculinos mayores de 60 años con cardiopatía confirmada

- **Pregunta:** Paciente masculino mayor de 60 años con angina típica, hipertensión estadio 2, fluoroscopía con vasos mayores comprometidos y alto riesgo cardiovascular
- **Filtro:** `doc_type: 'patient_record' + sex: 'Masculino' + disease_present: true + age >= 60`
- **Similitud coseno:** `0.7014`
- **Fuente recuperada:** Paciente #125
- **Resultado esperado:** Historia clínica de paciente de alto riesgo del dataset UCI.
- **Fragmento devuelto:**

  > HISTORIA CLÍNICA CARDIOLÓGICA - PACIENTE #125 Datos Demográficos: Paciente de 65 años de edad, sexo Masculino. Motivo de Consulta y Dolor Torácico: Angina típic

### Q5 — Consulta híbrida abierta: isquemia inducible y decisión terapéutica

- **Pregunta:** Tratamiento de revascularización y pronóstico para defecto de perfusión reversible en gammagrafía SPECT con angina inducida por ejercicio
- **Filtro:** `Sin filtro (consulta cross-modal)`
- **Similitud coseno:** `0.5676`
- **Fuente recuperada:** Protocolo Clínico de Pruebas de Esfuerzo, Ergometría y Angiografía Coronaria
- **Resultado esperado:** Correlación entre hallazgo gammagráfico y cateterismo coronario.
- **Fragmento devuelto:**

  > ## 5. Centellografía de Perfusión Miocárdica / Gammagrafía (Thal) El estudio de perfusión miocárdica con radiotrazadores (Talio-201 o Tecnecio-99m Sestamibi / S

---

*Documento generado automáticamente. Para regenerarlo: `uv run rag report`.*
