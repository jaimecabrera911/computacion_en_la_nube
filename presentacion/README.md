# Presentación Interactiva Reveal.js: Arquitectura RAG Cloud Serverless en Salud

Diapositivas interactivas desarrolladas en **Reveal.js + HTML5 + CSS3** basadas en la arquitectura clínica cardiológica, con diseño adaptativo de alta fidelidad, animaciones fluidas, notas de ponente y enlaces directos a la demostración en vivo.

---

## 🚀 Inicio Rápido

### Opción 1: Con Python (Inmediato sin dependencias externas)
Desde la raíz del repositorio:
```bash
make slides
```
O directamente dentro de esta carpeta:
```bash
python3 -m http.server 8080
```
Luego abre en tu navegador: **[http://localhost:8080](http://localhost:8080)**

---

### Opción 2: Con Node / npm
```bash
cd presentacion
npm start
```

---

### Opción 3: Abrir directamente el archivo HTML
Puedes abrir `presentacion/index.html` con doble clic o con tu navegador favorito. Todos los estilos y scripts de Reveal.js están empaquetados localmente en `vendor/reveal.js/` para garantizar funcionamiento **100% offline**.

---

## ⌨️ Atajos de Navegación

| Tecla / Acción | Función |
|---|---|
| <kbd>→</kbd> / <kbd>Espacio</kbd> | Siguiente diapositiva |
| <kbd>←</kbd> | Diapositiva anterior |
| <kbd>O</kbd> o <kbd>ESC</kbd> | **Vista general (Overview)** en cuadrícula de las 14 diapositivas |
| <kbd>S</kbd> | **Modo Ponente (Speaker Notes)** con temporizador, cronómetro y notas guiadas |
| <kbd>F</kbd> | Modo Pantalla Completa |
| <kbd>Alt</kbd> + Clic | Zoom a cualquier elemento de la diapositiva |
| <kbd>?</kbd> | Abre el modal de ayuda con todos los atajos |

---

## 📂 Estructura del Proyecto

```
presentacion/
├── index.html              # Lienzo Reveal.js con las 14 diapositivas estructuradas y notas
├── css/
│   ├── theme.css           # Variables de color, tema clínico-tecnológico, HUD flotante y modal
│   └── custom.css          # Estructura visual, diagramas SVG, medidores de similitud y tarjetas
├── js/
│   └── presentation.js     # Configuración Reveal.js, sincronización del HUD y atajos
├── assets/                 # Recursos gráficos en alta resolución (diagramas, medidores, iconos)
├── vendor/
│   └── reveal.js/          # Dependencias Reveal.js empaquetadas para soporte offline
├── package.json            # Metadatos del proyecto y scripts de ejecución
└── README.md               # Esta documentación
```

---

## 🎯 Contenido de las 14 Diapositivas

1. **01 · Portada Institucional:** Título, autores, especialización y créditos académicos.
2. **02 · Problema y Objetivo:** Pregunta de investigación, meta central y los 4 hitos operativos.
3. **03 · Ecosistema de Datos Clínicos:** UCI Heart Disease + 3 Guías de práctica clínica integradas.
4. **04 · Motor RAG: Flujo y Pipeline:** Pipeline completo de 5 pasos desde ingesta hasta consulta semántica.
5. **05 · Stack Tecnológico & Chunking:** uv, Python 3.12, Qdrant Cloud, paraphrase-multilingual-MiniLM-L12-v2 y RecursiveSplitter.
6. **06 · Batería de Pruebas (Q1 - Q5):** Estrategia de estrés cognitivo y métricas de similitud coseno.
7. **07 · Escenario Q1:** Diagnóstico de angina típica y algoritmo de riesgo (Similitud: 0.6154).
8. **08 · Escenario Q2:** Análisis de ergometría y depresión del segmento ST (Similitud: 0.5733).
9. **09 · Escenario Q3:** Farmacología cardiológica y metas c-LDL (Similitud: 0.5764).
10. **10 · Escenario Q4:** Filtrado multivariable complejo con lógica booleana AND/OR (Similitud: 0.7014).
11. **11 · Escenario Q5:** Inferencia global cross-modal sobre todo el corpus (Similitud: 0.5676).
12. **12 · Rendimiento Cuantitativo:** Gráficas de distribución Top-5 y similitud media por rango (Top-1: 0.607).
13. **13 · Conclusiones Técnicas:** Latencia (<150 ms, mediana 103 ms), costo cero y valor clínico.
14. **14 · Demo en Vivo & Enlaces:** Tarjetas interactivas conectadas a Streamlit Cloud, GitHub y Google Drive.
