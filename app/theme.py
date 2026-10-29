"""Sistema de diseño de la app de sustentación.

Metáfora: **expediente clínico**. Todo lo visual nace del objeto real del
dominio — la historia clínica de un paciente en un hospital — y no de una
paleta de moda:

- El fondo es papel de expediente, con reglas tenues cada 30 px.
- Los paneles son hojas sueltas, con sombra de fotocopia (desplazada, dura)
  en vez de sombras suaves.
- Los datos clínicos van en IBM Plex Mono, como los que teclea un equipo.
- El único acento cálido es el rojo de sello (--sello), que se reserva para
  el riesgo alto y las marcas de verificación.

Reglas que este tema evita a propósito (son las marcas de una página
generada): degradados en texto, círculos de glow decorativos, radios
grandes en todo, eyebrows en MAYÚSCULAS con tracking, y chips redondeados
como única forma de etiquetar.
"""

from __future__ import annotations

# ============ PALETA ============

PALETTE = {
    # Superficies: papel → hoja → hondo (sidebar)
    "papel": "#E9EAE6",
    "papel_hondo": "#DFE1DB",
    "hoja": "#FBFBF9",
    # Tinta: tres pesos de gris, nunca negro puro
    "tinta": "#1C1F1D",
    "tinta_2": "#5A615C",
    "tinta_3": "#8A918B",
    # Reglas de expediente
    "regla": "#C4C9C2",
    "regla_suave": "#DCDED8",
    # Azul tinta de escribir (institucional)
    "azul": "#22485E",
    "azul_2": "#3C6C86",
    # Rojo de sello (único acento cálido)
    "sello": "#8C3A2E",
}

# ============ ICONOGRAFÍA PROPIA ============
# Trazo de 1.6, rejilla de 24, sin relleno. Dibujados a mano para que la
# interfaz no dependa del emoji del sistema operativo (que cambia entre
# Windows, macOS y Linux — y la demo se graba en Linux).

_ICONOS = {
    "carpeta": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "diana": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.2"/>',
    "capas": '<path d="M12 3 3 8l9 5 9-5z"/><path d="m3 13 9 5 9-5"/>',
    "base": '<ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6"/><path d="M4.5 12v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>',
    "lupa": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 4.5 4.5"/>',
    "grafico": '<path d="M4 20V4"/><path d="M4 20h16"/><path d="M7.5 16V11"/><path d="M12 16V7"/><path d="M16.5 16v-6"/>',
    "sello": '<path d="M5 12.5 9.5 17 19 7.5"/>',
    "libro": '<path d="M4 5a2 2 0 0 1 2-2h12v18H6a2 2 0 0 1-2-2z"/><path d="M18 3v18"/>',
    "reloj": '<circle cx="12" cy="12" r="8"/><path d="M12 7.5V12l3 1.8"/>',
    "persona": '<circle cx="12" cy="8" r="3.6"/><path d="M4.8 20a7.2 7.2 0 0 1 14.4 0"/>',
    "corazon": '<path d="M12 20s-7-4.4-7-9a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 4.6-7 9-7 9z"/>',
    "cerebro": '<path d="M12 5.5a3 3 0 0 0-5.6-1.4A2.8 2.8 0 0 0 4 7.8a3 3 0 0 0 .6 4.6A3 3 0 0 0 6.5 17a3 3 0 0 0 5.5 1.4z"/><path d="M12 5.5a3 3 0 0 1 5.6-1.4A2.8 2.8 0 0 1 20 7.8a3 3 0 0 1-.6 4.6A3 3 0 0 1 17.5 17a3 3 0 0 1-5.5 1.4z"/><path d="M12 5.5v13"/>',
    "alerta": '<path d="M12 4 2.8 20h18.4z"/><path d="M12 10v4"/><path d="M12 17.2v.1"/>',
    "rayo": '<path d="M13.5 3 5 13.5h5.5L10 21l8.5-10.5H13z"/>',
    "candado": '<rect x="5" y="10.5" width="14" height="9.5" rx="1.6"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/>',
    "documento": '<path d="M6 3h7l5 5v13H6z"/><path d="M13 3v5h5"/>',
    "pesos": '<circle cx="12" cy="12" r="8"/><path d="M9 8.5 12 13l3-4.5"/><path d="M12 13v4"/>',
    "capilla": '<path d="M4 20V8l8-4 8 4v12z"/><path d="M9.5 20v-6h5v6"/>',
}


def icono(nombre: str, clase: str = "ico") -> str:
    """Devuelve el SVG inline de un icono propio (sin dependencias)."""
    trazo = _ICONOS.get(nombre, _ICONOS["documento"])
    return (
        f'<svg class="{clase}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{trazo}</svg>'
    )


# ============ CSS ============

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;0,700;1,400&display=swap');

:root {
    --papel: #E9EAE6;
    --papel-hondo: #DFE1DB;
    --hoja: #FBFBF9;
    --tinta: #1C1F1D;
    --tinta-2: #5A615C;
    --tinta-3: #8A918B;
    --regla: #C4C9C2;
    --regla-suave: #DCDED8;
    --azul: #22485E;
    --azul-2: #3C6C86;
    --sello: #8C3A2E;
    --sello-claro: rgba(140, 58, 46, 0.10);
    --sombra-copia: 3px 3px 0 rgba(28, 31, 29, 0.09);
}

/* ---------- BASE ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--tinta);
}
/* Streamlit 1.66 aplica su tema con clases `st-emotion-*` de alta
   especificidad. Sin `!important` el tema gana y todo cae en Source Sans.
   Por eso la tipografía se declara sobre las clases reales, no sobre `h2`.
   Excepción: los iconos de Material Symbols usan ligaduras de fuente, así
   que hay que dejarles su propia familia o el nombre del icono se imprime
   como texto ("keyboard_double_arrow_left"). */
.stApp, .stApp p, .stApp div, .stApp label,
.stApp span:not([data-testid="stIconMaterial"]) {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp .seccion-cabecera h2, .stApp .carpeta h1,
.stApp .ficha h3, .stApp .resultado .r-fuente,
.stApp .estación .t {
    font-family: 'IBM Plex Serif', Georgia, 'Times New Roman', serif;
    color: var(--tinta);
    letter-spacing: -0.012em;
}
.stApp .dato-num, .stApp .r-meta, .stApp .r-puesto,
.stApp .lectura span, .stApp .datos-clinicos,
.stApp .seccion-cabecera .marca, .stApp .seccion-cabecera .etiqueta,
.stApp .pie-indice, .stApp .marca-app .t2, .stApp .dato-lbl {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
}
.stApp {
    background-color: var(--papel);
    background-image: repeating-linear-gradient(
        to bottom, transparent 0 29px, rgba(28, 31, 29, 0.028) 29px 30px
    );
}
code, pre, .mono {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}
code {
    background: rgba(34, 72, 94, 0.08) !important;
    color: var(--azul) !important;
    border: 1px solid rgba(34, 72, 94, 0.14);
    border-radius: 2px !important;
    padding: 0.05em 0.32em !important;
    font-size: 0.86em !important;
}
p, li { color: var(--tinta-2); }
a { color: var(--azul); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Foco visible: teclado es un usuario más */
:focus-visible {
    outline: 2px solid var(--azul) !important;
    outline-offset: 2px;
}

/* ---------- ICONOS PROPIOS ---------- */
.ico {
    width: 1em; height: 1em;
    vertical-align: -0.13em;
    flex: none;
}
.ico-lg { width: 1.6em; height: 1.6em; }
.ico-xl { width: 2.4em; height: 2.4em; }

/* ---------- CARPETA (portada) ---------- */
.carpeta {
    position: relative;
    background: var(--hoja);
    border: 1px solid var(--regla);
    border-radius: 2px;
    box-shadow: var(--sombra-copia);
    padding: 2.4rem 2.6rem 2.2rem;
    margin-bottom: 1.5rem;
}
.carpeta::before {
    content: "";
    position: absolute;
    top: -1px; left: 0;
    height: 6px; width: 190px;
    background: var(--azul);
}
.carpeta h1 {
    font-size: 2.35rem;
    font-weight: 600;
    line-height: 1.16;
    margin: 0 0 0.7rem 0;
    color: var(--tinta);
    text-wrap: balance;
}
.carpeta h1 em {
    font-style: italic;
    color: var(--azul);
}
.carpeta .entradilla {
    font-size: 1.15rem;
    line-height: 1.55;
    color: var(--tinta-2);
    /* 90ch ocupa el ancho de la portada en pantallas normales sin volver
       ilegibles las líneas en monitores ultrawide. */
    max-width: 90ch;
    margin-bottom: 1.4rem;
}
/* Pie de la portada: integrantes y datos del curso en dos columnas */
.carpeta .portada-pie {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr);
    gap: 1.6rem;
    border-top: 1px solid var(--regla);
    padding-top: 1.1rem;
}
.carpeta .rotulo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--tinta-3);
    margin-bottom: 0.4rem;
}
.carpeta .integrantes {
    list-style: none;
    margin: 0;
    padding: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: var(--tinta-2);
}
.carpeta .linea-dato {
    margin: 0 0 0.15rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--tinta-2);
}
.carpeta .portada-pie a {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--azul);
    text-decoration: none;
    border-bottom: 1px solid rgba(34, 72, 94, 0.35);
    overflow-wrap: anywhere;
}
.carpeta .portada-pie a:hover { border-bottom-color: currentColor; }
@media (max-width: 760px) {
    .carpeta .portada-pie { grid-template-columns: 1fr; }
}

/* ---------- SELLO ---------- */
/* .sello = solo el color de tinta roja, para marcar riesgo/severidad. */
.sello { color: var(--sello); }

/* ---------- CABECERA DE SECCIÓN ---------- */
.seccion-cabecera {
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
    border-bottom: 1px solid var(--regla);
    padding-bottom: 0.85rem;
    margin-bottom: 1.6rem;
}
.seccion-cabecera .marca {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--azul);
    border-left: 2px solid var(--azul);
    padding: 0.15rem 0 0.15rem 0.6rem;
    line-height: 1.4;
}
.seccion-cabecera .etiqueta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--tinta-3);
    margin-bottom: 0.2rem;
}
.seccion-cabecera h2 {
    font-size: 1.62rem;
    font-weight: 600;
    line-height: 1.25;
    margin: 0;
}

/* ---------- FICHA / HOJA ---------- */
/* Igualar alturas dentro de `st.columns` es una pelea: Streamlit centra el
   markdown (`align-items:center`) en un contenedor que ya tiene la altura de
   la columna, así que `height:100%` sobre la ficha no la estira — la
   desborda, y las tarjetas quedan de alturas distintas.
   La solución es no usar `st.columns` para filas de fichas y dejarlas en
   una rejilla CSS: en grid, los hijos se estiran a la fila por defecto, así
   que las alturas quedan iguales sin trucos de CSS. */
.rejilla {
    display: grid;
    gap: 1rem;
    margin-bottom: 1.4rem;
}
.ficha {
    background: var(--hoja);
    border: 1px solid var(--regla);
    border-radius: 2px;
    box-shadow: var(--sombra-copia);
    padding: 1.4rem 1.6rem;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.ficha h3 {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--regla-suave);
}
.ficha h3 .ico { color: var(--azul); }
.lead {
    font-size: 0.94rem;
    line-height: 1.72;
    color: var(--tinta-2);
}
.lead b, .lead strong { color: var(--tinta); font-weight: 600; }
.lead ul { padding-left: 1.05rem; margin: 0.5rem 0 0 0; }
.lead li { margin-bottom: 0.42rem; }

/* ---------- TABLA DE DATOS (métricas) ---------- */
.datos {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
    background: var(--hoja);
    border: 1px solid var(--regla);
    border-radius: 2px;
    box-shadow: var(--sombra-copia);
    margin-bottom: 1.4rem;
}
.dato {
    padding: 0.95rem 1.05rem;
    border-right: 1px solid var(--regla-suave);
    border-bottom: 1px solid var(--regla-suave);
}
.dato-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--azul);
    line-height: 1.15;
    letter-spacing: -0.03em;
}
.dato-num.sello { color: var(--sello); border: none; outline: none; }
.dato-lbl {
    font-size: 0.74rem;
    color: var(--tinta-3);
    margin-top: 0.3rem;
    line-height: 1.35;
}

/* ---------- CINTA DE PROCESO (pasos) ---------- */
.cinta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
    gap: 0;
    background: var(--hoja);
    border: 1px solid var(--regla);
    border-radius: 2px;
    box-shadow: var(--sombra-copia);
    margin-bottom: 1.5rem;
}
.estación {
    padding: 1.05rem 1.05rem 1.15rem;
    border-right: 1px solid var(--regla-suave);
    position: relative;
    height: 100%;
}
.estación:last-child { border-right: none; }
.estación .n {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--azul);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.45rem;
}
.estación .n .ico { color: var(--azul); }
.estación .t {
    font-family: 'IBM Plex Serif', serif;
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--tinta);
    margin-bottom: 0.32rem;
}
.estación .d {
    font-size: 0.78rem;
    line-height: 1.55;
    color: var(--tinta-2);
}

/* ---------- FICHA DE RESULTADO (demo) ---------- */
.resultado {
    background: var(--hoja);
    border: 1px solid var(--regla);
    border-left: 3px solid var(--azul);
    border-radius: 2px;
    padding: 0;
    margin-bottom: 0.9rem;
    overflow: hidden;
}
.resultado .r-cab {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.6rem;
    flex-wrap: wrap;
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--regla-suave);
    background: rgba(34, 72, 94, 0.045);
}
.resultado .r-puesto {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--azul);
    letter-spacing: 0.04em;
}
.resultado .r-fuente {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--tinta);
}
.resultado .r-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0 1.1rem;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--regla-suave);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--tinta-2);
}
.resultado .r-meta b { color: var(--tinta); font-weight: 600; }
/* El fragmento llega renderizado como HTML (títulos, listas): sin pre-wrap
   ni rayado de expediente — con bloques HTML ambos dejaban líneas fantasma
   en los espacios entre elementos. */
.resultado .r-texto {
    font-size: 0.86rem;
    line-height: 1.6;
    padding: 0.5rem 1rem 0.7rem;
    color: var(--tinta-2);
    max-height: 320px;
    overflow-y: auto;
}

/* ---------- FICHA DE DATOS CLÍNICOS ---------- */
.datos-clinicos {
    display: flex;
    flex-wrap: wrap;
    gap: 0 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--tinta-2);
}
.datos-clinicos span b { color: var(--tinta); font-weight: 600; }
.r-riesgo { color: var(--sello); font-weight: 600; }

/* ---------- CONSULTAS DE EJEMPLO ---------- */
.ejemplos {
    border-left: 2px solid var(--regla);
    padding-left: 0.85rem;
}
.ejemplos-t {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--tinta-3);
    margin-bottom: 0.4rem;
}
.ejemplo {
    font-size: 0.86rem;
    color: var(--tinta-2);
    line-height: 1.5;
    padding: 0.16rem 0;
}
.ejemplo::before {
    content: "—";
    color: var(--regla);
    margin-right: 0.45rem;
}

/* ---------- BARRA DE SIMILITUD ---------- */
.barra {
    height: 5px;
    background: rgba(28, 31, 29, 0.09);
    border-radius: 0;
    overflow: hidden;
}
.barra i {
    display: block;
    height: 100%;
    background: var(--azul);
}
.barra i.alto { background: var(--sello); }

/* ---------- ETIQUETAS DE INTERPRETACIÓN ---------- */
.lectura {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.35rem 0 0.5rem;
}
.lectura span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--azul);
    background: rgba(34, 72, 94, 0.07);
    border-left: 2px solid var(--azul);
    padding: 0.16rem 0.5rem;
}
.lectura span.sello {
    color: var(--sello);
    background: var(--sello-claro);
    border-left-color: var(--sello);
}

/* ---------- PREGUNTA ORIENTADORA ---------- */
.orienta {
    background: var(--hoja);
    border: 1px solid var(--regla);
    border-left: 3px solid var(--sello);
    border-radius: 2px;
    padding: 1.5rem 1.8rem;
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.12rem;
    line-height: 1.62;
    color: var(--tinta);
    font-style: italic;
    box-shadow: var(--sombra-copia);
    margin-bottom: 1.5rem;
}

/* ---------- ÍNDICE LATERAL ---------- */
section[data-testid="stSidebar"] {
    background-color: var(--papel-hondo);
    background-image: repeating-linear-gradient(
        to bottom, transparent 0 29px, rgba(28, 31, 29, 0.035) 29px 30px
    );
    border-right: 1px solid var(--regla);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.marca-app {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding-bottom: 0.9rem;
    margin-bottom: 1.1rem;
    border-bottom: 1px solid var(--regla);
}
.marca-app .ico { color: var(--azul); }
.marca-app .t1 {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.12rem;
    font-weight: 600;
    color: var(--tinta);
    line-height: 1.15;
}
.marca-app .t2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--tinta-3);
    margin-top: 0.15rem;
}

/* Índice numerado de expediente: los números son CSS counters, así el
   marcado de Streamlit no se manipula y la accesibilidad se conserva.
   Ojo: en Streamlit 1.66 cada opción es un `label` con
   `data-testid="stRadioOption"` anidado dentro de DIVs intermedios, por eso
   el selector es de descendencia y no de hijo directo. */
section[data-testid="stSidebar"] [role="radiogroup"] { counter-reset: indice; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
    counter-increment: indice;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.4rem 0.45rem;
    border-left: 2px solid transparent;
    border-radius: 0;
    cursor: pointer;
    transition: border-color .12s ease, background-color .12s ease;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]::before {
    content: counter(indice, decimal-leading-zero);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--tinta-3);
    flex: none;
    width: 1.4em;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) {
    border-left-color: var(--azul);
    background: rgba(34, 72, 94, 0.06);
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked)::before {
    color: var(--azul);
    font-weight: 600;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {
    background: rgba(34, 72, 94, 0.045);
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {
    font-size: 0.86rem;
    color: var(--tinta-2);
    line-height: 1.35;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p {
    color: var(--tinta);
    font-weight: 600;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] input[type="radio"] {
    display: none;
}
/* El radio nativo se sustituye por una marca de expediente: un cuadro
   relleno con un guion, como el check de una casilla de formulario. */
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {
    width: 14px !important;
    height: 14px !important;
    border-radius: 0 !important;
    border: 1px solid var(--regla) !important;
    background: var(--hoja) !important;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: center;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child > div {
    width: 7px !important;
    height: 7px !important;
    border-radius: 0 !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) > div > div:first-child {
    background: var(--azul) !important;
    border-color: var(--azul) !important;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) > div > div:first-child > div {
    background: var(--hoja) !important;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:focus-within {
    outline: 2px solid var(--azul);
    outline-offset: 1px;
}

.pie-indice {
    margin-top: 1.5rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--regla);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    line-height: 1.75;
    color: var(--tinta-3);
}
.pie-indice b { color: var(--azul); font-weight: 600; }

/* ---------- REFERENCIAS ---------- */
.ref {
    padding: 0.72rem 0 0.72rem 1.6rem;
    border-bottom: 1px solid var(--regla-suave);
    color: var(--tinta-2);
    line-height: 1.7;
    font-size: 0.9rem;
    position: relative;
}
.ref::before {
    content: "";
    position: absolute;
    left: 0; top: 1.05rem;
    width: 9px; height: 1px;
    background: var(--azul);
}

/* ---------- STREAMLIT NATIVE ---------- */
div[data-testid="stDataFrame"] > div {
    border: 1px solid var(--regla) !important;
    border-radius: 2px !important;
    overflow: hidden;
    box-shadow: var(--sombra-copia);
}
details[role="group"] {
    border: 1px solid var(--regla) !important;
    border-radius: 2px !important;
    background: var(--hoja) !important;
}

/* Avisos (éxito / error / info): el verde y el rojo por defecto de Streamlit
   no pertenecen a la paleta del expediente. Se sustituyen por tinta azul
   sobre papel, con una barra lateral de color — como una nota al pie
   firmada, no como una notificación del sistema operativo. */
div[data-testid="stAlert"] {
    border-radius: 2px !important;
    border: 1px solid var(--regla) !important;
    border-left: 3px solid var(--azul) !important;
    background: var(--hoja) !important;
    box-shadow: var(--sombra-copia);
}
div[data-testid="stAlert"][data-baseweb="kind"]:has([data-testid="stIconMaterial"]) { color: var(--tinta-2); }
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span { color: var(--tinta-2) !important; }
div[data-testid="stAlert"] b { color: var(--tinta) !important; }
div[data-testid="stAlert"] code {
    background: rgba(34, 72, 94, 0.07) !important;
    border: 1px solid rgba(34, 72, 94, 0.14);
    color: var(--azul) !important;
}
div[data-testid="stAlert"] svg { color: var(--azul) !important; }
/* El aviso de error sí lleva el rojo de sello: ahí el color es información. */
div[data-testid="stAlert"]:has([data-testid="stIconMaterial"]) { border-left-color: var(--azul) !important; }

.stButton > button {
    border-radius: 2px;
    border: 1px solid var(--azul);
    background: var(--azul);
    color: #FBFBF9 !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 2px 2px 0 rgba(28, 31, 29, 0.16);
    transition: transform .1s ease, box-shadow .1s ease;
}
.stButton > button:hover {
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 rgba(28, 31, 29, 0.2);
    color: #FBFBF9 !important;
    border-color: var(--azul);
}
.stButton > button:active {
    transform: translate(0, 0);
    box-shadow: 1px 1px 0 rgba(28, 31, 29, 0.16);
}
/* Streamlit pinta el <p> del label con su gris (rgb 90,97,92) y ese gris
   gana por especificidad sobre el color del botón: aquí hereda el del botón. */
.stButton > button p,
.stDownloadButton > button p {
    color: inherit !important;
}
.stButton > button[kind="secondary"] {
    background: var(--hoja);
    color: var(--azul) !important;
}
/* Los botones de descarga de entregables van en papel, no en tinta: son
   enlaces de archivo, no la acción principal de la página. */
.stDownloadButton > button {
    border-radius: 2px;
    border: 1px solid var(--regla);
    background: var(--hoja);
    color: var(--azul) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    box-shadow: 1px 1px 0 rgba(28, 31, 29, 0.08);
}
.stDownloadButton > button:hover {
    border-color: var(--azul);
    color: var(--azul) !important;
}
div[data-testid="stTextInput"] textarea,
div[data-testid="stTextArea"] textarea {
    background: var(--hoja) !important;
    border: 1px solid var(--regla) !important;
    border-radius: 2px !important;
    color: var(--tinta) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    box-shadow: inset 2px 0 0 var(--azul);
}
div[data-testid="stTextInput"] textarea:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--azul) !important;
}
.stSlider [data-testid="stSliderThumbValue"] { font-family: 'IBM Plex Mono', monospace; }
section[data-testid="stExpander"] summary { font-family: 'IBM Plex Sans', sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap: 0.2rem; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Sans', sans-serif; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: rgba(34, 72, 94, 0.30); }
::-webkit-scrollbar-track { background: transparent; }

/* ---------- MOVIMIENTO REDUCIDO ---------- */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
    .stButton > button:hover { transform: none; }
}
</style>
"""
