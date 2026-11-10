/**
 * Control y configuración de la presentación Reveal.js
 * Proyecto: Arquitectura RAG Cloud Serverless en Salud
 */

document.addEventListener('DOMContentLoaded', () => {
  // Inicialización de Reveal.js
  Reveal.initialize({
    width: 1376,
    height: 768,
    margin: 0.02,
    minScale: 0.2,
    maxScale: 2.5,
    center: false,
    hash: true,
    history: true,
    controls: true,
    controlsTutorial: false,
    controlsLayout: 'bottom-right',
    controlsBackArrows: 'faded',
    progress: true,
    slideNumber: 'c / t',
    showSlideNumber: 'all',
    transition: 'slide',
    transitionSpeed: 'fast',
    backgroundTransition: 'fade',
    keyboard: {
      // Atajo personalizado para el modal de ayuda
      191: () => toggleShortcutsModal(), // '?'
    },
    plugins: [
      typeof RevealNotes !== 'undefined' ? RevealNotes : null,
      typeof RevealZoom !== 'undefined' ? RevealZoom : null
    ].filter(Boolean)
  });

  // Nombres de diapositivas para el HUD
  const slideTitles = [
    '01 · Portada Institucional',
    '02 · Problema y Objetivo',
    '03 · Ecosistema de Datos Clínicos',
    '04 · Motor RAG: Flujo y Pipeline',
    '05 · Stack Tecnológico & Chunking',
    '06 · Batería de Pruebas (Q1 - Q5)',
    '07 · Escenario Q1: Diagnóstico',
    '08 · Escenario Q2: Ergometría',
    '09 · Escenario Q3: Farmacología',
    '10 · Escenario Q4: Filtro Multivariable',
    '11 · Escenario Q5: Inferencia Global',
    '12 · Rendimiento Cuantitativo',
    '13 · Conclusiones Técnicas',
    '14 · Demo en Vivo & Enlaces'
  ];

  const hudTitle = document.getElementById('hud-slide-title');

  Reveal.on('slidechanged', event => {
    const idx = event.indexh;
    if (hudTitle && slideTitles[idx]) {
      hudTitle.textContent = slideTitles[idx];
    }
  });

  // Configuración de botones del HUD
  const btnOverview = document.getElementById('btn-overview');
  if (btnOverview) {
    btnOverview.addEventListener('click', (e) => {
      e.preventDefault();
      Reveal.toggleOverview();
    });
  }

  const btnNotes = document.getElementById('btn-notes');
  if (btnNotes) {
    btnNotes.addEventListener('click', (e) => {
      e.preventDefault();
      const notesPlugin = Reveal.getPlugin('notes');
      if (notesPlugin && typeof notesPlugin.open === 'function') {
        notesPlugin.open();
      } else {
        window.open('?notes=true', 'reveal-notes', 'width=1100,height=700');
      }
    });
  }

  const btnFullscreen = document.getElementById('btn-fullscreen');
  if (btnFullscreen) {
    btnFullscreen.addEventListener('click', (e) => {
      e.preventDefault();
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });
  }

  const btnHelp = document.getElementById('btn-help');
  const shortcutsModal = document.getElementById('shortcuts-modal');
  const btnCloseShortcuts = document.getElementById('btn-close-shortcuts');

  function toggleShortcutsModal() {
    if (shortcutsModal) {
      shortcutsModal.classList.toggle('open');
    }
  }

  if (btnHelp) {
    btnHelp.addEventListener('click', (e) => {
      e.preventDefault();
      toggleShortcutsModal();
    });
  }

  if (btnCloseShortcuts) {
    btnCloseShortcuts.addEventListener('click', (e) => {
      e.preventDefault();
      toggleShortcutsModal();
    });
  }

  if (shortcutsModal) {
    shortcutsModal.addEventListener('click', (e) => {
      if (e.target === shortcutsModal) {
        shortcutsModal.classList.remove('open');
      }
    });
  }

  // Prevenir que clics en enlaces de diapositiva 14 disparen navegación de Reveal
  document.querySelectorAll('.link-card').forEach(card => {
    card.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  });
});
