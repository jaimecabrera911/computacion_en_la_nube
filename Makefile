# Atajos de comandos del proyecto ACA N° 1 — Computación en la Nube
#
#   make setup    → instala el entorno con uv
#   make pipeline → ejecuta el pipeline completo (PASO 1 a 5) contra Qdrant Cloud
#   make test     → ejecuta las pruebas automáticas
#   make ask      → abre la consola interactiva de consultas
#
.DEFAULT_GOAL := help
.PHONY: help setup pipeline evaluate ask chat queries test notebook notebook-report poster summary-pdf diagram corpus-pdf report demo clean

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Instala el entorno virtual con uv
	uv sync

pipeline: ## Ejecuta el pipeline completo y guarda las métricas
	uv run rag pipeline

evaluate: ## Ejecuta solo el benchmark semántico
	uv run rag evaluate

ask: ## Consulta puntual: make ask "pregunta"
	uv run rag ask "$(filter-out $@,$(MAKECMDGOALS))"

chat: ## Consola interactiva de consultas
	uv run rag chat

queries: ## Muestra la batería oficial de consultas
	uv run rag queries

test: ## Ejecuta las pruebas automáticas
	uv run pytest

notebook: ## Reejecuta el notebook reproducible y guarda sus outputs
	uv run python scripts/execute_notebook.py

notebook-report: ## Genera el informe del notebook en PDF y HTML
	uv run rag notebook-report

poster: ## Regenera el póster infografía en PDF
	uv run rag poster

summary-pdf: ## Regenera la guía fácil de resultados en PDF
	uv run rag summary-pdf

diagram: ## Regenera el diagrama de arquitectura cloud
	uv run rag diagram

corpus-pdf: ## Regenera los PDF del corpus documental
	uv run rag corpus-pdf

report: ## Regenera docs/RESULTADOS.md con las métricas medidas
	uv run rag report

demo: ## Levanta la app web de sustentación (Streamlit)
	uv run streamlit run app/app_sustentacion.py

clean: ## Elimina artefactos temporales (no toca los entregables)
	find . -path ./.venv -prune -o -name '__pycache__' -type d -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache
