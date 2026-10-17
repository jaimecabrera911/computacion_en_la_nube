import nbformat
from nbclient import NotebookClient

def run_notebook(nb_path: str):
    print(f"Iniciando ejecución secuencial de {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    # Also mirror to notebooks/
    mirror_path = "notebooks/heart_disease_rag_qdrant.ipynb"
    with open(mirror_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Ejecución completa y exitosa guardada en {nb_path} y {mirror_path}.")

if __name__ == "__main__":
    run_notebook("heart_disease_rag_qdrant.ipynb")
