import os

folders = [
    "01_data_understanding",
    "02_data_preparation",
    "03_modeling",
    "04_evaluation",
    "05_deployment",
    "artifacts"
]

files = {
    "README.md": "# Proyecto Examen Machine Learning\n\n## Descripción\nProyecto de scoring de riesgo crediticio.\n\n## Estructura\nBasada en CRISP-DM.",
    "requirements.txt": "pandas\nnumpy\nscikit-learn\nmatplotlib\nseaborn\npyarrow\nfastapi\nuvicorn\njoblib\n",
    ".gitignore": "# Datos\n*.parquet\n*.csv\n*.zip\n\n# Modelos grandes\n*.pkl\n*.joblib\n\n# Entornos y temporales\n__pycache__/\n.env\n.ipynb_checkpoints/\nvenv/\n"
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, ".gitkeep"), "w") as f:
            pass
        print(f"✅ Carpeta creada: {folder}")

    for filename, content in files.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Archivo creado: {filename}")

if __name__ == "__main__":
    create_structure()
    print("\n🚀 ¡Estructura lista! Inicializa git ahora con: git init")