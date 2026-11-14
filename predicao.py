from pathlib import Path
from collections import Counter
import json
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "modelo" / "best.pt"
SOURCE = ROOT / "imagens_ineditas"   # imagens que NÃO participaram do treino nem da validação


def main() -> None:
    images = sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < 5:
        raise RuntimeError("Coloque pelo menos cinco imagens em imagens_ineditas.")
    if not WEIGHTS.exists():
        raise FileNotFoundError("Treine o modelo antes: python treinamento.py")

    results = YOLO(str(WEIGHTS)).predict(
        source=[str(p) for p in images], conf=0.25, save=True,
        project=str(ROOT / "resultados"), name="imagens_ineditas", exist_ok=True,
    )
    resumo = {}
    for path, r in zip(images, results):
        nomes = [r.names[int(c)] for c in r.boxes.cls]
        confs = [round(float(c), 2) for c in r.boxes.conf]
        resumo[path.name] = {"contagem": dict(Counter(nomes)), "deteccoes": list(zip(nomes, confs))}
        print(f"{path.name}: {dict(Counter(nomes))}")
    (ROOT / "resultados" / "deteccoes_ineditas.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Imagens com as detecções salvas em resultados/imagens_ineditas/")


if __name__ == "__main__":
    main()
