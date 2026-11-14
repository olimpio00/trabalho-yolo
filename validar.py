from pathlib import Path
import json
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "modelo" / "best.pt"
DATA = ROOT / "dataset" / "data.yaml"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError("Treine o modelo e copie runs/treino_veiculos/weights/best.pt para modelo/")
    metrics = YOLO(str(WEIGHTS)).val(
        data=str(DATA), imgsz=640, batch=8, plots=True,
        project=str(ROOT / "runs"), name="validacao", exist_ok=True,
    )
    box = metrics.box
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall:    {box.mr:.4f}")
    print(f"mAP50:     {box.map50:.4f}")
    print(f"mAP50-95:  {box.map:.4f}")

    per_class = {
        metrics.names[c]: {"precision": float(p), "recall": float(r), "map50": float(m50), "map50_95": float(m)}
        for c, p, r, m50, m in zip(box.ap_class_index, box.p, box.r, box.ap50, box.ap)
    }
    summary = {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
               "map50_95": float(box.map), "por_classe": per_class}
    out = ROOT / "resultados" / "metricas_validacao.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Métricas salvas em {out}")


if __name__ == "__main__":
    main()
