from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "dataset" / "data.yaml"
EPOCHS = 150


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Execute primeiro: python prepare_dataset.py")
    model = YOLO("yolo11s.pt")  # modelo leve (versão "small") com pesos pré-treinados
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        device=0,          # GPU NVIDIA RTX 4060 (CUDA)
        patience=40,       # para antes se o mAP de validação não melhorar por 40 épocas
        freeze=10,         # congela as 10 primeiras camadas (backbone pré-treinado) — melhor resultado
        workers=4,
        seed=42,
        project=str(ROOT / "runs"),
        name="treino_veiculos",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
