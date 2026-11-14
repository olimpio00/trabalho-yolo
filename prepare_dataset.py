"""Monta o dataset de veículos a partir do COCO 2017 (val2017).

- Baixa as anotações do COCO 2017 já convertidas para o formato YOLO (Ultralytics).
- Filtra apenas as imagens que contêm bicicleta, carro, moto, ônibus ou caminhão.
- Seleciona um subconjunto equilibrado entre as classes (todas as fotos são distintas).
- Baixa só as imagens selecionadas e separa 80% treino / 20% validação por imagem,
  garantindo que nenhuma foto apareça nos dois conjuntos.
"""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import random
import shutil

ROOT = Path(__file__).parent
DOWNLOADS = ROOT / "downloads"
DATASET = ROOT / "dataset"
LABELS_ZIP = DOWNLOADS / "coco2017labels.zip"
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{}.jpg"

# Índices COCO: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
COCO_TO_LOCAL = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4}
NAMES = ["bicicleta", "carro", "moto", "onibus", "caminhao"]
IMAGES_PER_CLASS = 90      # quantas imagens sortear contendo cada classe
MIN_BOX_AREA = 0.004       # ignora imagens em que o veículo é só um ponto ao fundo
SEED = 42


def load_vehicle_labels() -> dict[str, list[str]]:
    DOWNLOADS.mkdir(exist_ok=True)
    if not LABELS_ZIP.exists():
        print("Baixando anotações do COCO 2017 (~48 MB)...")
        urlretrieve(LABELS_URL, LABELS_ZIP)
    result = {}
    with ZipFile(LABELS_ZIP) as archive:
        for member in archive.namelist():
            if "/labels/val2017/" not in member or not member.endswith(".txt"):
                continue
            rows, big_enough = [], False
            for line in archive.read(member).decode().splitlines():
                fields = line.split()
                if fields and int(fields[0]) in COCO_TO_LOCAL:
                    fields[0] = str(COCO_TO_LOCAL[int(fields[0])])
                    rows.append(" ".join(fields))
                    if float(fields[3]) * float(fields[4]) >= MIN_BOX_AREA:
                        big_enough = True
            if rows and big_enough:
                result[Path(member).stem] = rows
    return result


def select_balanced(labels: dict[str, list[str]]) -> list[str]:
    rng = random.Random(SEED)
    selected: set[str] = set()
    for cls in range(len(NAMES)):
        with_class = sorted(s for s, rows in labels.items() if any(r.split()[0] == str(cls) for r in rows))
        rng.shuffle(with_class)
        selected.update(with_class[:IMAGES_PER_CLASS])
    stems = sorted(selected)
    rng.shuffle(stems)
    return stems


def download(stem: str, folder: Path) -> None:
    target = folder / f"{stem}.jpg"
    if not target.exists():
        urlretrieve(IMAGE_URL.format(stem), target)


def main() -> None:
    labels = load_vehicle_labels()
    print(f"Imagens do COCO val2017 com veículos: {len(labels)}")
    stems = select_balanced(labels)

    if DATASET.exists():
        shutil.rmtree(DATASET)
    split_at = round(len(stems) * 0.8)
    splits = {"train": stems[:split_at], "val": stems[split_at:]}
    for split, items in splits.items():
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda s: download(s, img_dir), items))
        for stem in items:
            (lbl_dir / f"{stem}.txt").write_text("\n".join(labels[stem]) + "\n", encoding="utf-8")

    assert not set(splits["train"]) & set(splits["val"]), "vazamento entre treino e validação"

    (DATASET / "data.yaml").write_text(
        f"path: {DATASET.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )

    print(f"Dataset criado: {len(stems)} imagens ({len(splits['train'])} treino, {len(splits['val'])} validação)")
    for split, items in splits.items():
        counts = Counter(r.split()[0] for s in items for r in labels[s])
        print(f"  {split}: " + ", ".join(f"{NAMES[i]}={counts.get(str(i), 0)}" for i in range(len(NAMES))))


if __name__ == "__main__":
    main()
