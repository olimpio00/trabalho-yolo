"""Converte a saída real do console (treino_log.txt / validacao_log.txt) em imagem PNG,
para servir de captura de tela do treinamento e da validação no relatório."""
from pathlib import Path
import re
import sys
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS = re.compile(r"\d+%\s")
FONT ="C:/Windows/Fonts/consola.ttf"


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = ANSI.sub("", raw.split("\r")[-1]).rstrip()
        if PROGRESS.search(line) and "100%" not in line:
            continue  # barra de progresso: fica só o estado final
        lines.append(line)
    return lines


def render(lines: list[str], out: Path, title: str) -> None:
    font = ImageFont.truetype(FONT, 15)
    char_w = font.getbbox("M")[2]
    width = max(len(l) for l in lines) * char_w + 40
    height = len(lines) * 19 + 60
    img = Image.new("RGB", (max(width, 900), height), (12, 12, 12))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 28], fill=(40, 40, 40))
    d.text((12, 6), title, font=font, fill=(200, 200, 200))
    y = 40
    for line in lines:
        color = (204, 204, 204)
        if "all" in line.split()[:1] or line.strip().startswith("all"):
            color = (110, 220, 110)
        elif line.strip().startswith(("Epoch", "Class")):
            color = (120, 180, 255)
        d.text((16, y), line, font=font, fill=color)
        y += 19
    img.save(out)


def pick(lines: list[str], spec: str) -> list[str]:
    """spec: intervalos de linhas, ex.: "1,3,29-30,131-155" ou "-38:" (últimas 38)."""
    if spec.endswith(":"):
        return lines[int(spec[:-1]):]
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += lines[int(a):int(b or a) + 1]
    return out


def main(log_name: str, out_name: str, spec: str, title: str) -> None:
    lines = clean_lines((ROOT / log_name).read_text(encoding="utf-8", errors="replace"))
    selected = [l[:150] + (" ..." if len(l) > 150 else "") for l in pick([l for l in lines if l.strip()], spec)]
    out = ROOT / "resultados" / out_name
    out.parent.mkdir(exist_ok=True)
    render(selected, out, title)
    print(out)


if __name__ == "__main__":
    # uso: python gerar_print_console.py treino_log.txt print_treino_fim.png "-38:" "titulo"
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
