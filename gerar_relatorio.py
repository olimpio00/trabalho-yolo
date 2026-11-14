"""Gera o relatorio.pdf a partir dos resultados reais salvos em runs/ e resultados/."""
from pathlib import Path
import csv
import json
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).parent
RUN = ROOT / "runs" / "treino_veiculos"
RES = ROOT / "resultados"
CFG = json.loads((ROOT / "relatorio_textos.json").read_text(encoding="utf-8"))

AZUL = colors.HexColor("#1f3a5f")
CINZA = colors.HexColor("#eef2f6")
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, textColor=AZUL, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, textColor=AZUL, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
CELL = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8.5, leading=10.5)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8, leading=10, backColor=CINZA, borderPadding=6)
TITLE = ParagraphStyle("T", parent=styles["Title"], fontSize=20, textColor=AZUL, leading=24)
SUB = ParagraphStyle("S", parent=BODY, alignment=TA_CENTER, fontSize=12, textColor=colors.HexColor("#2b6777"))

MAX_W = 17 * cm


def p(text, style=BODY):
    return Paragraph(text, style)


def fig(path: Path, caption: str, width=MAX_W, max_h=11 * cm):
    w, h = PILImage.open(path).size
    scale = min(width / w, max_h / h)
    return KeepTogether([Image(str(path), w * scale, h * scale), p(caption, CAP), Spacer(1, 6)])


def table(rows, widths, header=True, align_right_from=1):
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")),
             ("FONTSIZE", (0, 0), (-1, -1), 9),
             ("ALIGN", (align_right_from, 0), (-1, -1), "CENTER"),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), AZUL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), CINZA))
    t.setStyle(TableStyle(style))
    return t


def dataset_counts():
    names = CFG["classes"]
    out = {}
    for split in ("train", "val"):
        imgs = {n: 0 for n in names}
        inst = {n: 0 for n in names}
        files = list((ROOT / "dataset" / "labels" / split).glob("*.txt"))
        for f in files:
            seen = set()
            for line in f.read_text().splitlines():
                c = names[int(line.split()[0])]
                inst[c] += 1
                seen.add(c)
            for c in seen:
                imgs[c] += 1
        out[split] = {"total": len(files), "imgs": imgs, "inst": inst}
    return out


def best_epoch(run: Path):
    rows = list(csv.DictReader((run / "results.csv").open()))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    best = max(rows, key=lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]))
    return len(rows), best


def main() -> None:
    m = json.loads((RES / "metricas_validacao.json").read_text(encoding="utf-8"))
    det = json.loads((RES / "deteccoes_ineditas.json").read_text(encoding="utf-8"))
    counts = dataset_counts()
    n_epochs, best = best_epoch(RUN)
    total_imgs = counts["train"]["total"] + counts["val"]["total"]

    s = []
    # ---------- Identificação ----------
    s += [Spacer(1, 1.2 * cm), p("Atividade Prática Final — YOLO", SUB), Spacer(1, 4),
          p(CFG["titulo"], TITLE), Spacer(1, 0.4 * cm)]
    s.append(table([["Aluno", CFG["aluno"]], ["Disciplina", "Inteligência Artificial"],
                    ["Instituição", CFG["instituicao"]], ["Data", CFG["data"]],
                    ["Modelo", f"{CFG['modelo']} (pesos pré-treinados) — {n_epochs} épocas executadas"],
                    ["Dataset", f"{total_imgs} imagens anotadas, {len(CFG['classes'])} classes"]],
                   [4 * cm, 12 * cm], header=False, align_right_from=99))
    s += [Spacer(1, 0.5 * cm), p("<b>Resumo.</b> " + CFG["resumo"])]

    # ---------- 1. Descrição ----------
    s += [p("1. Descrição do projeto", H1), p(CFG["descricao"])]

    # ---------- 2. Dataset ----------
    s += [p("2. Dataset (Etapa 1)", H1), p(CFG["dataset"])]
    rows = [["Classe", "Imagens treino", "Objetos treino", "Imagens val.", "Objetos val."]]
    for c in CFG["classes"]:
        rows.append([c, counts["train"]["imgs"][c], counts["train"]["inst"][c],
                     counts["val"]["imgs"][c], counts["val"]["inst"][c]])
    rows.append(["Total de imagens", counts["train"]["total"], sum(counts["train"]["inst"].values()),
                 counts["val"]["total"], sum(counts["val"]["inst"].values())])
    s += [table(rows, [4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]),
          p("Tabela 1 — Distribuição das classes (uma imagem pode conter mais de uma classe).", CAP), Spacer(1, 4),
          p("<b>Formato das anotações.</b> Cada imagem tem um arquivo .txt com uma linha por objeto, no formato "
            "<font name='Courier'>classe x_centro y_centro largura altura</font>, com coordenadas normalizadas entre 0 e 1. "
            "Exemplo real (<font name='Courier'>" + CFG["exemplo_label_arquivo"] + "</font>):"),
          Preformatted(CFG["exemplo_label"], CODE), Spacer(1, 6),
          p("<b>data.yaml utilizado:</b>"), Preformatted((ROOT / "dataset" / "data.yaml").read_text(encoding="utf-8"), CODE),
          Spacer(1, 6),
          fig(RUN / "val_batch0_labels.jpg", "Figura 1 — Exemplos de imagens do conjunto de validação com as anotações (caixas reais).", max_h=12 * cm),
          fig(RUN / "labels.jpg", "Figura 2 — Distribuição das anotações: quantidade por classe, posição e tamanho das caixas.", max_h=9 * cm)]

    # ---------- 3. Treinamento ----------
    s += [PageBreak(), p("3. Configuração e treinamento (Etapas 2 e 3)", H1), p(CFG["treinamento"]),
          Preformatted(CFG["comando_treino"], CODE), Spacer(1, 6),
          fig(RES / "print_treino_inicio.png", "Figura 3 — Console no início do treinamento (ambiente, GPU e dataset).", max_h=9 * cm),
          fig(RES / "print_treino_fim.png", "Figura 4 — Console no fim do treinamento: early stopping, geração do best.pt e validação final.", max_h=10 * cm),
          fig(RUN / "results.png", f"Figura 5 — Evolução das perdas (losses) e métricas ao longo das {n_epochs} épocas.", max_h=9 * cm),
          p(CFG["analise_curvas"])]

    # ---------- 4. Validação ----------
    s += [PageBreak(), p("4. Validação do modelo (Etapa 4)", H1), p(CFG["validacao"])]
    rows = [["Classe", "Precision", "Recall", "mAP50", "mAP50-95"],
            ["Todas (média)", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['map50']:.3f}", f"{m['map50_95']:.3f}"]]
    for c, v in m["por_classe"].items():
        rows.append([c, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['map50']:.3f}", f"{v['map50_95']:.3f}"])
    s += [table(rows, [4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]),
          p("Tabela 2 — Métricas de validação do best.pt (77 imagens que não participaram do treino).", CAP),
          fig(RES / "print_validacao.png", "Figura 6 — Saída do console do comando de validação.", max_h=6 * cm),
          p("<b>O que cada indicador representa</b>", H2)]
    for item in CFG["explicacao_metricas"]:
        s.append(p(item))
    s += [fig(ROOT / "runs" / "validacao" / "confusion_matrix_normalized.png",
              "Figura 7 — Matriz de confusão normalizada (validação). 'background' = objeto não detectado ou detecção sem objeto real.", max_h=11 * cm),
          p(CFG["analise_confusao"]),
          fig(ROOT / "runs" / "validacao" / "val_batch0_pred.jpg", "Figura 8 — Predições do modelo nas mesmas imagens de validação da Figura 1.", max_h=12 * cm)]

    # ---------- 5. Inéditas ----------
    s += [PageBreak(), p("5. Reconhecimento em imagens inéditas (Etapa 5)", H1), p(CFG["ineditas"])]
    for i, (name, info) in enumerate(det.items(), start=1):
        cont = ", ".join(f"{v} {k}" for k, v in info["contagem"].items())
        s.append(fig(RES / "imagens_ineditas" / (Path(name).stem + ".jpg"),
                     f"Figura {8 + i} — {name}: {cont}.", max_h=8.5 * cm))
        s.append(p(CFG["comentarios_ineditas"][name]))

    # ---------- 6. Experimentos ----------
    s += [PageBreak(), p("6. Experimentos e evolução do projeto", H1), p(CFG["experimentos_intro"]),
          table([CFG["experimentos_tabela"][0]] + [[p(str(c), CELL) for c in row] for row in CFG["experimentos_tabela"][1:]],
                [2.8 * cm, 5.0 * cm, 2.6 * cm, 2.4 * cm, 4.2 * cm]),
          p("Tabela 3 — Comparação entre as versões testadas.", CAP)]
    for par in CFG["experimentos_analise"]:
        s.append(p(par))

    # ---------- 7. Questões ----------
    s += [p("7. Respostas às questões (Etapa 6)", H1)]
    for q, a in CFG["questoes"]:
        s += [p(f"<b>{q}</b>", BODY), p(a)]

    s += [p("8. Conclusão", H1), p(CFG["conclusao"]),
          p("9. Organização da entrega", H1), Preformatted(CFG["estrutura"], CODE),
          p("10. Uso de Inteligência Artificial generativa (LLM)", H1)]
    s += [p(par) for par in CFG["uso_llm"]]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1 * cm, f"Inteligência Artificial — Atividade Prática Final YOLO — {CFG['aluno']}")
        canvas.drawRightString(19 * cm, 1 * cm, f"Página {doc.page}")
        canvas.restoreState()

    out = ROOT / "relatorio.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                      bottomMargin=1.6 * cm, title=CFG["titulo"], author=CFG["aluno"]).build(s, onFirstPage=footer, onLaterPages=footer)
    print(out)


if __name__ == "__main__":
    main()
