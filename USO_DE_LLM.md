# Declaração de uso de LLM

Conforme orientação do professor, declaro que este trabalho foi desenvolvido com o auxílio de um modelo de linguagem (LLM).

| | |
|---|---|
| **Ferramenta** | Claude Code — assistente de programação da Anthropic |
| **Modelo** | Claude Opus 5.5 |
| **Onde** | Extensão do VS Code, no Windows |
| **Quando** | Novembro de 2026 |

## Para que foi usado

1. Revisar a primeira versão do projeto, quando foi identificado o **vazamento de dados** entre treino e validação: as fotos de validação eram cópias escurecidas das fotos de treino.
2. Escrever os scripts Python: `prepare_dataset.py`, `treinamento.py`, `validar.py`, `predicao.py`, `gerar_print_console.py` e `gerar_relatorio.py`.
3. Instalar o PyTorch com CUDA para treinar na GPU (NVIDIA RTX 4060) em vez da CPU.
4. Executar os treinamentos, a validação e as predições, e comparar os experimentos: YOLO11s sem e com congelamento do backbone.
5. Redigir o rascunho do relatório (`relatorio.pdf`) a partir dos resultados obtidos.

## Principais pedidos feitos ao LLM

- "fazer essa última atividade sobre YOLO, treinar ela mais um pouco também e fazer o relatório"
- "tá usando a CPU e não a GPU, né? Queria usar a GPU pra ir mais rápido e melhor"
- incluir no trabalho a declaração de uso de LLM pedida pelo professor

## Papel do aluno

Definição do tema e do escopo, acompanhamento da execução, decisão de treinar na GPU, conferência dos resultados e revisão final do relatório.

## Verificação

Todas as métricas, gráficos e imagens do relatório foram gerados pela execução real dos scripts. A saída completa do console está em `treino_log.txt` e `validacao_log.txt`, e tudo pode ser reproduzido com os comandos de `comandos.txt`.
