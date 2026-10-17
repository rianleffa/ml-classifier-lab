# Conferência do relatório AP2 com a implementação

Documento consultado: `Relatorio_AP2_Machine_Learning.pdf` (5 páginas). O PDF original foi copiado para esta pasta sem alterações.

## Conteúdo conferido

- A capa informa a instituição/campus, disciplina, curso, estudante Rian Cardoso Leffa e data 17/10/2026.
- O texto descreve classificação supervisionada, 569 registros, 30 atributos e as duas classes.
- O fluxo, o modelo `DecisionTreeClassifier`, a profundidade 4, a divisão estratificada 80/20 e a semente 42 estão especificados.
- As métricas, a matriz de confusão e as cinco partições de validação cruzada são comparadas na validação automatizada do projeto; veja abaixo o resultado medido.
- O relatório discute falsos negativos, a limitação da acurácia isolada, o conjunto maior de registros benignos e a finalidade exclusivamente acadêmica.

## Resultado da execução local

Executei `train_and_evaluate` com Scikit-learn, usando os parâmetros padrão do relatório (`max_depth=4`, teste de 20%, estratificação e `random_state=42`). Os valores foram calculados pelo modelo; não são constantes inseridas no código.

| Medida | Execução local | PDF | Conferência |
|---|---:|---:|---|
| Dataset | 569 registros, 30 atributos | 569, 30 | Igual |
| Distribuição completa | 212 malignos, 357 benignos | 212, 357 | Igual |
| Acurácia no teste | 0,938596 (93,86%) | 0,9386 | Igual por arredondamento |
| Precisão para maligno | 0,906977 (90,70%) | 0,9070 | Igual por arredondamento |
| Recall para maligno | 0,928571 (92,86%) | 0,9286 | Igual por arredondamento |
| F1-score para maligno | 0,917647 (91,76%) | 0,9176 | Igual por arredondamento |
| Matriz (real × previsto, Maligno/Benigno) | `[[39, 3], [4, 68]]` | `[[39, 3], [4, 68]]` | Igual |
| Acurácias CV | 0,9561; 0,8684; 0,9211; 0,9298; 0,9381 | Mesmos valores | Igual por arredondamento |
| Média CV | 0,922698 (92,27%) | 0,9227 | Igual por arredondamento |
| Desvio-padrão CV (`ddof=0`) | 0,029508 | 0,0295 | Igual por arredondamento |

O conjunto de teste tem 42 malignos e 72 benignos. A matriz confere com as métricas: 39/42 malignos foram identificados (3 falsos negativos); entre os benignos, 68/72 foram classificados corretamente (4 falsos positivos). A soma da diagonal é 107/114, que produz a acurácia de 0,938596.

## Exportação PDF da aplicação

A API foi exercitada com uma nova execução (`e54bb731-dfc1-4049-8eab-c00c0a326cbe`). O arquivo retornado pelo endpoint de relatório tem seis páginas A4: capa, página com as oito seções e quatro páginas de gráficos. As páginas foram renderizadas para conferência; títulos, texto, matriz de confusão, validação cruzada, distribuição e métricas estão visíveis. O espaço da legenda do gráfico de validação cruzada foi ajustado para não encostar no valor da última partição.

## Ajustes de coerência feitos no projeto

- `projeto_ap2.py` oferece a execução por linha de comando mencionada no anexo e chama a mesma função que a API usa.
- `backend/app/training.py` é a fonte real exibida na página Código Python; os parâmetros e resultados são compartilhados entre a interface e a execução CLI.
- O desvio-padrão da validação cruzada usa a definição populacional do NumPy (`ddof=0`), igual a `scores.std()` no código do anexo.
- A navegação da aplicação segue a numeração e os títulos do relatório, e mantém uma seção separada para relatórios/evidências.
- As referências bibliográficas devem identificar fontes efetivamente usadas. A versão consultada não contém uma seção formal de referências; não foi acrescentada citação acadêmica inventada.

### Ajustes recomendados no PDF fornecido

1. **Página 4, seção 8:** trocar a frase “O código-fonte completo está no arquivo separado projeto_ap2.py” por uma descrição dos dois arquivos: `projeto_ap2.py` inicia a execução de terminal e `backend/app/training.py` contém a rotina compartilhada de treinamento, métricas e gráficos, que também é usada pela API. A saída gerada pela interface é mantida em `results/runs/`.
2. **Página 5, anexo A:** o bloco exibido no PDF é uma versão anterior da rotina. Para que o código do relatório e a aplicação sejam a mesma implementação, substituir o anexo pelo conteúdo baixado na página **8. Anexo: código-fonte e execução** (fonte `backend/app/training.py`) ou anexar esse arquivo junto ao PDF.
3. **Após a seção 8:** acrescentar uma seção “Referências” com as páginas oficiais realmente consultadas e a data de acesso. A versão original não mostra essa seção.

Referências técnicas verificadas para essa seção sugerida:

- Scikit-learn Developers. *load_breast_cancer*. Documentação da API. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html (acesso em 17 out. 2026).
- Scikit-learn Developers. *DecisionTreeClassifier*. Documentação da API. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html (acesso em 17 out. 2026).
- Scikit-learn Developers. *Cross-validation: evaluating estimator performance*. Guia do usuário. https://scikit-learn.org/stable/modules/cross_validation.html (acesso em 17 out. 2026).

## Capturas

O diretório `docs/evidencias/` contém instruções para capturar telas reais após iniciar o sistema. As imagens devem ser feitas pelo estudante durante a execução; nenhuma captura simulada foi criada.
