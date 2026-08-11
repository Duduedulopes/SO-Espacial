# SO Espacial

Sistema de percepção espacial para ambientes de varejo — o "gêmeo digital" de
uma loja física, construído a partir de câmeras.

Projeto de pesquisa e estudo. Prazo aberto.

---

## O que é, em uma frase

Transformar imagens de câmeras em um **modelo vivo do espaço**: quem está na
loja, onde, e o que está fazendo — mantido em coordenadas de planta baixa, não
em pixels.

## O que não é

Não é um detector de objetos. Detecção é a primeira linha do problema, não o
problema.

O problema difícil é **manter identidade** ao longo do tempo e entre câmeras,
sob oclusão e mudança de aparência. Toda a literatura da área gira em torno
disso.

## Nome técnico do campo

Para buscar literatura, use os termos certos:

- **MTMC** — Multi-Target Multi-Camera Tracking
- **MOT** — Multiple Object Tracking
- **Re-ID** — Person Re-Identification
- **Spatial Intelligence / World Models** — o guarda-chuva de 2026

"Digital twin de loja" traz marketing. "MTMC tracking" traz ciência.

---

## A escada

Cada degrau funciona e demonstra sozinho. Nada aqui depende de terminar tudo.

| # | Degrau | Entrega |
|---|---|---|
| 0 | **Dataset e benchmark** | Aparato experimental: dados sincronizados e como medir |
| 1 | Detecção | Pessoas e objetos em quadros isolados |
| 2 | Rastreamento | Identidade que persiste entre quadros |
| 3 | Zonas e permanência | Mapa de calor, tempo de permanência — já vendável sozinho |
| 4 | **Homografia** | Vista de cima em tempo real — o gêmeo digital nasce aqui |
| 5 | Multi-câmera | Mesma pessoa, várias câmeras, um só ID |
| 6 | Interação mão-produto | O degrau mais alto |
| 7 | Fusão com RFID | Encontro com o sistema da loja autônoma |

Estamos no **degrau 0**.

---

## Por que começar pelo dataset

O gargalo da área não é modelo nem GPU. É **dado anotado**. Anotar "esta mão
pegou este produto neste quadro" é caro e lento.

Este projeto tem um ativo raro: um sistema de loja autônoma com **RFID
funcionando**, que sabe com precisão de milissegundos qual produto foi retirado
e quando. Vídeo sincronizado com esses eventos vira **rotulagem automática**.

Isso se chama **supervisão fraca cross-modal** — um sinal de uma modalidade
rotulando outra. É a pergunta de pesquisa central deste projeto:

> Em que medida eventos RFID sincronizados podem substituir anotação humana no
> treinamento de modelos de interação mão-produto em varejo?

---

## Arquitetura

Pipeline de estágios desacoplados:

```
Fonte     →   Percepção    →   Estado          →   Saída
frames        detecções        gêmeo digital       grava / exibe / publica
```

**Duas decisões que valem mais que todas as outras:**

**1. A Fonte é uma interface, com duas implementações:** webcam ao vivo e
arquivo gravado. O mesmo código roda nos dois casos. É o que torna os
experimentos reproduzíveis — você grava uma vez e experimenta cem vezes sobre
o mesmo material.

**2. Captura e processamento são programas separados.** A captura precisa ser
burra, estável e nunca falhar; ela cria um ativo insubstituível. O
processamento é experimental e vai quebrar toda semana. Juntar os dois faz um
bug no detector derrubar a gravação — e dado perdido não volta.

### Onde entra o C#

Pesquisa e treino em Python. Produção pode ser em C#:

```
Python (laboratório)   →  ONNX  →   C# (fábrica)
treino, experimentos               EdgeDesktop roda a inferência
```

O modelo treinado é exportado para ONNX e executado com `Microsoft.ML.OnnxRuntime`
dentro do EdgeDesktop — na borda, no PC da loja. Nenhuma imagem sai do local,
que é a promessa de privacidade do projeto.

---

## Estrutura

```
SO-Espacial/
├─ captura/          fonte de vídeo e gravação — código estável
│    fonte.py          câmera em thread, descarta quadros velhos
│    gravar.py         grava sessões com carimbo de tempo
│    diagnostico.py    mede fps, brilho, codec da câmera
├─ calibracao/       imagem → metros no chão
│    homografia.py     calibração interativa (bloco 1)
├─ percepcao/        do pixel ao mundo
│    chao.py           BIBLIOTECA: homografia, ponto do pé, filtros
│    pose3d.py         BIBLIOTECA: pose 3D relativa, inclinação
│    mapa.py           programa: vista 2D de cima
│    gemeo3d.py        programa: o gêmeo completo
├─ estado/           o mundo, sem desenho
│    rastreio.py       Kalman 2D e recostura de identidade
│    ocupacao.py       mapa de calor e zonas
│    planta.py         loja lida de JSON, estado publicado
├─ visual/           só desenho, não sabe de câmera
│    cena2d.py         vista de cima
│    cena3d.py         esqueletos numa cena 3D
├─ loja/             plantas em JSON — loja nova, arquivo novo
├─ experimentos/     programas superados, citados no caderno
├─ dados/            sessões, registros, estado_atual.json
└─ docs/             plano de estudo, esquema de dados, caderno
```

**Regra que organiza tudo isso:** um arquivo é biblioteca **ou** programa,
nunca os dois. Biblioteca não tem `main()`; programa não é importado.

Até 08/08 o `mapa.py` era os dois, e o `gemeo3d.py` importava classes de dentro
dele — então rodar o gêmeo carregava o código de desenho do mapa, e mexer num
quebrava o outro. O núcleo foi para `percepcao/chao.py`.

**Regra do `dados/bruto`: somente escrita.** Nada é corrigido, limpo ou
sobrescrito ali. Todo processamento gera arquivo novo em outro lugar. Parece
exagero até o dia em que você descobre um erro no processamento e precisa
refazer tudo sem regravar nada.

---

## Ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

IDE: **VS Code**, com as extensões Python, Pylance e Jupyter.

---

## Documentos

- [`docs/DATASET.md`](docs/DATASET.md) — esquema de dados e protocolo de captura
- [`docs/PLANO-DE-ESTUDO.md`](docs/PLANO-DE-ESTUDO.md) — o que estudar, em que ordem
