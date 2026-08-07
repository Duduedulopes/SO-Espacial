# Plano de estudo

A ordem importa. Cada bloco assume o anterior.

Não há prazo. Há **critério de domínio** — uma coisa que você deve conseguir
fazer antes de seguir adiante. Se não conseguir, o bloco não terminou, por mais
tempo que tenha passado.

---

## Bloco 0 — Python suficiente

Você já programa. Não precisa aprender Python "do zero", precisa da porção que
a área usa.

**O que estudar**

- Sintaxe básica, listas, dicionários, list comprehensions
- `numpy`: arrays, slicing, broadcasting, operações vetorizadas
- Ambientes virtuais e `pip`
- Notebooks Jupyter: célula, kernel, estado

**A mudança mental que importa:** vindo de C#, o instinto é escrever laços. Em
Python científico, laço sobre pixels é 100× mais lento que a operação
vetorizada equivalente. Aprender a *pensar em arrays* é o pulo do gato.

> **Critério de domínio:** carregar uma imagem, inverter as cores, recortar uma
> região e salvar — sem escrever um único `for` sobre pixels.

---

## Bloco 1 — Geometria de câmeras

**Não pule este bloco.** É o alicerce de tudo que é "espacial". Sem ele,
"gêmeo digital" continua sendo metáfora; com ele, vira coordenada.

**O que estudar**

- Modelo pinhole: como o mundo 3D vira imagem 2D
- Parâmetros **intrínsecos** (distância focal, centro óptico) e **extrínsecos**
  (onde a câmera está e para onde aponta)
- Distorção de lente e calibração com tabuleiro de xadrez
- **Homografia** — a transformação entre dois planos. É a peça central do
  degrau 4
- Geometria epipolar e triangulação (para quando houver várias câmeras)

**Material**

- Szeliski, *Computer Vision: Algorithms and Applications* — gratuito e legal em
  szeliski.org/Book. Capítulos 2 e 11.
- Hartley & Zisserman, *Multiple View Geometry* — a bíblia. Denso demais para
  ler linearmente; use como consulta.
- Tutoriais de calibração do OpenCV — prática direta.

> **Critério de domínio:** marcar 4 pontos no chão da sua cena, calcular a
> homografia, e converter a posição de um pé na imagem para coordenada em
> metros na planta. Se a pessoa andar 1 metro real, seu mapa deve mostrar
> 1 metro.

Quando isso funcionar, você terá o gêmeo digital em sua forma mais simples — e
terá entendido por que ele funciona.

---

## Bloco 2 — Estimação de estado

A matemática de "acompanhar uma coisa que se move e às vezes some".

**O que estudar**

- Filtro de Kalman: predição e correção, por que ele lida bem com ruído
- Associação de dados: dadas N detecções e M rastros, quem é quem?
- Algoritmo húngaro (atribuição de custo mínimo)
- IoU como métrica de associação

**Por que antes dos modelos:** rastreadores modernos são, em essência, Kalman +
associação com um detector na frente. Entender essa base torna ByteTrack e
BoT-SORT legíveis em vez de mágicos.

> **Critério de domínio:** implementar um filtro de Kalman 1D do zero, em
> numpy, que siga um ponto ruidoso. Sem biblioteca pronta.

---

## Bloco 3 — Detecção e rastreamento

Agora sim, os modelos.

**O que estudar**

- Paradigma *tracking-by-detection*
- Família YOLO: como funciona a detecção em uma passagem
- ByteTrack — a ideia central: aproveitar detecções de baixa confiança
- BoT-SORT, UCMCTrack — compensação de movimento de câmera

**Papers**

- *In Pursuit of Many: A Review of Modern Multiple Object Tracking Systems*
  (arxiv 2209.04796) — leia este primeiro, dá o mapa do campo
- *UCMCTrack* (arxiv 2312.08952)

> **Critério de domínio:** rodar detecção + rastreamento sobre um vídeo seu e
> explicar, olhando o resultado, **por que** o ID trocou naquele momento
> específico. Diagnosticar é mais difícil que rodar.

---

## Bloco 4 — Métricas

**Estude métricas antes de tentar melhorar qualquer coisa.** Este é o conselho
que separa amador de pesquisador.

**O que estudar**

- MOTA — e por que ela é enganosa
- IDF1 — foco em identidade
- HOTA — por que foi criada, e o que ela equilibra

Entender *por que HOTA existe* já ensina qual é o problema difícil: não é
detectar pessoas, é **manter a identidade**.

> **Critério de domínio:** calcular HOTA e IDF1 sobre uma sessão sua anotada à
> mão, e explicar a diferença entre os dois números.

Quando este bloco terminar, você terá o **benchmark** — e a capacidade de medir
qualquer ideia futura. É o que transforma tentativa e erro em pesquisa.

---

## Bloco 5 — Re-identificação

O coração do problema multi-câmera.

**O que estudar**

- Metric learning: aprender um espaço onde "mesma pessoa" fica perto
- Triplet loss, contrastive loss
- Embeddings de aparência
- Por que roupa, iluminação e ângulo destroem tudo

**Datasets de referência:** DukeMTMC, Market-1501, MSMT17

---

## Bloco 6 — MTMC

Onde tudo se junta: restrições espaço-temporais + aparência.

**Benchmarks**

- **DukeMTMC** — 8 câmeras sincronizadas, 2 milhões de quadros anotados,
  2.000+ identidades. A pedra fundamental.
- **MMPTrack** — multi-câmera denso, ambientes internos
- **DivOTrack** — cenas abertas e diversas

**Surveys**

- *Multi-Camera Multi-Object Tracking: A Review of Current Trends and Future
  Advances* — ScienceDirect
- *Multi Camera Connected Vision System with Multi View Analytics: A
  Comprehensive Survey* (arxiv 2510.09731)
- *FusionTrack: End-to-End MOT in Arbitrary Multi-View Environment*
  (arxiv 2505.18727)

---

## Bloco 7 — World models

A fronteira de 2026. Deixe para quando tiver base para ler criticamente — antes
disso é consumo de notícia, não estudo.

**Contexto**

- World Labs (Fei-Fei Li) levantou US$ 1 bi no início de 2026
- AMI Labs (Yann LeCun) — seed de US$ 1,03 bi em março de 2026
- *A Functional Taxonomy of World Models* (World Labs, junho de 2026):
  **renderers** (como o mundo se parece), **simulators** (o que acontece
  depois), **planners** (o que fazer)

Seu projeto é percepção + estimação de estado alimentando um simulator leve.
**Não é renderer.** Saber disso poupa metade da literatura.

**Leituras de orientação**

- *From Words to Worlds: Spatial Intelligence is AI's Next Frontier* — Fei-Fei Li
- *The World Model and Spatial Intelligence Era* — Stanford HAI

---

## Como estudar

**Um caderno de laboratório.** Um arquivo em `docs/` por semana: o que tentou,
o que esperava, o que aconteceu, o que entendeu. Resultado negativo anotado vale
mais que resultado positivo esquecido.

**Reimplemente antes de importar.** Kalman, IoU, o húngaro — escreva uma vez,
mesmo mal, mesmo lento. Depois use a biblioteca. A intuição não vem de ler.

**Prefira poucos papers lidos a fundo.** Um paper entendido de verdade vale
vinte lidos por alto. Comece pelos surveys: eles dão o mapa antes do detalhe.

**Meça antes de melhorar.** Sem benchmark, "melhorou" é opinião.

---

## Ordem de execução, resumida

```
0. Python suficiente        →  manipular imagem sem laço
1. Geometria de câmeras     →  homografia funcionando        ← gêmeo digital nasce
2. Estimação de estado      →  Kalman escrito do zero
3. Detecção e rastreamento  →  diagnosticar troca de ID
4. Métricas                 →  HOTA calculada                ← benchmark pronto
5. Re-identificação         →  embeddings de aparência
6. MTMC                     →  múltiplas câmeras, um ID
7. World models             →  fronteira
```

Em paralelo, desde hoje: **coletar dados**. Ver `DATASET.md`.

O dado é a única coisa que não dá para recuperar depois.
