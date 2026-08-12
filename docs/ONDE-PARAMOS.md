# Onde paramos — 12/08/2026 (fim do dia)

> Este arquivo existe para que o PROJETO seja suficiente para retomar o
> trabalho. Nenhuma informação necessária deve morar fora do repositório —
> nem em histórico de conversa, nem na cabeça de ninguém.
>
> Se você está lendo isto sem contexto nenhum, comece aqui e depois vá para
> `docs/caderno/`, que tem o dia a dia com os números medidos.

---

## O que o sistema faz hoje

Três câmeras, em tempo real, produzem a **descrição da ação** de uma pessoa
num vocabulário fechado — e a **altura da mão em metros**, que é o número que
decide de qual prateleira um produto foi retirado.

```
alto      C920 no teto      YOLO11n-pose + homografia  ->  ONDE, estatura, rumo
frontal   webcam do notebook   MediaPipe               ->  braços, altura da mão
lateral   tablet (IP Webcam)   MediaPipe               ->  reserva quando a frontal perde
```

    Nenhuma delas capta 100% de tudo. As 3 existem ao mesmo tempo para uma
    complementar a outra.                                  — Eduardo, 11/08

Essa frase reorganizou a arquitetura inteira e continua sendo o critério de
projeto: cada câmera responde o que enxerga, e o relatório sempre diz **qual
delas respondeu**.

## O estado da medição, em números

Gabarito físico: estante de aço, 5 prateleiras medidas com trena
(`loja/estante.json`). É o único jeito de verificar a altura da mão contra a
realidade.

Última medição, 11/08 18:06 (`dados/confer/altura_2026-08-11_180622.json`):

| prateleira | postura | lido | erro |
|---|---|---|---|
| 1,90 m | em pé, braço para cima | 1,89 | **−0,01** |
| 1,35 m | em pé, braço à frente | 1,12 | −0,23 |
| 0,95 m | leve inclinação | 1,10 | +0,15 |
| 0,55 m | dobrado | 0,97 | +0,42 |
| 0,15 m | agachado fundo | 0,90 | +0,75 |

**Em pé e ereto, um centímetro.** O erro cresce com o quanto o corpo sai da
vertical — porque a âncora do quadril era a constante de quem está de pé.

Isso foi consertado depois dessa corrida e **ainda não foi medido**. É o
primeiro passo pendente.

Outros números estabelecidos:

- estatura medida: **1,83 / 1,84 m** contra 1,80 real (estável entre corridas)
- fator de escala: **5,25**, dispersão 5%, 148 amostras (`config/escala.json`)
- homografia: 0,43 m medidos contra ~0,50 m caminhados
- área útil: 140 × 140 cm
- vãos entre prateleiras: 40 cm

## O pendente, em ordem

### 1. Altura do quadril pela câmera do alto  ← COMEÇA AQUI

**O problema.** Agachado, nenhuma vista de pose enxerga tornozelo (frontal 0%,
lateral 0% no laudo de 12/08 manhã), e sem tornozelo não há âncora. Hoje o
sistema recusa responder — o que é honesto, e insuficiente.

    A câmera superior consegue captar SIM a primeira prateleira. O que não
    está acontecendo é as três trabalharem juntas.        — Eduardo, 12/08

Ele está certo. A câmera do alto tem tudo o que falta, e a fórmula já está
escrita no projeto:

```
altura_do_quadril = fator × (v_pé − v_quadril) / (v_pé − v_horizonte)
```

É a **mesma** relação de metrologia de vista única que `percepcao/chao.py` já
usa para a estatura (`razao = altura_px / (v_pé − v_horizonte)`), aplicada a
outro par de pontos. Mesma constante `fator = 5,25`, nada novo a calibrar.

Peças que já existem:

- `src/espacial/motor.py`, campo `_ombros[tid] = (juntas_2d, conf_2d)` — guarda
  os 17 keypoints 2D da câmera do alto por rastro (o nome ficou pequeno; ele
  guarda o esqueleto inteiro, não só ombros)
- `percepcao/chao.py`, `FiltroDePlausibilidade.v_horizonte(u)` — a linha do
  horizonte, tirada da terceira linha da homografia
- `src/acao/escala.py` — o fator calibrado

**Risco declarado:** visto quase de cima, `v_pé − v_quadril` são poucos pixels.
Pode sair ruidoso. Mas é medição, não suposição — e o boletim dirá com número.

### 2. Medir o gabarito de novo e declarar a faixa útil

Com a âncora funcionando agachado, rodar `ferramentas/conferir_altura.py`
inteiro. O que sair é a especificação do arranjo:

    "o sistema mede a altura da mão entre X e Y metros"

Limite medido vira requisito de projeto. Limite ignorado vira defeito em
produção.

### 3. Limpeza: duas árvores paralelas

O projeto tem `captura/` + `percepcao/` (antiga) e `src/` (nova), e a migração
ficou pela metade. Quem chega hoje não sabe qual é a viva. **2.778 linhas
mortas**, auditadas em 12/08:

```
percepcao/gemeo3d.py        345   rodar.py diz no cabeçalho que o substituiu
percepcao/gemeo_multi.py    317   idem
percepcao/mapa.py           207   rodar.py --sem-pose faz o mesmo
visual/cena2d.py           ~150   só mapa.py usava
captura/dispositivos.py    ~200   duplicata de src/cameras/dispositivos.py
captura/fonte.py           ~250   duplicata de src/cameras/fonte.py
captura/identificar.py     ~180   idem
captura/diagnostico.py      237   conferir.py --so-cameras faz melhor
captura/reparar.py          153   absorvido por src/cameras/usb.py
experimentos/ (3 arquivos)  624   rascunho de estudo
```

Comando:

```powershell
git rm -r percepcao/gemeo3d.py percepcao/gemeo_multi.py percepcao/mapa.py visual/cena2d.py captura/dispositivos.py captura/fonte.py captura/identificar.py captura/diagnostico.py captura/reparar.py experimentos
python -m pytest testes/ -q
```

**Fica:** `calibracao/homografia.py` (sem ela não há metro), `visual/cena3d.py`
(o `rodar.py` usa), e tudo em `src/`, `ferramentas/`, `testes/`.

**Decisão do Eduardo, ainda aberta:**

- `calibracao/intrinseca.py` (208) — serve para TRIANGULAR pontos fora do chão.
  A arquitetura v3 decidiu não triangular. Só fica se essa porta pode reabrir.
- `captura/gravar.py` (224) — gravador de sessões de `docs/DATASET.md`.

### 4. O mundo declarado em dados

O ambiente virtual que o Eduardo propôs em 12/08 **já existe** em
`estado/planta.py` + `loja/bancada.json`: chão em metros, móveis com posição e
dimensão, zonas. Formato AAS (identidade + submodelos), serializável.

O problema é que `bancada.json` descreve uma **loja fictícia** — duas gôndolas
e um checkout que não existem no quarto — e `estante.json` descreve a estante
real mas mora fora da planta e **não tem posição no chão**.

Falta escrever `loja/quarto.json` com as medidas reais, e dar ao `Movel` um
campo `prateleiras`. Umas dez linhas de Python; o resto já funciona.

**Correção importante, registrada para não se repetir:** a pegada da estante
NÃO ajuda a escolher o nível — as cinco prateleiras compartilham a mesma
pegada. O que a posição no chão resolve é o **evento** (a mão entrou na
estante, e não um braço levantado em outro canto). O nível continua sendo
pergunta de altura, e por isso o passo 1 vem antes.

### 5. Três sistemas — DECIDIDO EM 12/08: Java

Eduardo interliga Python (gêmeo), C# (LOJA AUTÔNOMA PRO) e um ambiente virtual
em **Java**.

**O motivo é de recurso, medido, e é o que torna a decisão defensável:** Unity
Hub mais um editor passa de 10 GB. A máquina tinha 12,8 GB livres depois da
faxina de 12/08, e precisa rodar três câmeras em tempo real. Java resolve em
centenas de megabytes, e o JDK 24 já estava instalado.

    Escolher a ferramenta pelo recurso que ela cobra, e não pela que parece
    mais moderna, é a mesma disciplina que o resto do projeto usa com número.

Registrado também o que NÃO é motivo: "Java é bom com objetos" não distingue
nada — Python e C# também são. Se alguém perguntar na banca, a resposta é o
orçamento de disco e a independência do runtime, não o paradigma.

A regra que mantém isso são: **o mundo é declarado em dados, e nenhum dos três
é dono dele.**

```
mundo.json (metros)
   ├─> Python   onde a mão entrou
   ├─> C#       qual produto está ali
   └─> Unity/Java   desenhar
```

Ninguém guarda a medida da estante em código. Mede-se uma vez, no arquivo.

Ponderação registrada: "Java é bom com objetos" não distingue nada — Python e
C# também são. Se o objetivo é mostrar domínio de três ecossistemas para uma
banca, é um motivo legítimo, mas não é um motivo de engenharia. Alternativa
não considerada até então: **Unity é C#**, que o Eduardo já domina, e daria uma
simulação 3D melhor que Java escrito à mão.

## COMECE AQUI — os quatro que a tela revelou em 12/08

`apresentar.py` rodou pela primeira vez com hardware às 17:52 e mostrou, em
uma linha só, quatro defeitos que nenhuma ferramenta anterior tornava
visíveis. Em ordem de dano à demonstração:

### 1. Palpite de prateleira sem leitura de braço  ← COMEÇA AQUI

```
bracos   esq ? dir ?      sem fonte          e mesmo assim:  P4
```

Os dois braços em `DESCONHECIDO`, nenhuma câmera reivindicando a leitura, e o
classificador opinou assim mesmo. O braço é a evidência principal — é ele que
diz de qual altura a mão veio.

    o esqueleto ainda levanta os bracos sozinho ou nao levanta tudo
                                                    — Eduardo, 12/08

Onde mexer: `motor._palpitar()` alimenta `ClassificadorDePrateleira.observar()`
a cada quadro. Ele precisa recusar o quadro quando `lado_que_alcanca(leitura)`
não devolve lado. Ver `src/acao/prateleira.py` e `src/acao/corpo.py`.

### 2. `TRACK_LOST`: três ids para uma pessoa

```
  #2  {'p1': 3, 'p3': 6, 'p4': 3, 'p5': 5}
  #3  {'p1': 1}
  #4  {'p2': 2, 'p3': 1, 'p4': 2}
```

Uma pessoa só na sala. Cada troca de id joga fora a estatura já fechada
(`EscalaVertical._fechadas`) e reabre a contagem de unidades do zero — é o que
inflou o total para 23. Três vezes por sessão desde 11/08.

### 3. Postura `desconhecida` com a pessoa em pé

Em pé, parado, na frente das três câmeras, e a postura não foi classificada.
`verticalidade_da_coxa` é o sinal; ver por que ele não respondeu.

### 4. 4,7 fps na tela (era 9,7 no `rodar.py`)

Compor 1600×900 mais três miniaturas a cada quadro é caro. Candidatos: compor
a faixa de câmeras a cada N quadros, reduzir a resolução da cena, e o ONNX que
está pendente desde sempre.

---

## Aberto, medido, não consertado

- pico de 0,50 m/s durante `parado` (salto de rastreio)
- frontal contra a janela em parte do dia
- brilho das duas câmeras de pose em ~45 de 255 sem luz acesa; com luz, a
  frontal foi a 67,7 e os pulsos de 13% para 61%
- ONNX continua sendo o único ganho de velocidade que sobra

## Como rodar

```powershell
python ferramentas/achar_ip.py --gravar     # o IP do tablet muda todo dia
python ferramentas/conferir.py --so-cameras # laudo: brilho e enquadramento
python ferramentas/conferir_altura.py       # gabarito contra a estante
python ferramentas/conferir.py              # roteiro de ações completo
python -m pytest testes/ -q                 # 491 testes

python rodar.py                             # bancada: 4 janelas, painel, 9.7 fps
python apresentar.py                        # A TELA: uma janela, QUEM/O QUE/QUANTAS
python apresentar.py --tela-cheia           # o modo da banca (ESC ou q saem)
python apresentar.py --falsas               # ensaiar sem hardware
```

Git: `del .git\index.lock, .git\HEAD.lock -ErrorAction SilentlyContinue` antes
do commit — o sandbox deixa locks para trás.

## As lições que valem mais que o código

- Cada câmera responde o que enxerga. Nenhuma precisa enxergar tudo.
- Uma referência medida no mesmo espaço do erro herda o erro. Independência
  vale mais que precisão nominal.
- Erro que cresce com uma variável é erro daquela variável, não ruído.
- Quando existe um teste geométrico para a pergunta, ele ganha do palpite do
  modelo — ainda mais quando o modelo está seguro.
- Um bloco `except` que não registra nada não trata o erro: apaga o erro.
- Métrica agregada esconde exatamente o erro que o uso real sofre um a um.
- Um instrumento errado é útil quando o erro é estável. Um instrumento
  instável não serve nem quando acerta a média.
- Limite medido vira requisito de projeto; limite ignorado vira defeito em
  produção.
- Aumentar a amostra de uma hipótese falsa não a torna verdadeira: torna o
  erro confiante.
- Um movimento que não foi medido não é informação: é ruído com forma de perna.
- Marcar tudo é o mesmo que não marcar nada. O que dá forma não é a quantidade
  de pontos: é QUAIS deles ganham peso.
- A mediana de uma janela deslizante acompanha o sinal. Quando o que se mede é
  constante, acompanhar é o defeito.
- O mesmo número serve a dois donos com exigências opostas. Duplicar o filtro é
  mais barato que escolher um dono.
- Um evento só pode ser contado depois de terminar. Contar no começo é contar
  intenção, e intenção se desfaz.
- Um sistema que não mostra o que usa está pedindo para não ser acreditado.
- Copiar a estrutura de um laço sem copiar o que ele faz nos casos que não são
  o caso feliz é como copiar só a parte do código que se entende.
- O padrão é para o dia comum. Um modo que dificulta sair não pode ser o que
  acontece quando você não pede nada.
- Uma coluna que explica respostas inexistentes ensina a plateia a ignorá-la.
