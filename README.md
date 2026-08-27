# SO Espacial

**Percepção espacial multi-câmera — o gêmeo digital de uma loja, em tempo real**

Visão computacional × geometria projetiva × varejo autônomo

Eduardo Lopes, 2026 — MIT, ver [`LICENSE`](LICENSE)

---

## O problema

Uma loja autônoma que identifica produtos por RFID sabe **o quê** saiu e
**quantos**. Não sabe **quem** pegou, de **qual prateleira**, nem o que foi
pego e devolvido antes de ir embora.

Câmeras responderiam isso. Mas uma câmera comum entrega uma imagem plana:
ao projetar o mundo em pixels, a profundidade se perde e não há como
recuperá-la — dois pontos na mesma linha de visada caem no mesmo pixel.

O caminho usual é comprar o sensor de volta: câmera de profundidade, LiDAR,
estéreo calibrado. Isso resolve a geometria e destrói o custo por loja.

## A solução

**Devolver a dimensão perdida com restrições, não com hardware.**

Uma projeção só é irreversível no caso geral. Com uma restrição conhecida,
ela volta a ter inversa — e as duas restrições necessárias já existem de graça
em qualquer loja:

| restrição | o que devolve |
|---|---|
| os pés estão **no chão** (um plano) | posição em metros, por homografia |
| a razão altura/horizonte é **invariante à distância** | altura da mão, por metrologia de vista única |

Nenhum sensor especial. Três webcams comuns e geometria.

E uma terceira decisão, que é a que faz o sistema funcionar de verdade:

> **A fusão não é média — é voto.**
>
> As câmeras não trocam coordenadas para tirar uma média. Cada uma publica um
> valor de um **vocabulário fechado**, e a decisão é discreta.

Isso importa porque a média herda o erro de todas as fontes, enquanto o voto
sobrevive ao erro da pior delas. Medido em 12/08: entre prateleiras há 40 cm de
vão e o erro de altura é de ±8 cm. Escolher **uma entre cinco** aguenta um ruído
que **medir em centímetros** não aguenta.

> Um bit sobrevive ao ruído que destrói um ângulo.

---

## Arquitetura

Estágios desacoplados. Cada um só conhece o anterior.

```
   3 CÂMERAS            PERCEPÇÃO              MUNDO               SAÍDA
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌─────────────┐
│  alto        │   │  detector    │   │  motor espacial  │   │ estado.json │
│  frontal     │──▶│  YOLO-pose   │──▶│  Kalman, zonas   │──▶│ eventos     │
│  lateral     │   │  MediaPipe   │   │  ação, prateleira│   │ cena 3D     │
└──────────────┘   └──────────────┘   └──────────────────┘   └─────────────┘
   fonte.py          visao/            espacial/ + acao/       gemeo/
   thread por        thread por        aritmética, sem I/O      serializável
   câmera            câmera
```

**Cada câmera tem um papel, e nenhuma faz tudo.** É o critério de projeto
central: em vez de uma câmera perfeita, três câmeras que erram de formas
diferentes.

| papel | responde | por que ela |
|---|---|---|
| **alto** (cenital) | posição no piso, rumo do corpo, estatura | é a única com homografia — vê o chão sem perspectiva enganosa |
| **frontal** | qual braço se move, a que altura a mão chega | vê o corpo de frente, onde o gesto acontece |
| **lateral** | o quanto o braço avança para a gôndola | separa *pegar* de *passar perto* |

Quando uma delas não enxerga, o campo chega `None` e **simplesmente não vota**.
Nada é inventado para preencher a lacuna — abster-se é um resultado de primeira
classe em todo o sistema.

### Três decisões que valem mais que as outras

**1. Vocabulário fechado em vez de coordenadas.**
Até 10/08 o sistema transmitia 17 juntas 3D e mandava desenhar exatamente
aquilo. O desenho herdava todo erro da reconstrução — e produziu um esqueleto
deitado no chão enquanto a pessoa andava em pé. Não havia como consertar o
desenho: ele mostrava fielmente dados ruins. Com vocabulário fechado, o
renderizador **anima um corpo que já sabe ser corpo**. Se "deitado" não está no
vocabulário, o boneco não consegue deitar. A classe inteira de defeito
desaparece por construção.

**2. Eventos são fatos, nunca ordens.**
`PESSOA_ENTROU_NA_ZONA` é um fato consumado com carimbo de tempo. Nenhum evento
manda ninguém fazer nada; quem quiser agir se inscreve. É isso que permite
gravar uma sessão e reproduzi-la depois, e mandar tudo para um painel sem que o
núcleo saiba que existe painel.

**3. Limiar temporal se declara em segundos, nunca em quadros.**
Com 3 quadros de confirmação a 14 fps, o limiar vale 0,21 s; a 30 fps, 0,1 s.
O mesmo código se comporta diferente conforme a máquina. Quadro não é unidade
de tempo.

---

## Pilha tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Visão | OpenCV, YOLO11-pose (Ultralytics), MediaPipe Pose Landmarker |
| Geometria | homografia por DLT, metrologia de vista única, SVD para registro |
| Rastreamento | filtro de Kalman 2D em metros, recostura de identidade |
| Fusão | por eixo e por mérito — cada vista responde o que enxerga melhor |
| Câmeras | USB (DirectShow/MJPG) e remotas por MJPEG sobre HTTP |
| Saída | JSON atômico, JSONL de eventos, cena 3D em OpenCV |
| Testes | pytest — 814 testes, todos sem hardware |

---

## Estrutura

```
SO-Espacial/
├─ src/
│   ├─ nucleo/       erros, log estruturado, métricas, voz
│   ├─ fluxo/        Quadro, buffer limitado, sincronizador
│   ├─ cameras/      fontes USB e remota, gerenciador, diagnóstico
│   ├─ visao/        detector YOLO, pose, trabalhador por câmera
│   ├─ espacial/     motor: pixel → metro, fusão, rastreio
│   ├─ acao/         O CORAÇÃO — vocabulário, classificador, prateleira
│   ├─ gemeo/        o estado do mundo, serializável
│   ├─ eventos/      fatos com carimbo de tempo
│   ├─ mundo/        reconstrução do ambiente, mapeamento
│   └─ app/          orquestrador
├─ percepcao/        chão, fusão de vistas, pose 3D
├─ estado/           Kalman, zonas, planta da loja, publicador
├─ visual/           só desenho — não sabe de câmera
├─ ferramentas/      13 utilitários de calibração e diagnóstico
├─ testes/           814 testes, 33 arquivos
├─ config/           câmeras, escala, rumo, assinaturas de prateleira
├─ loja/             plantas em JSON — loja nova, arquivo novo
└─ docs/caderno/     o diário de bordo: o que quebrou e por quê
```

**Duas regras organizam tudo isso.**

Um arquivo é biblioteca **ou** programa, nunca os dois. Biblioteca não tem
`main()`; programa não é importado. Até 08/08 o `mapa.py` era os dois, e mexer
num quebrava o outro.

E `dados/bruto` é **somente escrita**: nada é corrigido, limpo ou sobrescrito
ali. Todo processamento gera arquivo novo. Parece exagero até o dia em que você
acha um erro no processamento e precisa refazer tudo sem regravar nada.

---

## Estado atual

**MVP funcionando com três câmeras simultâneas em hardware real.**

O caminho crítico opera de ponta a ponta: três câmeras capturam em paralelo,
o detector roda por câmera em thread própria, a posição vira metros por
homografia, o Kalman mantém identidade, o classificador emite ação em
vocabulário fechado, e o estado sai como JSON a cada 200 ms.

### O que foi medido, não estimado

| | antes | depois | o que era |
|---|---|---|---|
| taxa do sistema | 3,4 fps | **14,4 fps** | a primeira inferência do YOLO custava 15,2 s e envenenava a média |
| câmera do alto | 1,0 fps | **15,0 fps** | uma câmera lenta ditava o ritmo de todas |
| detecções recusadas | 55% | **0,4%** | o filtro de plausibilidade não cabia nos dados e agora se abstém |
| inclinação do esqueleto | 42° | **0,0°** | o motor chamava a projeção errada; o estimador existia e não era usado |
| falhas de leitura | 300/s | **45/s** | laço de leitura girando sem pausa, queimando CPU |

Outros números de bancada: posição no chão com **2 a 5 cm** de erro,
**99,6%** de sobrevivência do rastro, κ = 5,19 calibrado com uma pessoa de
estatura conhecida e **6% de dispersão em 254 amostras**.

### O que ainda não funciona, dito com número

Ser honesto aqui vale mais que parecer pronto:

- **Identidade contínua falha em sessões longas.** Re-ID entre câmeras ainda
  não existe; a recostura é geométrica.
- **Duas das cinco prateleiras são confiáveis.** As outras três, não.
- **A calibração de cada espaço novo é manual** — quatro pontos no chão e uma
  pessoa de altura conhecida.
- **A perda evitada nunca foi medida em operação real.** É o argumento
  comercial mais forte do projeto e, justamente por isso, não entra em
  projeção nenhuma enquanto não for medido.

---

## Roteiro

Cada degrau funciona e demonstra sozinho. Nada depende de terminar tudo.

- [x] **0 · Aparato experimental** — captura sincronizada, dataset, como medir
- [x] **1 · Detecção** — pessoas em quadros isolados
- [x] **2 · Rastreamento** — identidade que persiste entre quadros
- [x] **3 · Zonas e permanência** — mapa de calor e tempo de permanência
- [x] **4 · Homografia** — vista de cima em tempo real; o gêmeo nasce aqui
- [x] **5 · Multi-câmera** — três vistas, fusão por eixo e por mérito
- [x] **6 · Ação em vocabulário fechado** — locomoção, postura, braços
- [~] **7 · Prateleira por evidência conjunta** — 2 de 5 confiáveis
- [ ] **8 · Re-identificação** — mesma pessoa depois de sair e voltar
- [ ] **9 · Fusão com RFID** — encontro com o sistema da loja autônoma

### Sobre o degrau 9

Os dois sistemas existem e **ainda não se falam**. O SO Espacial calcula de
qual prateleira a mão veio e guarda o resultado só para a tela; a loja em C#
tem um endpoint que espera duas fotos e um nome de produto.

São perguntas diferentes sobre o mesmo gesto, e ligá-las é a próxima grande
decisão de arquitetura — não um trabalho de encanamento.

---

## Como rodar

**Requisitos:** Python 3.11+, uma webcam. Três câmeras para o sistema completo.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

O `ultralytics` está comentado no `requirements.txt` porque puxa o PyTorch
junto — cerca de 2,5 GB. Descomente quando for usar detecção de verdade.

```powershell
python rodar.py                 # o sistema completo, com a cena 3D
python rodar.py --falsas        # sem hardware nenhum, com câmeras sintéticas
python apresentar.py            # o modo de demonstração
pytest                          # 814 testes, nenhum precisa de câmera
```

### Calibrar um espaço novo

```powershell
python ferramentas/abrir_camera.py        # descobrir índices e resoluções
python ferramentas/mapear.py              # 4 pontos no chão -> homografia
python ferramentas/calibrar_escala.py     # uma pessoa de altura conhecida
python ferramentas/aprender_prateleiras.py  # a assinatura de cada prateleira
```

A configuração fica em `config/`. Nenhum caminho está cravado no código.

---

## Documentação

| Onde | O quê |
|---|---|
| [`docs/caderno/`](docs/caderno/) | **o diário de bordo** — um arquivo por dia, com o que quebrou, o que foi medido e por que a decisão mudou |
| [`docs/ARQUITETURA-v3-ACAO.md`](docs/ARQUITETURA-v3-ACAO.md) | a proposta do vocabulário fechado, escrita **antes** de implementar |
| [`docs/ARQUITETURA-v2.md`](docs/ARQUITETURA-v2.md) | a arquitetura anterior, mantida para comparação |
| [`docs/AUDITORIA-2026-08-09.md`](docs/AUDITORIA-2026-08-09.md) | auditoria do próprio código, com os defeitos encontrados |
| [`docs/DATASET.md`](docs/DATASET.md) | esquema de dados e protocolo de captura |
| [`docs/PLANO-DE-ESTUDO.md`](docs/PLANO-DE-ESTUDO.md) | o que estudar, em que ordem |
| [`docs/ONDE-PARAMOS.md`](docs/ONDE-PARAMOS.md) | estado da última sessão de trabalho |

Os documentos de negócio — dossiês, CANVAS, deck e o site — vivem no
repositório de apresentação, não aqui. Código e material de venda mudam por
motivos diferentes e em ritmos diferentes; misturá-los faz o histórico de um
poluir o do outro.

O caderno é a parte mais útil do repositório para quem quiser entender **por
que** o sistema é assim. Os comentários no código seguem a mesma regra: eles
explicam a decisão e o que aconteceu quando ela era outra, não o que a linha faz.

---

## Nome técnico do campo

Para achar literatura, os termos certos:

- **MTMC** — Multi-Target Multi-Camera Tracking
- **MOT** — Multiple Object Tracking
- **Re-ID** — Person Re-Identification
- **Single-view metrology** — a altura da mão sai daqui
- **Spatial Intelligence / World Models** — o guarda-chuva de 2026

"Digital twin de loja" traz marketing. "MTMC tracking" traz ciência.

---

## Privacidade

O sistema opera sobre **posição, postura e altura da mão** — nunca sobre
reconhecimento facial. A identidade é um número de rastro que morre quando a
pessoa sai. Isso não é um detalhe de implementação: muda o enquadramento na
LGPD, e foi decisão de projeto desde o primeiro dia.

---

## Licença e dependências

O código deste repositório está sob **MIT** — veja [`LICENSE`](LICENSE). Use,
copie, modifique e venda; basta manter o aviso de copyright.

Isso cobre o que está aqui dentro. As bibliotecas instaladas em tempo de
execução têm licenças próprias — a lista completa está em
[`NOTICE.md`](NOTICE.md). Uma delas exige atenção.

### O caso do Ultralytics

`src/visao/detector.py` usa **YOLO11-pose**, da biblioteca `ultralytics`,
distribuída sob **AGPL-3.0**. Ela não está versionada aqui: o `pip` a baixa a
partir do `requirements.txt`.

A AGPL é uma licença *copyleft de rede*. Simplificando o que ela exige:

> quem **distribui** um software que incorpora código AGPL — ou o oferece como
> serviço pela rede — precisa disponibilizar o código-fonte do conjunto,
> também sob AGPL.

Na prática, por cenário:

| você quer | a AGPL atrapalha? |
|---|---|
| estudar, pesquisar, rodar na sua máquina | não |
| publicar o fonte aberto, como aqui | não |
| vender um produto fechado que embarque isto | **sim** |
| oferecer como serviço em nuvem, sem abrir o fonte | **sim** |

As duas últimas linhas são o caso comercial da Smart Store, e há três saídas
conhecidas:

1. **Licença comercial da Ultralytics.** Vendida justamente para quem não pode
   abrir o fonte. É a saída direta, e custa dinheiro.
2. **Trocar o detector.** O projeto já usa MediaPipe (Apache-2.0) para pose 3D;
   usá-lo também para detecção eliminaria a dependência AGPL. Custaria precisão
   e trabalho de medição — quanto, não foi medido.
3. **Abrir o produto.** Compatível com a AGPL, incompatível com a maioria dos
   modelos de assinatura.

Nada disso impede publicar este repositório. Fica registrado para a decisão não
ser tomada por esquecimento lá na frente.

*Isto é a leitura de um desenvolvedor sobre um texto jurídico, não parecer de
advogado. Antes de fechar contrato apoiado nesta seção, consulte alguém
habilitado.*

---

## Projetos relacionados

| Repositório | O quê |
|---|---|
| [LOJA-AUT-NOMA-PRO](https://github.com/Duduedulopes/LOJA-AUT-NOMA-PRO) | a loja autônoma em .NET 8 — API, apps Blazor, firmware ESP32 |
| **este** | a percepção espacial em Python |

Site do projeto: **[smart-store.contato-dudulopes.workers.dev](https://smart-store.contato-dudulopes.workers.dev)**
