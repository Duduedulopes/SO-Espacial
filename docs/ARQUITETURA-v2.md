# Arquitetura v2 — SO Espacial

Fase 2. Contratos e responsabilidades. **Nenhuma implementação.**

Segue a auditoria de 09/08. Preserva ~60% do código existente e reestrutura
captura e orquestração.

---

## 0. Princípios

Cinco regras que decidem as dúvidas de projeto. Quando algo estiver ambíguo
adiante, a resposta sai daqui.

**1. Falhar visível é melhor que degradar em silêncio.**
O bug mais caro dos últimos dias foi um *fallback* que trocava a câmera sem
avisar. Nenhum componente deve substituir um recurso por outro parecido.

**2. O que não é medido não existe.**
Todo componente expõe contadores. `frames_dropped` sem contador é fé.

**3. Uma responsabilidade por módulo; biblioteca não tem `main()`.**
Já adotada em 08/08 com `chao.py`. Agora vale para tudo.

**4. O tempo real prefere o presente.**
Entre processar um quadro velho e descartá-lo, descartar. Sempre.

**5. Estado do mundo vive num lugar só.**
O `DigitalTwin` é o dono da verdade. Ninguém mais guarda posição de pessoa.

---

## 1. Visão geral

```
┌── FONTES ─────────────────────────────────────────┐
│  UsbCameraSource(C920)   thread + fila(2)         │
│  UsbCameraSource(VGA)    thread + fila(2)         │
│  RemoteCameraSource(iPhone, MJPEG)  thread+fila(2)│
└───────────────┬───────────────────────────────────┘
                │ Frame
                ▼
       GerenciadorDeCameras      ciclo de vida, reconexão, métricas
                │
                ▼
          Sincronizador          agrupa por timestamp → Instante
                │
                ▼
┌── VisionEngine ───────────────────────────────────┐
│  DetectorDePessoas   (papel: alto)                │
│  EstimadorDePose     (papel: frontal)             │
│  EstimadorDePose     (papel: lateral)             │
└───────────────┬───────────────────────────────────┘
                │ Observacao[]
                ▼
        SpatialEngine            chão, Kalman, fusão de eixos
                │ EstadoDePessoa[]
                ▼
         DigitalTwin             dono da verdade
                │
      ┌─────────┼──────────┬─────────────┐
      ▼         ▼          ▼             ▼
 EventEngine  Publicador  Cena2D/3D   Dashboard
```

---

## 2. Modelo de dados

Estruturas imutáveis, sem lógica. São o vocabulário do sistema.

```
Frame
    camera_id      str      identidade estável (o NOME do dispositivo)
    papel          str      "alto" | "frontal" | "lateral"
    seq            int      contador monotônico por câmera
    t_mono         float    relógio monotônico, para intervalos
    t_wall         str      ISO 8601 com fuso, para casar com o mundo
    imagem         ndarray  BGR
    largura, altura int

Instante                    quadros considerados simultâneos
    t_ref          float    timestamp de referência
    quadros        dict[papel -> Frame]
    defasagem_ms   float    maior diferença entre os quadros do grupo

Observacao                  o que UMA câmera viu de UMA pessoa
    camera_id, papel
    t_mono
    caixa          (x1,y1,x2,y2)  pixels
    id_externo     int      id do rastreador daquela câmera
    juntas_2d      ndarray | None   (17,2) pixels
    juntas_3d      ndarray | None   (17,3) metros, relativo ao quadril
    confianca      float

EstadoDePessoa              o que o SISTEMA concluiu
    id             int      identidade global, sobrevive a sumiços
    x, y           float    metros no chão
    vx, vy         float    m/s
    incerteza      float    metros
    rumo           float    radianos
    esqueleto      ndarray | None   (17,3) metros, no mundo
    prevendo       int      quadros sem medição
    percorrido     float    metros acumulados
    visto_por      set[str] papéis que contribuíram
    t_mono         float

Evento
    tipo           str      ver §8
    t_wall         str
    dados          dict
```

**Por que `camera_id` é o nome e não o índice:** índices reordenaram entre duas
execuções seguidas em 08/08. Nome é estável; o índice é detalhe de acesso,
resolvido dentro da fonte.

---

## 3. Fontes de vídeo

### 3.1 Contrato

```
FonteDeVideo                          (abstrata)
    # identidade
    id            str                 nome do dispositivo ou URL
    papel         str
    tipo          "usb" | "remota" | "arquivo"

    # configuração
    largura, altura, fps_alvo
    exposicao     float | None        None = automática

    # estado observável
    estado        DESCONECTADA | CONECTANDO | ONLINE | DEGRADADA | FALHA
    ultimo_erro   str | None

    # ciclo de vida
    iniciar()                         não bloqueia; sobe a thread
    parar()
    reconectar()                      chamado pelo gerenciador

    # consumo
    ler()         -> Frame | None     o mais recente; None se vazia
    disponivel()  -> bool

    # métricas
    metricas()    -> Metricas
```

```
Metricas
    recebidos, descartados, falhas_leitura, reconexoes
    fps_medido, latencia_ms, brilho_medio
    ultimo_quadro_em    float
```

### 3.2 Máquina de estados

```
DESCONECTADA ──iniciar()──► CONECTANDO
                                │
                  abriu e entregou quadro
                                ▼
                             ONLINE ◄──────┐
                                │          │
              sem quadro > 2 s  │          │ quadro voltou
                                ▼          │
                            DEGRADADA ─────┘
                                │
              sem quadro > 10 s │
                                ▼
                              FALHA
                                │
                    espera com recuo exponencial
                                │
                                ▼
                           CONECTANDO
```

**DEGRADADA existe** porque hoje a fonte devolve o último quadro para sempre
quando a câmera cai — o sistema processa uma imagem congelada sem perceber.
Pior que falhar: mente. Em DEGRADADA, `ler()` devolve `None`.

**Recuo exponencial:** 1 s, 2 s, 4 s, 8 s, máximo 30 s. Evita martelar um
dispositivo ausente.

### 3.3 `UsbCameraSource`

Resolve nome → índice via `captura/dispositivos.py`, **sempre em DirectShow**.

Absorve o `reparar.py` como método `recuperar()`, disparado ao entrar em FALHA:
aquecimento antes de julgar brilho, exposição automática, propriedades ao meio
da escala, varredura manual. Já escrito e testado — muda de lugar, não de
conteúdo.

**Não há fallback de backend.** A enumeração por nome vem do DirectShow; abrir
por MSMF aponta para outro dispositivo. Falha em DSHOW é falha, e é reportada.

### 3.4 `RemoteCameraSource`

Fonte por URL. `VideoCapture` lê MJPEG/HTTP e RTSP nativamente.

Diferenças que justificam a classe separada:

| | USB | Remota |
|---|---|---|
| identidade | nome do dispositivo | URL |
| queda | rara, exige recuperação de driver | comum, exige reconexão de rede |
| latência | ~1 quadro | variável, medida por quadro |
| recuperação | `recuperar()` no driver | reabrir a conexão |

**Escolha do protocolo — MJPEG sobre HTTP para começar.**

*Avaliadas:*

| opção | latência | esforço | por que não / por que sim |
|---|---|---|---|
| Iriun (driver virtual) | baixa | zero | **rejeitada:** esconde que a fonte é remota; sem reconexão; ocupa um índice e já trocou identidades |
| MJPEG/HTTP | 200-500 ms | trivial — `VideoCapture(url)` | **escolhida:** nativa, explicitamente remota, reconectável |
| RTSP | 0,5-2 s | baixo | mais latência que MJPEG em LAN; vale se a rede piorar |
| WebRTC via MediaMTX | 100-500 ms | alto — servidor intermediário | melhor latência; adiar até a latência doer |

*Impacto:* MJPEG consome mais banda de rede que RTSP, mas em Wi-Fi local isso
não é gargalo. CPU: decodificação JPEG por quadro, ~3-5 ms a 720p.

*Expansão:* trocar o protocolo mexe só nesta classe.

### 3.5 `GerenciadorDeCameras`

```
GerenciadorDeCameras
    registrar(fonte)
    iniciar_todas() / parar_todas()
    fontes()          -> list[FonteDeVideo]
    por_papel(papel)  -> FonteDeVideo | None
    supervisionar()   chamado periodicamente: reconecta o que caiu
    metricas()        -> dict[camera_id -> Metricas]
```

**Uma câmera que cai não derruba o sistema:** o gerenciador remove a fonte do
conjunto ativo, emite `CAMERA_DISCONNECTED`, e o resto opera com o que sobrou.

---

## 4. Buffer e sincronização

### 4.1 `FrameBuffer`

Fila limitada, uma por fonte, com **descarte do mais antigo**.

`maxlen=2`, não 1: com 1, produtor e consumidor competem pelo mesmo slot e a
taxa efetiva cai. Com 2, há folga sem acumular atraso perceptível.

Cada descarte incrementa `descartados` — é assim que `frames_dropped` deixa de
ser fé.

### 4.2 `Sincronizador`

```
Sincronizador(tolerancia_ms=120)
    montar() -> Instante | None
```

Pega o quadro mais recente de cada fonte, usa o **mais antigo** como
referência, e inclui os que estiverem dentro da tolerância. Reporta
`defasagem_ms` no `Instante`.

**Por que 120 ms:** a 1,4 m/s isso é 17 cm de erro no pior caso. Aceitável para
postura, que é o que a fusão usa. A **posição** vem só da câmera do alto, que é
síncrona consigo mesma — o erro de sincronia não a afeta.

**O que o sincronizador não faz:** esperar. Se a lateral está atrasada, o
`Instante` sai sem ela e a fusão trabalha com uma vista. Esperar violaria o
princípio 4.

---

## 5. VisionEngine

```
Trabalhador                           (abstrato)
    papeis_aceitos   list[str]
    processar(frame) -> list[Observacao]
    metricas()       -> {ms_medio, quadros, deteccoes}

DetectorDePessoas(Trabalhador)        YOLO11 + ByteTrack
EstimadorDePose(Trabalhador)          MediaPipe Tasks
```

```
VisionEngine
    registrar(papel, trabalhador)
    processar(instante) -> list[Observacao]
    metricas()
```

**Threads por trabalhador**, cada um com fila de entrada de tamanho 1. É o que
resolve o consumo serializado medido na auditoria (84+26+33 = 143 ms em
sequência). Em paralelo, o custo vira o do mais lento — cerca de 84 ms.

*Alternativa avaliada:* processos separados. Isolamento melhor, mas exige
serializar quadros de 2,7 MB. Revisar se passar de 6 câmeras.

**Frame skipping por papel:** cada trabalhador declara `a_cada_n_quadros`. A
detecção pode rodar a cada 2 quadros enquanto a pose roda em todos, ou o
inverso. Configurável, não fixo no código.

---

## 6. SpatialEngine

Reaproveita **integralmente** o que já está medido:

| peça | origem | validação |
|---|---|---|
| `carregar_homografia`, `para_metros` | `chao.py` | trena: 2 mm e 0 mm |
| `EstimadorDePe` | `chao.py` | saltos >50 cm: 5 → 0 |
| `FiltroDeTornozelo` | `chao.py` | 0 tornozelos em 79 quadros de mobília |
| `FiltroDePlausibilidade` | `chao.py` | rejeita cadeira e poste, aceita criança |
| `Kalman2D`, recostura | `rastreio.py` | testado a 30 e 4 fps |
| `fundir`, `para_o_mundo` | `fusao.py` | 33 cm → 1,3 cm em simulação |

```
SpatialEngine
    atualizar(observacoes, dt) -> list[EstadoDePessoa]
```

**Ordem interna, e ela importa:**

```
1. filtro de plausibilidade   (grátis — antes de gastar 30 ms de pose)
2. ponto do pé                (tornozelo, ou caixa corrigida pelo viés)
3. homografia                 pixel → metros
4. filtro de tornozelo        o rastro já provou ser gente?
5. Kalman + recostura         identidade que sobrevive a sumiços
6. fusão de eixos             frontal dá largura/altura, lateral dá profundidade
7. ancoragem no chão          rumo pela velocidade
```

**Associação entre vistas — limitação declarada.** Com uma pessoa, o único
fusor recebe tudo. Com duas ou mais, o sistema não sabe qual pessoa da frontal
corresponde a qual do alto. Resolver exige re-identificação por aparência.
**Fica explícito no código e no `DigitalTwin`**, que marcará
`associacao_confiavel=False` quando houver mais de uma pessoa.

---

## 7. DigitalTwin

Dono único da verdade sobre o ambiente.

```
DigitalTwin
    ambiente     Ambiente      da planta JSON
    cameras      dict[id -> EstadoDeCamera]
    pessoas      dict[id -> EstadoDePessoa]
    zonas        list[Zona]
    calor        MapaDeCalor
    t_wall

    atualizar(estados, metricas_cameras, dt)
    instantaneo() -> dict          serializável, para JSON/WebSocket
    assinar(callback)              notifica mudanças
```

**Perda e reencontro:** uma pessoa não some ao deixar de ser detectada. Passa a
`prevendo`, com incerteza crescendo, por até `max_coasting_s` (3 s por padrão).
Só então é removida, com evento `TRACK_LOST`.

**Objetos, equipamentos e sensores IoT** entram como `DigitalObject` no mesmo
dicionário, com `tipo` distinguindo. É o que permite acrescentar RFID e
sensores sem mudar a estrutura — cada um vira uma fonte de observações.

---

## 8. EventEngine

```
EventEngine
    emitir(tipo, dados)
    assinar(tipo | "*", callback)
    historico(n) -> list[Evento]
```

**Tipos previstos:**

```
CAMERA_CONNECTED       CAMERA_DISCONNECTED    CAMERA_DEGRADED
CAMERA_ERROR           CAMERA_RECONNECTED
TRACK_STARTED          TRACK_LOST             TRACK_REIDENTIFIED
PERSON_ENTERED_ZONE    PERSON_LEFT_ZONE
OBJECT_DETECTED        OBJECT_MOVED
SYSTEM_DEGRADED        FRAME_DROPPED_BURST
```

Eventos são **fato consumado**, no passado, com timestamp. Não são comandos.
Essa distinção é o que permite gravá-los, reproduzi-los e, adiante, disparar
automações.

---

## 9. Observabilidade

**Logging** com a `logging` da stdlib, formato JSON por linha, dois destinos:
terminal legível e `dados/logs/sistema.jsonl`.

Cada registro: `t`, `nivel`, `componente`, `mensagem`, `dados`.

**Taxonomia de erros** — hierarquia própria, para que o tratamento seja
específico:

```
ErroDoSistema
├── ErroDeCamera
│   ├── CameraNaoEncontrada       nome não está presente
│   ├── CameraNaoAbriu            presente mas recusou abrir
│   ├── CameraSemImagem           abriu, não entrega quadro
│   └── CameraImagemInvalida      entrega, mas preta ou corrompida
├── ErroDeStream
│   ├── ConexaoPerdida
│   └── TempoEsgotado
├── ErroDeVisao
│   ├── ModeloIndisponivel
│   └── FalhaNaInferencia
└── ErroDeCalibracao
    ├── HomografiaAusente
    └── ResolucaoIncompativel
```

**Métricas** coletadas continuamente: fps por fonte e por trabalhador, latência
fim a fim, quadros recebidos/descartados, pessoas rastreadas, uso de CPU e
memória.

---

## 10. Configuração

Um arquivo, `config/sistema.json`, com tudo que hoje está espalhado em
argumentos de linha de comando:

```
cameras[]        id (nome ou URL), papel, tipo, resolução, fps, exposição
visao            modelo, imgsz, confiança, a_cada_n por papel
espacial         homografia, ruído de processo, coasting, tolerâncias
twin             planta, meia-vida do calor
sistema          nível de log, destino, taxa de publicação
```

Argumentos de linha de comando **sobrescrevem** o arquivo, nunca o substituem.

---

## 11. Estrutura de diretórios

```
src/
  nucleo/        Configuracao, Log, Metricas, Erros
  cameras/       FonteDeVideo, UsbCameraSource, RemoteCameraSource,
                 GerenciadorDeCameras, dispositivos
  fluxo/         Frame, FrameBuffer, Sincronizador, Instante
  visao/         VisionEngine, DetectorDePessoas, EstimadorDePose
  espacial/      SpatialEngine, chao, rastreio, fusao, pose3d
  gemeo/         DigitalTwin, Ambiente, Zona, DigitalObject, ocupacao
  eventos/       EventEngine, TiposDeEvento
  saidas/        Publicador, Cena2D, Cena3D
  app/           orquestrador — apenas monta e conecta
ferramentas/     identificar, diagnostico, calibrar, gravar
testes/
```

### Mapa de migração

| hoje | vai para | ação |
|---|---|---|
| `percepcao/chao.py` | `espacial/chao.py` | mover |
| `estado/rastreio.py` | `espacial/rastreio.py` | mover |
| `percepcao/fusao.py` | `espacial/fusao.py` | mover |
| `percepcao/pose3d.py` | `espacial/pose3d.py` + `visao/` | dividir |
| `estado/ocupacao.py` | `gemeo/ocupacao.py` | mover |
| `estado/planta.py` | `gemeo/` + `saidas/` | dividir |
| `visual/cena2d.py`, `cena3d.py` | `saidas/` | mover |
| `captura/dispositivos.py` | `cameras/` | mover |
| `captura/fonte.py` | `cameras/` | **reescrever** |
| `percepcao/gemeo_multi.py` | `app/` | **reescrever** como orquestrador |
| `percepcao/gemeo3d.py` | — | **remover** (caso N=1) |
| `calibracao/intrinseca.py` | `ferramentas/` | congelar, não usado |

---

## 12. Testes por componente

**Fontes** (com fonte falsa e vídeo gravado, sem hardware):
abre e entrega; não abre; abre e para de entregar → DEGRADADA → FALHA;
reconecta com recuo; métricas conferem; URL inválida.

**Buffer:** enche e descarta o antigo; contador de descarte confere; consumidor
lento não trava o produtor.

**Sincronizador:** monta com 3 fontes; monta com 1; descarta fora da
tolerância; reporta defasagem.

**Visão:** modelo indisponível; nenhuma detecção; múltiplas pessoas; frame
inválido.

**Espacial:** homografia ausente; pose sem tornozelo; fusão com uma vista;
todos os testes já escritos de Kalman e recostura.

**Twin:** sem câmera; três câmeras; pessoa perdida e reencontrada; zona
contando tempo por rastro.

**Regressão:** os números da auditoria viram testes — recostura a 30 e 4 fps,
render abaixo de 10 ms, fusão reduzindo erro.

---

## 13. Decisões que preciso de você

**D1 — Aplicativo do celular.** MJPEG/HTTP exige um app que sirva a câmera
como URL. No iOS as opções mudam com frequência; preciso que você verifique o
que está disponível hoje e me diga. Alternativa: manter o Iriun nesta fase e
trocar depois — funciona, mas não terá reconexão.

**D2 — Interface.** As janelas do OpenCV já falharam duas vezes na captura de
teclado. Manter por enquanto, ou já ir para um dashboard web via FastAPI +
WebSocket? O dashboard é mais trabalho, mas resolve interação de vez e é o que
o requisito descreve.

**D3 — Ordem das fases 3 e 4.** Posso entregar o `GerenciadorDeCameras` com
buffer junto, ou separar. Junto é menos ida e volta; separado é mais fácil de
validar.

---

## 14. O que esta arquitetura NÃO resolve

Honestidade sobre limites, para não haver surpresa:

**Não acelera nada por si só.** O sistema está em 7 fps porque o YOLO custa
84 ms. A arquitetura permite paralelizar, o que deve levar a ~11 fps. Ganho
real virá de ONNX, que é fase posterior.

**Não resolve associação entre vistas com várias pessoas.** Exige
re-identificação por aparência.

**Não resolve oclusão.** É limitação de informação, não de código.

**Não melhora a pose sozinha.** A qualidade continua limitada pelo MediaPipe e
pelo enquadramento das câmeras.

---

**Aguardando aprovação e as respostas de §13 para iniciar a Fase 3.**
