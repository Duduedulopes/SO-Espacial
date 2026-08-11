# Auditoria do SO Espacial — 09/08/2026

Levantamento do estado real do sistema antes de qualquer reestruturação.
Nenhum código foi alterado para produzir este documento.

---

## 1. Arquitetura atual

O projeto **não tem** uma arquitetura de captura. Tem uma biblioteca de
percepção razoável pendurada num laço de captura improvisado.

```
CameraAoVivo (1 thread por câmera, sem fila)
        │
        └──► laço único em gemeo_multi.py ──┬─► YOLO (câmera do alto)
                                            ├─► MediaPipe (frontal)
                                            ├─► MediaPipe (lateral)
                                            ├─► homografia
                                            ├─► Kalman
                                            ├─► ocupação e zonas
                                            ├─► publicação JSON
                                            └─► desenho 2D/3D
```

**Tudo acontece no mesmo laço, no mesmo thread, em sequência.** As câmeras
capturam em threads próprias, mas o consumo é serializado: se o MediaPipe da
lateral demora 33 ms, todo o resto espera.

### Inventário

| módulo | linhas | papel |
|---|---|---|
| `captura/fonte.py` | 256 | thread de captura, exposição, aquecimento |
| `captura/dispositivos.py` | 103 | nomes das câmeras via DirectShow |
| `captura/identificar.py` | ~230 | atribuição de papéis, gravação em JSON |
| `captura/diagnostico.py` | ~200 | medição de fps, brilho, codec |
| `captura/reparar.py` | ~160 | recuperação de câmera travada |
| `captura/gravar.py` | 224 | gravação de sessões |
| `calibracao/homografia.py` | 310 | calibração do plano do chão |
| `calibracao/intrinseca.py` | ~250 | calibração de lente (não usado ainda) |
| `percepcao/chao.py` | 285 | pixel→metro, ponto do pé, filtros |
| `percepcao/pose3d.py` | ~380 | MediaPipe, inclinação, suavização |
| `percepcao/fusao.py` | ~180 | fusão de vistas por eixo |
| `percepcao/gemeo_multi.py` | 317 | **o laço monolítico** |
| `percepcao/gemeo3d.py` | ~320 | versão de uma câmera |
| `percepcao/mapa.py` | ~200 | versão 2D |
| `estado/rastreio.py` | 230 | Kalman 2D, recostura de identidade |
| `estado/ocupacao.py` | 130 | mapa de calor, zonas |
| `estado/planta.py` | 163 | planta em JSON, publicação de estado |
| `visual/cena2d.py` | 262 | render 2D |
| `visual/cena3d.py` | 435 | render 3D |

Cerca de **4.400 linhas**. Não é um protótipo pequeno.

---

## 2. Tecnologias

| camada | tecnologia | avaliação |
|---|---|---|
| linguagem | Python 3.13 | adequada |
| captura | OpenCV `VideoCapture` + DSHOW | **é a fonte dos problemas** |
| enumeração | `pygrabber` (DirectShow) | correta, recém-introduzida |
| detecção | Ultralytics YOLO11n / YOLO11n-pose | adequada; AGPL-3.0 é risco comercial |
| pose 3D | MediaPipe Tasks (PoseLandmarker lite) | adequada |
| rastreio | ByteTrack (Ultralytics) + Kalman próprio | adequado |
| geometria | homografia (OpenCV) | correta e validada em 2 a 5 cm |
| render | OpenCV `imshow` | limite claro de escalabilidade |
| persistência | JSON em arquivo | suficiente para o estado atual |
| câmera do celular | Iriun (driver de webcam virtual) | **decisão errada, ver §5** |

**Ausentes:** filas, backpressure, logging estruturado, métricas,
reconexão, testes, API, banco.

---

## 3. Fluxo de dados atual

```
1. identificar.py grava config/cameras.json com NOMES
2. gemeo_multi.py resolve nome → índice via pygrabber
3. CameraAoVivo abre cada câmera, uma thread por câmera
4. o laço principal chama cam.ler() — pega o último quadro, sem fila
5. YOLO roda no quadro do alto → caixas + IDs + keypoints
6. filtros: plausibilidade geométrica, tornozelo
7. homografia converte pixel → metros
8. Kalman suaviza e mantém identidade
9. MediaPipe roda em frontal e lateral, quadro inteiro
10. Fusor combina eixos das duas vistas
11. ancoragem no chão + rumo pela velocidade
12. ocupação, zonas, mapa de calor
13. Publicador grava dados/estado_atual.json
14. Cena3D desenha
```

**Timestamps existem** (`CameraAoVivo` guarda `t` por quadro), mas **não são
usados para sincronizar nada**. O `Fusor` usa uma janela de validade de 0,5 s,
que é tolerância, não sincronização.

---

## 4. Como as três câmeras estão sendo tratadas

Todas as três passam pela **mesma classe** `CameraAoVivo`, que assume
dispositivo local acessível por índice do DirectShow.

| câmera | tratamento atual | correto? |
|---|---|---|
| C920 (USB) | `VideoCapture(idx, CAP_DSHOW)` | sim |
| VGA (notebook) | `VideoCapture(idx, CAP_DSHOW)` | sim |
| iPhone (Iriun) | `VideoCapture(idx, CAP_DSHOW)` | **não** |

O celular é tratado como webcam local porque um driver o disfarça de webcam.
Isso esconde a natureza remota da fonte: latência variável, queda de rede,
reconexão. O sistema não tem como saber que aquela fonte é diferente.

---

## 5. Problemas encontrados

### P1 — Backends misturados corrompem a identidade da câmera

**Severidade: crítica. Já causada e diagnosticada em 08/08.**

`pygrabber` enumera pela ordem do **DirectShow**. O MSMF tem ordem própria. O
fallback "se DSHOW falhar, tenta MSMF" fazia o índice 1 apontar para **outro
dispositivo físico**.

Sintomas que isso produzia, e que foram atribuídos a cinco causas diferentes:

- janela "alto" mostrando a cena da lateral
- imagem esverdeada (formato de pixel de outra câmera)
- aviso de "duas câmeras mostram a mesma cena"
- "VGA camera" reportando 1280×720 e a C920 reportando 640×480

Já corrigido (fallback removido), mas revela a falha de projeto: **o backend é
parte da identidade da câmera e não estava modelado.**

### P2 — Ausência de fila e de backpressure

`CameraAoVivo.ler()` devolve o último quadro. Não há fila, não há contagem de
descarte, não há como saber quantos quadros se perderam.

Consequência: impossível medir `frames_received` e `frames_dropped`, que são
critérios de sucesso declarados.

### P3 — Consumo serializado no laço principal

As câmeras capturam em paralelo, mas são **consumidas em sequência**. Medido:
`yolo 84ms + frontal 26ms + lateral 33ms ≈ 143 ms` → ~7 fps.

Uma câmera lenta atrasa todas. Não atende ao requisito "se uma câmera estiver
lenta, ela não pode travar o pipeline inteiro".

### P4 — Sem reconexão

Se uma câmera cair, `ler()` devolve o último quadro **para sempre**, e o
sistema continua processando uma imagem congelada sem perceber. Pior que
falhar: mente.

### P5 — Sem sincronização temporal

Os quadros têm carimbo mas ninguém os alinha. O `Fusor` combina o mais recente
de cada vista, com validade de 0,5 s. A 1,4 m/s, 200 ms de defasagem são 28 cm
de erro — e a fusão de eixos assume que as vistas mostram **o mesmo instante**.

### P6 — Estado global implícito no hardware

A C920 guarda exposição, brilho e ganho **no próprio driver**, entre execuções.
Descoberto em 08/08 depois de horas: `Status: OK` no Windows, imagem preta.

Agravado por um erro meu: o código media o brilho **antes de a câmera acordar**,
via zero, e disparava dezenas de `set()` que travavam o dispositivo. O
diagnóstico apressado virou a causa do defeito.

### P7 — Acoplamento: `gemeo_multi.py` sabe de tudo

O laço principal conhece câmeras, YOLO, MediaPipe, homografia, Kalman, zonas,
publicação e desenho. 317 linhas com nove responsabilidades.

Consequência prática: adicionar uma quarta câmera exige editar o laço.

### P8 — Sem logging estruturado nem métricas

Há `print()` espalhados. Não há níveis, nem timestamps, nem arquivo, nem
contadores. Depurar depende de ler o terminal ao vivo.

### P9 — Sem testes

Zero testes automatizados em 4.400 linhas. Toda verificação foi manual, o que
explica a quantidade de regressões nos últimos dois dias.

### P10 — O celular como webcam virtual

Ver §4. Esconde a natureza remota da fonte.

### P11 — `imshow` como interface

Cinco janelas do OpenCV. Já falhou duas vezes na captura de teclado, obrigando
a trocar o mecanismo de interação duas vezes. Não escala para dashboard.

---

## 6. Causas raiz

Agrupando os onze problemas, sobram **quatro causas**:

**C1. A fonte de vídeo não é uma abstração.** Não há um tipo `FonteDeVideo`
com contrato explícito (estado, reconexão, métricas, origem). Há uma classe
que abre webcam local. Tudo em P1, P4, P10 vem daí.

**C2. Não há pipeline.** Há um laço. Sem filas, o desacoplamento é impossível,
e P2, P3, P5 são consequências inevitáveis.

**C3. Não há observabilidade.** Sem métricas e logs, cada problema exigiu uma
sessão de investigação manual. P6, P8, P9.

**C4. Crescimento por acréscimo, sem revisão de estrutura.** O sistema nasceu
para uma câmera e foi esticado para três. P7 e P11 são o resultado.

---

## 7. O que deve ser MANTIDO

Estas partes estão corretas, medidas, e devem ser preservadas:

| módulo | por quê |
|---|---|
| `percepcao/chao.py` | homografia validada com trena: 2 mm e 0 mm de erro; ponto do pé sem teletransporte (saltos >50 cm: 5→0); filtro de plausibilidade por horizonte |
| `estado/rastreio.py` | Kalman em metros, recostura de identidade testada a 30 e 4 fps |
| `estado/ocupacao.py` | mapa de calor com esquecimento, zonas por rastro |
| `estado/planta.py` | planta declarativa em JSON, publicação atômica |
| `percepcao/fusao.py` | **validado pela literatura** — fusão de esqueletos 3D é mais robusta que triangulação de 2D |
| `percepcao/pose3d.py` | MediaPipe, inclinação automática, suavização adaptativa |
| `visual/cena3d.py` | render com cache: 87 ms → 4 ms |
| `calibracao/homografia.py` | ferramenta funcional, ordenação automática de cantos |
| `captura/dispositivos.py` | identificação por nome, correta |

**Cerca de 60% do código é aproveitável.** O problema está concentrado na
captura e na orquestração.

---

## 8. O que deve ser REMOVIDO ou REESCRITO

| item | ação | justificativa |
|---|---|---|
| `percepcao/gemeo_multi.py` | **reescrever** | 317 linhas com 9 responsabilidades; vira orquestrador fino |
| `captura/fonte.py` | **reescrever** | vira hierarquia de fontes com estado e reconexão |
| `percepcao/gemeo3d.py` | **remover** | duplica `gemeo_multi` com uma câmera; caso particular de N=1 |
| `percepcao/mapa.py` | **manter** | visualização 2D leve, útil para diagnóstico rápido |
| `visual/cena2d.py` | manter | consumido por `mapa.py` |
| `captura/reparar.py` | **absorver** | vira método `recuperar()` da fonte USB |
| `captura/diagnostico.py` | manter | ferramenta de bancada, independente |
| `calibracao/intrinseca.py` | **congelar** | escrito mas não usado; só necessário se houver triangulação, que a literatura desaconselha |
| `experimentos/` | manter | citados no caderno; não importados |

Nada é apagado sem substituto funcionando.

---

## 9. Arquitetura recomendada

```
┌─────────────── FONTES ──────────────┐
│  UsbCameraSource   (C920, VGA)      │  cada uma: thread própria,
│  RemoteCameraSource(iPhone)         │  fila limitada, estado, reconexão
└──────────────┬──────────────────────┘
               │  Frame{camera_id, ts_mono, ts_wall, seq, image}
               ▼
        ┌──────────────┐
        │ FrameBuffer  │  fila de 1 a 3 por câmera, descarta o antigo
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │ Sincronizador│  agrupa por proximidade de timestamp
        └──────┬───────┘
               ▼
   ┌───────────────────────────┐
   │       VisionEngine        │  processos separados por papel
   │  Detector (alto)          │
   │  PoseEstimator (front)    │
   │  PoseEstimator (lateral)  │
   └──────────┬────────────────┘
              ▼
   ┌───────────────────────────┐
   │      SpatialEngine        │  homografia, Kalman, fusão de eixos
   └──────────┬────────────────┘
              ▼
   ┌───────────────────────────┐
   │       DigitalTwin         │  estado do ambiente, único dono da verdade
   └──────────┬────────────────┘
              ├──► EventEngine  ──► logs, automações
              ├──► Publicador   ──► estado_atual.json / WebSocket
              └──► Renderers    ──► Cena2D, Cena3D, dashboard
```

### Decisões e alternativas avaliadas

**Threads por fonte, não processos.**
*Alternativa:* `multiprocessing` — isolamento real, mas exige serializar
quadros de 2,7 MB entre processos, o que custa mais que o ganho nesta escala.
*Alternativa:* `asyncio` — o `VideoCapture.read()` é bloqueante e não
cooperativo; teria que ir para executor de qualquer forma.
*Escolha:* threads. O `read()` do OpenCV libera o GIL, então há paralelismo
real na captura. Revisar se passar de 6 câmeras.

**Fila limitada com descarte do mais antigo.**
Para visão em tempo real, o quadro velho não tem valor. Fila de tamanho 1 a 3,
`drop-oldest`. Isso dá `frames_dropped` de graça, que hoje não existe.

**Celular por RTSP ou MJPEG, não por driver de webcam virtual.**
*Medido na literatura:* WebRTC 100 a 500 ms; RTSP 0,5 a 2 s; MJPEG sobre HTTP
tem latência maior mas implementação trivial e é suportado nativamente pelo
`VideoCapture` via URL.
*Escolha inicial:* **MJPEG sobre HTTP** (`IP Webcam` no Android, `Camo`/apps
equivalentes no iOS), porque o `VideoCapture` já lê URL e a fonte passa a ser
explicitamente remota. *Evolução:* WebRTC via MediaMTX quando a latência
importar.
*Vantagem imediata sobre o Iriun:* o sistema sabe que a fonte é remota, e pode
reconectar sem depender de driver.

**Backend fixo em DirectShow.**
Não é preferência técnica — é que a **enumeração por nome vem do DirectShow**.
Misturar backends quebra a identidade. Se o MSMF for necessário no futuro, a
enumeração precisa vir dele também.

**JSON em arquivo agora; banco depois.**
Não há consulta histórica no requisito atual. Quando houver, SQLite para
eventos e configuração, e nada de frames em banco.

---

## 10. Tecnologias recomendadas

| necessidade | escolha | motivo |
|---|---|---|
| captura local | OpenCV + DSHOW | já funciona; sem alternativa melhor no Windows |
| captura remota | MJPEG/HTTP → RTSP | explicitamente remota, reconectável |
| detecção | YOLO11n | 84 ms/quadro medido; exportar para ONNX deve render 2-3× |
| pose | MediaPipe Tasks | 26-33 ms medido |
| rastreio | ByteTrack + Kalman próprio | validado |
| logging | `logging` da stdlib, JSON | sem dependência nova |
| testes | `pytest` | padrão |
| API futura | FastAPI + WebSocket | quando houver dashboard web |

---

## 11. Referências

- [Multi-view Pose Fusion for Occlusion-Aware 3D HPE](https://arxiv.org/pdf/2408.15810) — fusão de esqueletos 3D é mais robusta que triangular 2D. **Valida `percepcao/fusao.py` e desaconselha o caminho de triangulação com tabuleiro.**
- [Real-time multi-camera 3D HPE at the edge](https://www.sciencedirect.com/science/article/pii/S0957417424009552) — nós distribuídos por câmera + agregador central. É a arquitetura proposta em §9.
- [Digital Twin Generation from Visual Data: A Survey](https://arxiv.org/abs/2504.13159) — 3DGS/NeRF para gerar a geometria do ambiente a partir de vídeo de celular.
- [BlendMimic3D](https://arxiv.org/pdf/2404.16136) — dataset de oclusão em cenário de supermercado.
- [OpenCV #27917 — MSMF lentidão ao definir resolução](https://github.com/opencv/opencv/issues/27917)
- [Kurokesu — resolução total de webcam no Windows](https://www.kurokesu.com/main/2020/07/12/pulling-full-resolution-from-a-webcam-with-opencv-windows/) — MJPG e banda USB
- [openpnp — USB Camera Troubleshooting FAQ](https://github.com/openpnp/openpnp/wiki/USB-Camera-Troubleshooting-FAQ)
- [WebRTC Latency — comparação de protocolos](https://www.nanocosmos.net/blog/webrtc-latency/)
- [Latência dos protocolos: RTSP, RTMP, SRT, HTTP-FLV, WebRTC](http://happytimesoft.com/knowledge/latency-in-mainstream-streaming-protocols.html)

---

## 12. Plano em fases

| fase | entrega | critério de conclusão |
|---|---|---|
| **1** | esta auditoria | aprovada |
| **2** | arquitetura detalhada, contratos das classes | aprovada, sem código |
| **3** | `FonteDeVideo` + `GerenciadorDeCameras` + tela de diagnóstico | 3 câmeras online, uma pode cair e voltar, contadores corretos |
| **4** | `FrameBuffer` + sincronizador | fila limitada, `dropped` medido, câmera lenta não trava as outras |
| **5** | `VisionEngine` isolado | mesma qualidade de hoje, mas desacoplado |
| **6** | `SpatialEngine` | reaproveita `chao.py`, `rastreio.py`, `fusao.py` |
| **7** | `DigitalTwin` + `EventEngine` | eventos emitidos e registrados |
| **8** | dashboard | status, métricas, twin ao vivo |

**Nada avança sem o anterior estar medido.**

---

## 13. Testes necessários

**Fontes:** uma câmera; duas; três; desconexão durante operação; reconexão
automática; câmera remota offline; URL inválida; resolução recusada.

**Pipeline:** fila cheia descarta o antigo; câmera lenta não bloqueia; quadro
inválido; contadores conferem.

**Visão:** detector indisponível; nenhuma detecção; múltiplas pessoas.

**Espacial:** homografia ausente; pose sem tornozelo; fusão com uma vista só.

**Twin:** sem câmera nenhuma; com três; objeto perdido e reencontrado.

Os testes de fonte usam **vídeo gravado** e uma fonte falsa — não dependem de
hardware, e é isso que permite rodá-los sempre.

---

## 14. Riscos

| risco | probabilidade | impacto | mitigação |
|---|---|---|---|
| banda USB insuficiente para 2 câmeras a 720p | alta | alto | MJPG forçado; iPhone por rede não usa USB |
| CPU insuficiente | **certa** — 7 fps hoje | alto | ONNX; detecção intercalada; resolução por papel |
| C920 travando | alta | médio | recuperação automática; reconexão |
| sincronização entre fontes | média | médio | tolerância explícita e medida |
| licença AGPL do Ultralytics | certa se virar produto | alto | avaliar RT-DETR ou YOLO sob licença permissiva |

---

## Conclusão

O sistema tem **percepção boa e captura frágil**. As medições que sustentam
isso são reais: 2 a 5 cm de precisão de posição, esqueleto fundido reduzindo
erro de 33 cm para 1,3 cm em simulação, render de 87 ms para 4 ms.

O que falha é tudo que cerca a percepção: abrir câmeras, mantê-las vivas, não
misturar identidades, medir o que acontece.

**A recomendação é reestruturar captura e orquestração, preservando percepção,
geometria, estado e render.** Cerca de 60% do código continua.

**Aguardando aprovação para a Fase 2.**
