# Esquema de dados e protocolo de captura

Este é o documento mais importante do projeto. Modelo se troca; dado mal
coletado não se conserta.

---

## O problema central: relógios

Você tem duas fontes que precisam contar a mesma história:

- **A câmera**, que produz quadros
- **O RFID**, que produz eventos "tag X foi lida"

Para o RFID rotular o vídeo, é preciso responder: *o evento das 19h32m10,481s
corresponde a qual quadro?*

Isso é mais sutil do que parece, por três motivos.

### 1. Relógios de parede pulam

`time.time()` (relógio de parede) pode andar para trás — sincronização NTP,
horário de verão, ajuste manual. Se ele pular no meio de uma gravação, seus
timestamps ficam inconsistentes e você não percebe.

`time.monotonic_ns()` nunca anda para trás, mas **não tem significado absoluto**
— só serve para medir intervalos, e o zero muda a cada reinício.

**Solução: grave os dois.**

- `t_mono_ns` → para ordenar quadros e medir intervalos dentro da sessão
- `t_wall` → para casar com o mundo exterior, incluindo o RFID

### 2. O carimbo tem que ser no lugar certo

O timestamp deve ser tomado **imediatamente depois** de `cam.read()` retornar,
nunca antes do laço nem depois do processamento.

Ainda assim ele não é o instante em que a luz atingiu o sensor — há latência de
exposição, transporte USB e buffer do driver. Essa latência é razoavelmente
constante, o que significa que ela vira um **viés fixo**, não ruído. Viés fixo
se mede e se corrige. É por isso que existe o procedimento da claquete abaixo.

### 3. O evento RFID chega com atraso de rede

A tag é lida na ESP32, mas o timestamp que temos é o da **chegada na API**.
Entre um e outro há wifi, HTTP e processamento — algo entre 30 e 300 ms,
variável.

Boa notícia: a API roda no **mesmo PC** que grava o vídeo. Então os dois usam o
mesmo relógio de parede, e sobra apenas a latência de rede — que também se mede.

---

## Procedimento da claquete

Emprestado do cinema, onde a claquete existe exatamente para sincronizar som e
imagem.

**No início e no fim de cada sessão de gravação:**

1. Segure uma tag RFID conhecida em frente à câmera, bem visível
2. Encoste-a no leitor com um movimento **rápido e seco**
3. Repita 3 vezes, com uns 2 segundos entre elas

Depois, no processamento: você identifica visualmente o quadro do contato e
compara com o timestamp do evento RFID. A diferença é o **offset** entre as
duas fontes.

Fazendo no início **e** no fim, você também detecta *drift* — se o offset mudou
ao longo da sessão, os relógios estão correndo em velocidades diferentes.

Sem isso, todo o resto do dataset é suspeito. Com isso, você tem uma medida de
erro em vez de uma esperança.

---

## Estrutura de uma sessão

Cada gravação vira uma pasta em `dados/sessoes/`:

```
dados/sessoes/2026-08-07_193200/
├─ meta.json            o que, onde, como
├─ video.mp4            imagem
├─ frames.jsonl         um registro por quadro
├─ eventos_rfid.jsonl   um registro por leitura de tag
└─ anotacoes.jsonl      (mais tarde) rótulos manuais ou derivados
```

### Por que JSONL e não CSV

**JSON Lines** = um objeto JSON por linha.

- **Append-safe**: escreve linha a linha; se o programa morrer, o que já foi
  gravado continua válido. CSV com cabeçalho e campos variáveis não dá isso.
- **Schema flexível**: campos novos não quebram leitores antigos.
- **Legível a olho nu** e fácil de inspecionar com qualquer editor.

### `meta.json`

```json
{
  "sessao_id": "2026-08-07_193200",
  "inicio_wall": "2026-08-07T19:32:00.123456-03:00",
  "camera": {
    "indice": 0,
    "modelo": "Logitech C270",
    "largura_pedida": 1280, "altura_pedida": 720, "fps_pedido": 30,
    "largura_real": 1280, "altura_real": 720, "fps_real": 30.0,
    "fourcc": "MJPG"
  },
  "cena": {
    "local": "bancada de teste",
    "descricao": "prateleira com 4 produtos, uma pessoa",
    "planta": null
  },
  "operador": "Eduardo",
  "notas": "claquete no inicio e no fim, tag 2A-B5-64-E1"
}
```

Registre **pedido e real** separadamente. É comum a câmera ignorar o que você
pediu, e descobrir isso três meses depois invalida comparações.

### `frames.jsonl`

Um objeto por quadro:

```json
{"i": 0, "t_mono_ns": 812374619283, "t_wall": "2026-08-07T19:32:00.157-03:00"}
{"i": 1, "t_mono_ns": 812407952616, "t_wall": "2026-08-07T19:32:00.190-03:00"}
```

- `i` — índice do quadro, casa com a posição no `.mp4`
- `t_mono_ns` — relógio monotônico em nanossegundos
- `t_wall` — relógio de parede ISO 8601 **com fuso**

Sempre grave o fuso. Timestamp sem fuso é uma dívida que sempre vence.

### `eventos_rfid.jsonl`

```json
{"t_wall": "2026-08-07T19:32:14.481-03:00", "tag": "2A-B5-64-E1", "produto": "Energetico 250ml", "sessao_loja": "a3f1...", "origem": "api"}
```

O campo `origem` diz de onde veio o carimbo — `api` (chegada no servidor) ou
`esp32` (se algum dia a placa tiver relógio confiável). Guardar a proveniência
do timestamp permite corrigir depois.

---

## Regras de coleta

**1. `dados/bruto/` é somente escrita.** Nada é editado, limpo ou sobrescrito.
Processamento gera arquivo novo em outro lugar.

**2. Uma sessão = uma pasta = um contexto.** Mudou a posição da câmera, a
iluminação ou a cena? Nova sessão. Não emende.

**3. Grave o fracasso também.** Sessão em que a leitura falhou, a pessoa passou
rápido demais, a luz estourou — isso é dado valioso. Modelos treinados só em
condições boas falham em condições reais.

**4. Anote a olho.** O campo `notas` do `meta.json` vale mais do que parece.
"A luz do fim da tarde bate na prateleira 2" explica, meses depois, por que
aquela sessão tem resultado estranho.

**5. Varie de propósito.** Ângulos, distâncias, iluminação, roupas, velocidade
do movimento. Um dataset uniforme produz um modelo frágil.

---

## Como saber se o dataset presta

Antes de treinar qualquer coisa, um dataset precisa responder sim a estas
perguntas:

- Consigo, para qualquer evento RFID, apontar o quadro correspondente?
- Sei qual é o erro dessa correspondência, em milissegundos?
- Consigo reproduzir exatamente o mesmo experimento amanhã?
- Se eu apagar todo o código, o dado ainda faz sentido sozinho?

A última é a mais importante. O dado deve ser **autodescritivo**. É por isso que
`meta.json` existe.
