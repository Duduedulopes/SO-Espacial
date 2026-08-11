# Arquitetura v3 — Descrição da ação

*Proposta de 10/08. Muda o eixo do projeto. Nenhuma implementação ainda.*

---

## 1. O problema, dito com precisão

Hoje o fluxo é:

```
cameras -> coordenadas 3D de 17 juntas -> desenhar essas coordenadas
```

O desenho **herda todo erro da reconstrução**. Em 10/08 isso produziu um
esqueleto deitado no chão enquanto a pessoa andava em pé. E não havia como
consertar o desenho: ele estava fielmente mostrando dados ruins.

Medições que sustentam o diagnóstico:

| sinal | qualidade hoje |
|---|---|
| posição no chão (homografia) | **2 a 5 cm** — sólida |
| continuidade do rastro | **99,6% de sobrevivência** — sólida |
| zonas, tempo, calor | corretos e testados |
| pose da frontal | **38% dos quadros** — fraca |
| fusão 3D das juntas | desenha deitado — **inutilizável** |

O elo fraco é a pose 3D. Toda a visualização depende dele.

## 2. A proposta

```
cameras -> DESCRIÇÃO do que está acontecendo -> animar um corpo que já sabe ser corpo
```

Em vez de transmitir onde cada junta está, transmitir **o que a pessoa está
fazendo**, num vocabulário fechado. O renderizador anima um esqueleto de
proporções corretas segundo essa descrição.

**Consequência que decide a questão:** se o vocabulário não tem "deitado", o
boneco não consegue deitar. A classe inteira de defeito desaparece por
construção, não por conserto.

Nomes para estudar: *Human Activity Recognition* sobre esqueletos, para a
classificação; animação procedural dirigida por parâmetros, para o desenho —
que é como motores de jogo funcionam há vinte anos. Não se transmite mocap
cru; transmitem-se parâmetros.

## 3. O vocabulário

Três eixos **independentes que se combinam**, e não uma lista única.

Lista única exigiria um item para cada combinação e explodiria. Com eixos,
"andando para frente + agachado + braço direito estendido" é válido sem
nenhuma entrada nova.

### Locomoção (relativa ao corpo, não ao mundo)

    parado
    andando para frente
    andando para trás
    andando para a esquerda        (de lado)
    andando para a direita         (de lado)
    virando para a esquerda
    virando para a direita
    meia-volta                     (giro > 150° numa janela curta)

O rumo é **relativo ao corpo**. É o que permite dizer "andou de lado" em vez
de "andou para o norte" — e é o que o desenho precisa para escolher a
animação.

### Postura

    em pé
    agachado

### Braços (um estado por lado, independentes)

    ao lado
    levantado
    estendido à frente

## 4. De onde sai cada estado

| estado | sinal | confiabilidade | precisa de pose? |
|---|---|---|---|
| parado / andando / rumo | velocidade do Kalman | **alta, provada** | **não** |
| virando / meia-volta | variação do rumo | **alta** | **não** |
| em pé / agachado | altura do quadril sobre o chão | média | 2D basta |
| braços | pulso vs ombro, **2D de uma câmera** | média-alta | 2D basta |

**Nada aqui depende da fusão 3D.** Os dois primeiros não dependem de pose
nenhuma. Os outros precisam apenas de landmarks 2D de **uma** câmera — a que
estiver vendo melhor naquele instante.

É esse o ganho central: **o sistema para de depender do elo mais fraco.**

## 5. A camada que importa para o negócio

O vocabulário de ação existe para servir a uma pergunta comercial, não para
fazer bonito na tela.

> **Não detectar o produto. Detectar a mão entrando no lugar onde o produto
> está cadastrado.**

Reconhecer um produto por imagem é problema duro: iluminação, oclusão,
embalagens parecidas, catálogo que muda. Mas se a planta declara que na
prateleira A, entre 1,10 m e 1,35 m de altura, está o produto X, a pergunta
que a visão precisa responder vira:

    o pulso desta pessoa entrou naquele volume e voltou?

Sim ou não. **O produto sai do cadastro, não da imagem.**

Isso converte visão computacional difícil em geometria fácil — e a geometria
é exatamente o que este projeto já faz bem.

### Eventos novos

    MAO_ENTROU_NA_PRATELEIRA   (pessoa, prateleira)
    MAO_SAIU_DA_PRATELEIRA     (pessoa, prateleira, duracao_s)
    PRODUTO_PEGO               (pessoa, produto)     <- inferido do cadastro

Os dois primeiros são observação. O terceiro é conclusão, e deve carregar a
confiança que a sustenta.

### Encontro com o outro projeto

O catálogo de produtos já existe no **AutonomousStore**. É aqui que os dois
projetos se juntam: o SO Espacial diz *quem pegou de onde*, o AutonomousStore
diz *o que é aquilo e quanto custa*.

## 6. Sobre "hiper rápido" — onde o tempo realmente está

A camada de descrição **não custa nada**. Classificar postura é aritmética
sobre uma dezena de números. O gêmeo inteiro custa 0,2 ms por quadro hoje.

Medido em 10/08, com o sistema gravando a tela:

```
visao          156,9 ms   67%      <- detector 156 ms, pose 52 ms
esperando       72,4 ms   31%
gemeo            0,2 ms    0,1%
```

**Sem gravação, o mesmo sistema roda a 10,7 fps; com gravação, 4,2.** O ato de
medir alterou a medida — vale lembrar sempre que comparar execuções.

A conclusão prática: acrescentar a descrição não deixa mais lento, e tirá-la
não deixaria mais rápido. **Se o objetivo é velocidade, o caminho é exportar o
detector para ONNX.** São duas obras diferentes e não competem.

## 7. O perigo, e como ele foi resolvido

Uma animação dirigida por descrição fica **bonita mesmo quando está errada**.
Se o classificador disser "em pé" e a pessoa estiver agachada, a tela mostra
uma figura perfeita e confiante. É o mesmo problema de métrica que mente que
custou o dia 10/08 inteiro — aplicado ao desenho.

Duas defesas, e a segunda veio da conversa:

**Vocabulário fechado.** O erro deixa de ser ilimitado. O boneco não faz nada
fisicamente impossível; no pior caso escolhe o padrão errado dentro de um
conjunto de padrões todos plausíveis.

**Escopo honesto do que importa.** Não há decisão de negócio pendurada em
postura livre. Se alguém se apoiar na bancada de um jeito não modelado, o
sistema mostra a pessoa ali e segue — sem drama e sem inventar. O que precisa
ser confiável é o par *mão entrou / mão saiu*, e esse tem alvo grande e pessoa
parada na frente dele.

**Limite que fica declarado:** a confiança de cada estado viaja junto com ele.
Estado incerto não pode chegar ao desenho com a mesma aparência de estado
medido. Como isso se mostra na tela é decisão de implementação; que se mostre
não é negociável.

## 8. Etapas, cada uma medível sozinha

**A. Locomoção e postura a partir do que já é confiável.**
Sem modelo novo, sem inferência, sem hardware para testar. Velocidade do
Kalman no referencial do corpo, variação de rumo, altura do quadril.
*Prova: testes com trajetórias sintéticas conhecidas.*

**B. Braços a partir de landmarks 2D de uma câmera.**
Pulso acima do ombro, pulso distante do tronco. Classificação, não coordenada.
*Prova: sequências gravadas com o gesto anotado à mão.*

**C. Renderizador dirigido por estado.**
Esqueleto de proporções conhecidas, interpolando entre estados. Nunca deita.
*Prova: o desenho não pode contradizer o estado publicado.*

**D. Prateleiras como volumes e o par mão-entrou/mão-saiu.**
Zonas ganham altura. Eventos de interação.
*Prova: pegar um objeto conhecido N vezes e contar acertos e erros.*

**E. Camada narrativa assíncrona.**
Aqui entra um modelo de linguagem — lendo o **fluxo de eventos**, não os
quadros. "A pessoa circulou pela zona A por 12 s, pegou algo, foi ao caixa."
Fora do laço crítico; 500 ms de latência não incomodam ninguém.

Colocar um modelo de linguagem dentro do laço de 10 fps mataria o sistema.
Colocá-lo sobre os eventos é onde ele ganha de qualquer regra que a gente
escreva.

## 9. O que NÃO muda

A percepção continua igual: câmeras, sincronizador, detector, homografia,
Kalman, rastros, zonas, calor, publicação. Tudo isso está medido e funciona.

A camada de ação **consome** o que o `SpatialEngine` já produz e **emite**
pelo `EventEngine` que já existe. Nenhuma reescrita.

## 10. Pendências herdadas de 10/08

- A linha `inclinacao` foi acrescentada ao `resumo()` do motor espacial e
  **não** ao painel. O número existe e não aparece na tela.
- O esqueleto 3D continua desenhando deitado. Com a v3 ele deixa de ser o
  caminho principal — mas a decisão de mantê-lo, corrigi-lo ou remover ainda
  não foi tomada.
- A frontal produz pose em 38% dos quadros. Investigar se é enquadramento ou
  outra coisa continua valendo, porque a etapa B depende dela.
