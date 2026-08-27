# Dependências de terceiros

A licença MIT em [`LICENSE`](LICENSE) cobre o código deste repositório. Ela não
se estende às bibliotecas que o projeto instala em tempo de execução — cada uma
tem a própria licença.

Este arquivo existe separado do `LICENSE` de propósito: o detector de licenças
do GitHub compara o `LICENSE` com os textos conhecidos e exige correspondência
quase exata. Qualquer nota anexada ali faz o repositório aparecer como "Other"
em vez de "MIT".

## A que exige atenção

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

## As demais

| Biblioteca | Licença |
|---|---|
| OpenCV (`opencv-python`) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| MediaPipe | Apache-2.0 |
| PyYAML | MIT |
| pygrabber | MIT |
| Ultralytics | **AGPL-3.0** |

---

*Isto é a leitura de um desenvolvedor sobre textos jurídicos, não parecer de
advogado. Antes de fechar contrato apoiado neste arquivo, consulte alguém
habilitado.*
