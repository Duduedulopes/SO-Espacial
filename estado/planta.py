"""
Planta da loja lida de um arquivo, e o estado publicado como dado.

DE ONDE VEIO A IDEIA

Do Eclipse Digital Twin / AAS: o gemeo tem identidade e submodelos, e e
DECLARADO em dados, nao programado. Nao adotamos o padrao inteiro — ele e
industrial, pesado e em Java. Adotamos a forma.

E do digital-twin-playground (Rust + Bevy), que usa um motor de jogo como
visualizador: isso so e possivel porque o ESTADO e serializavel e nao sabe
nada de desenho. Se o estado sai como JSON, qualquer coisa pode consumi-lo —
OpenCV, Unity, Godot, uma pagina web, ou a sua API em C#.

O que este modulo NAO faz: desenhar, detectar, filtrar. So descreve e publica.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from estado.ocupacao import MapaDeCalor, Zona


@dataclass
class Movel:
    """Um movel no chao, em metros.

    x, y SAO O CENTRO. Nao o canto. Isso precisa estar escrito porque as duas
    convencoes convivem em qualquer projeto que desenhe retangulos, e a
    diferenca entre elas e silenciosa: nada quebra, o movel so aparece meio
    corpo fora do lugar.

    ERRO DE 14/08, CORRIGIDO AQUI: `src/mundo/ambiente.py` media a estante e
    devolvia o CENTRO; `visual/cena3d.py` montava a caixa de (x, y) ate
    (x+largura, y+profundidade), ou seja, tratava o mesmo par como CANTO.
    Uma estante de 0,92 x 0,30 apareceria 46 cm ao lado e 15 cm a frente do
    lugar onde ela esta — sem erro nenhum na tela.

        Duas partes que concordam no nome de um campo e discordam no que ele
        significa nao tem um erro: tem um acordo falso.

    O centro venceu porque a rotacao exige um: um retangulo gira em torno do
    proprio centro, e `rumo_da_face` so faz sentido a partir dele.

    rumo_da_face   radianos, para onde a FACE do movel olha. A largura corre
                   ao longo de (cos, sin); a profundidade, ao longo da normal
                   (-sin, cos). Zero deixa a largura no eixo x, que e o
                   comportamento antigo — por isso o padrao nao quebra nada.

    prateleiras    [(id, altura_m)] medidas com trena, quando houver. Elas
                   vem do gabarito, nao das cameras: o que as cameras
                   acrescentam e ONDE o movel esta e para onde ele olha.
    """
    id: str
    nome: str
    tipo: str
    x: float
    y: float
    largura: float
    profundidade: float
    altura: float
    rumo_da_face: float = 0.0
    prateleiras: list = field(default_factory=list)
    estante: str | None = None


@dataclass
class Planta:
    id: str
    nome: str
    chao: tuple            # (xmin, xmax, ymin, ymax) — a CAIXA
    moveis: list = field(default_factory=list)
    zonas: list = field(default_factory=list)      # objetos Zona

    # O CONTORNO DO PISO, e ele nao e a caixa.
    #
    # A pegada de uma camera no chao e um quadrilatero torto — a imagem e um
    # retangulo, mas a projecao de um retangulo por uma perspectiva nao e.
    # A caixa que o contem tem 16,4 m2; o piso que a camera realmente mede tem
    # 8,4. Desenhar a caixa seria inventar metade do comodo.
    #
    #     A caixa serve para dimensionar (a grade do calor, o enquadramento).
    #     Para DESENHAR o chao, so o contorno diz a verdade.
    #
    # Vazio quando a planta e antiga: ai o desenho cai na caixa, como antes.
    contorno: tuple = ()   # ((x, y), ...) em metros

    @staticmethod
    def carregar(caminho):
        d = json.loads(Path(caminho).read_text(encoding="utf-8"))
        c = d["chao"]

        moveis = [
            Movel(m["id"], m["nome"], m.get("tipo", "movel"),
                  m["x"], m["y"], m["largura"], m["profundidade"], m["altura"],
                  rumo_da_face=float(m.get("rumo_da_face", 0.0)),
                  prateleiras=[(p["id"], float(p["altura"]))
                               for p in m.get("prateleiras", [])],
                  estante=m.get("estante"))
            for m in d.get("moveis", [])
        ]

        zonas = []
        for z in d.get("zonas", []):
            zona = Zona(z["nome"], z["x0"], z["x1"], z["y0"], z["y1"])
            zona.id = z["id"]
            zona.movel = z.get("movel")
            zonas.append(zona)

        return Planta(
            id=d["id"], nome=d["nome"],
            chao=(c["xmin"], c["xmax"], c["ymin"], c["ymax"]),
            moveis=moveis, zonas=zonas,
            contorno=tuple((float(p[0]), float(p[1]))
                           for p in d.get("contorno", ())),
        )

    def aplicar_na_cena(self, cena):
        for m in self.moveis:
            cena.add_movel(m.x, m.y, m.largura, m.profundidade, m.altura,
                           m.nome, rumo=m.rumo_da_face,
                           prateleiras=m.prateleiras)

    def novo_mapa_de_calor(self, px_por_m=60, meia_vida_s=90.0):
        return MapaDeCalor(*self.chao, px_por_m=px_por_m, meia_vida_s=meia_vida_s)


class Publicador:
    """Escreve o estado atual como JSON, para outro programa consumir.

    SEPARACAO ESTADO x HISTORICO — ideia vinda do OpenTwins, onde o Ditto
    guarda o AGORA e o InfluxDB guarda o PASSADO.

        estado_atual.json   sobrescrito a cada atualizacao. E a verdade agora.
        historico .jsonl    so cresce. E o que aconteceu.

    Misturar os dois num arquivo so parece economia e vira problema: quem quer
    saber "quem esta na loja agora" nao deveria precisar ler um arquivo de
    500 MB ate o fim.

    A escrita e ATOMICA (grava em .tmp e renomeia). Sem isso, quem le pode
    pegar o arquivo pela metade e receber JSON invalido.
    """

    def __init__(self, destino, a_cada_s=0.2):
        self.destino = Path(destino)
        try:
            self.destino.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[publicador] nao consegui criar {self.destino.parent}: {e}")
        self.a_cada = a_cada_s
        self._ultimo = 0.0
        self.falhas = 0

    def publicar_estado(self, estado, agora, forcar=False):
        """Grava um dicionario JA PRONTO — o instantaneo do gemeo.

        E a porta preferida. O gemeo e o dono da verdade; o publicador so leva
        essa verdade para fora. Montar o dicionario aqui, como faz `publicar()`
        abaixo, significaria ter uma SEGUNDA versao do estado, escrita em outro
        lugar, que pode divergir da primeira sem ninguem perceber.
        """
        if not forcar and (agora - self._ultimo) < self.a_cada:
            return False
        self._ultimo = agora
        return self._gravar(json.dumps(estado, indent=2, ensure_ascii=False))

    def publicar(self, planta, rastros, agora, forcar=False):
        """Porta antiga, usada pelo `gemeo_multi.py`, que nao tem DigitalTwin.

        Monta o estado a partir dos rastros crus. Sai de cena junto com o
        programa antigo.
        """
        if not forcar and (agora - self._ultimo) < self.a_cada:
            return False
        self._ultimo = agora

        estado = {
            "loja": {"id": planta.id, "nome": planta.nome},
            "t": round(agora, 3),
            "pessoas": [
                {
                    "id": meu,
                    "x": round(r.pos[0], 3),
                    "y": round(r.pos[1], 3),
                    "vx": round(r.kf.vel[0], 3),
                    "vy": round(r.kf.vel[1], 3),
                    "velocidade": round(r.kf.velocidade, 3),
                    "incerteza": round(r.kf.incerteza, 3),
                    "prevendo": bool(r.coasting),
                    "quadros": r.quadros,
                }
                for meu, r in rastros.items()
            ],
            "zonas": [
                {
                    "id": getattr(z, "id", z.nome),
                    "nome": z.nome,
                    "ocupacao": z.ocupacao,
                    "visitas": z.visitas,
                    "tempo_total_s": round(z.tempo_total, 1),
                    "tempo_medio_s": round(z.tempo_medio, 1),
                }
                for z in planta.zonas
            ],
        }

        return self._gravar(json.dumps(estado, indent=2, ensure_ascii=False))

    def _gravar(self, texto):
        """Escrita atomica. PUBLICAR NUNCA PODE DERRUBAR O PROGRAMA.

        Isto e um canal lateral: alguem la fora quer saber o estado. Se a
        escrita falhar, o certo e perder ESTA publicacao, nao a sessao.

        No Windows a renomeacao atomica falha com "acesso negado" quando o
        destino esta aberto em outro programa — um editor, um antivirus, o
        OneDrive sincronizando. Acontece de verdade, e nao e culpa de ninguem.
        """
        try:
            tmp = self.destino.with_suffix(".tmp")
            tmp.write_text(texto, encoding="utf-8")
            tmp.replace(self.destino)
            self.falhas = 0
            return True
        except (PermissionError, OSError) as e:
            self.falhas = getattr(self, "falhas", 0) + 1
            if self.falhas in (1, 50, 500):
                print(f"[publicador] falha {self.falhas} ao escrever "
                      f"{self.destino.name}: {e}")
                if self.falhas == 1:
                    print("[publicador] o arquivo esta aberto em outro programa? "
                          "Feche-o. O gemeo continua rodando.")
            return False
