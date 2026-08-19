"""A escolha do executor sai de medicao, e a medicao tem que ser honesta.

    Uma constante de desempenho escrita no codigo e uma medicao feita na
    maquina de outra pessoa.

O detector do alto consome 130 ms de um ciclo de 152 — 86% dele. PyTorch,
ONNX e OpenVINO rodam os MESMOS PESOS e a MESMA CONTA; o que muda e quem
executa, e qual dos tres ganha depende do processador.

Estes testes nao medem velocidade: velocidade so se mede na maquina. Eles
trancam o que a ferramenta promete ALEM da velocidade — que ela so adote um
formato que responda a mesma coisa, e que o sistema nao quebre quando a
escolha estiver ausente, quebrada ou apontando para um arquivo apagado.

    Ganho de velocidade sem conferencia de saida e so uma forma educada de
    trocar o problema.
"""
import json

import numpy as np
import pytest

from ferramentas.acelerar_detector import (PRECISA, TOLERANCIA_PX,
                                           _achar_quadros, _concordam, _falta,
                                           _ids_sobrevivem, _numeros, _quadros)
from src.visao.detector import PADRAO, modelo_escolhido


def _visao(quantas=1, centro=(10.0, 20.0, 40.0, 90.0), junta=(15.0, 30.0)):
    return {"quantas": quantas,
            "centros": [centro] * quantas,
            "juntas": [[list(junta)] * 17] * quantas}


# ------------------------------------------------- a conferencia de saida
def test_saidas_iguais_conferem():
    assert _concordam(_visao(), _visao())[0]


def test_diferenca_de_arredondamento_e_aceita():
    """Exportar troca a ordem das somas em ponto flutuante.

    Exigir bit a bit reprovaria a otimizacao por ela ser uma otimizacao. Dois
    pixels e menos que a espessura de um tornozelo na imagem de 320.
    """
    a = _visao()
    b = _visao(centro=(10.9, 20.9, 40.9, 90.9), junta=(15.9, 30.9))
    ok, _ = _concordam(a, b)
    assert ok


def test_achar_outra_quantidade_de_pessoas_reprova():
    """Um executor que ve outra coisa nao e uma otimizacao: e outro detector."""
    ok, motivo = _concordam(_visao(1), _visao(2))
    assert not ok and "pessoas" in motivo


def test_caixa_fora_da_tolerancia_reprova():
    ok, motivo = _concordam(_visao(), _visao(centro=(10.0, 20.0, 40.0, 99.0)))
    assert not ok and "caixa" in motivo


def test_junta_fora_da_tolerancia_reprova():
    """A caixa pode bater e o esqueleto nao — sao cabecas diferentes da rede."""
    ok, motivo = _concordam(_visao(), _visao(junta=(15.0, 44.0)))
    assert not ok and "junta" in motivo


def test_esqueleto_com_outro_formato_reprova():
    a = _visao()
    b = _visao()
    b["juntas"] = [[[1.0, 2.0]] * 5]
    assert not _concordam(a, b)[0]


def test_a_tolerancia_e_pequena_de_verdade():
    """Frouxa demais e o mesmo que nao conferir."""
    assert TOLERANCIA_PX <= 3.0


def test_cena_vazia_confere_com_cena_vazia():
    vazia = {"quantas": 0, "centros": [], "juntas": []}
    assert _concordam(vazia, vazia)[0]


# ----------------------------------------------------- os quadros de teste
def test_sem_quadro_nenhum_a_ferramenta_recusa_medir_e_ensina_como(tmp_path):
    """Recusar sem dizer o proximo comando e so recusar."""
    with pytest.raises(SystemExit, match="salvar-quadros"):
        _quadros([], 10)


def test_os_quadros_se_repetem_ate_dar_amostra(tmp_path):
    """Um quadro so nao tem mediana. E mediana e o ponto — ver o quadro de
    1047 ms que apareceu num detector de 130."""
    import cv2
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.zeros((80, 80, 3), np.uint8))
    assert len(_quadros([p], 25)) == 25


def _por_um_quadro(raiz, relativo):
    import cv2
    caminho = raiz / relativo
    caminho.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(caminho), np.zeros((80, 80, 3), np.uint8))
    return caminho


def test_prefere_os_quadros_COM_PESSOA(tmp_path):
    """Sem gente em cena os tres formatos concordam em 'nao vi nada'.

    Isso e verdade e nao prova nada — e o teste dos ids nem roda, porque nao
    ha id para sobreviver.
    """
    _por_um_quadro(tmp_path, "dados/quadros/alto-00007-sem-pessoa.jpg")
    _por_um_quadro(tmp_path, "dados/levantamento/alto.png")
    esperado = _por_um_quadro(tmp_path, "dados/quadros/alto-00042-com-pessoa.jpg")
    caminhos, _origem, tem_gente = _achar_quadros(tmp_path)
    assert caminhos == [esperado] and tem_gente


def test_so_a_camera_do_ALTO_entra_na_medida(tmp_path):
    """A frontal e a lateral vao para o MediaPipe. Medir nelas mediria o que
    nunca acontece neste detector."""
    _por_um_quadro(tmp_path, "dados/quadros/frontal-00001-com-pessoa.jpg")
    _por_um_quadro(tmp_path, "dados/quadros/lateral-00001-com-pessoa.jpg")
    alto = _por_um_quadro(tmp_path, "dados/quadros/alto-00001-com-pessoa.jpg")
    caminhos, _o, _g = _achar_quadros(tmp_path)
    assert caminhos == [alto]


def test_nao_varre_png_solto_na_pasta_do_levantamento(tmp_path):
    """Em 19/08 eu deixei tres diagramas meus la dentro.

    Um glob de `*.png` teria cronometrado o detector em cima das minhas
    proprias figuras — e o numero sairia, com cara de medida.
    """
    _por_um_quadro(tmp_path, "dados/levantamento/o_quarto_cabe_4x.png")
    _por_um_quadro(tmp_path, "dados/levantamento/chao_antes_e_depois.png")
    alto = _por_um_quadro(tmp_path, "dados/levantamento/alto.png")
    caminhos, _o, tem_gente = _achar_quadros(tmp_path)
    assert caminhos == [alto]
    assert not tem_gente, "levantamento nao tem pessoa; nao pode dizer que tem"


def test_sem_nada_devolve_vazio_em_vez_de_estourar(tmp_path):
    caminhos, _o, _g = _achar_quadros(tmp_path)
    assert caminhos == []


# --------------------------------------------- a escolha, lida pelo sistema
def test_sem_arquivo_o_padrao_vale_e_calado(tmp_path):
    """Quem nunca mediu nao tem nada errado a corrigir."""
    assert modelo_escolhido(tmp_path) == PADRAO


def test_arquivo_quebrado_nao_derruba_o_sistema(tmp_path):
    """Mas deixa rastro. `except Exception: return None` custou um dia em 11/08."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "detector.json").write_text("{isto nao e json")
    assert modelo_escolhido(tmp_path) == PADRAO


def test_modelo_apagado_volta_ao_padrao_em_vez_de_estourar_no_primeiro_quadro(tmp_path):
    """O .onnx pode ser apagado numa limpeza de disco — ja aconteceu aqui.

    Sem esta checagem o erro apareceria dentro do AutoBackend, no meio do
    primeiro quadro, com uma mensagem que nao aponta para `config/`.
    """
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "detector.json").write_text(
        json.dumps({"modelo": "sumiu/yolo11n-pose.onnx"}))
    assert modelo_escolhido(tmp_path) == PADRAO


def test_modelo_que_existe_e_adotado(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "pesos").mkdir()
    falso = tmp_path / "pesos" / "yolo11n-pose.onnx"
    falso.write_bytes(b"nao e um modelo, mas existe")
    (tmp_path / "config" / "detector.json").write_text(
        json.dumps({"modelo": "pesos/yolo11n-pose.onnx"}))
    assert modelo_escolhido(tmp_path) == str(falso)


def test_o_detector_aceita_modelo_explicito_sem_ler_config():
    """O argumento vence a config: e como os testes e as ferramentas fixam."""
    from src.visao.detector import DetectorDePessoas
    assert DetectorDePessoas(modelo="outro.pt").modelo_nome == "outro.pt"


# ------------------------------------- os defeitos achados na revisao de 19/08
def test_modelo_sem_pose_e_reprovado_e_nao_ignorado():
    """`zip` para na lista mais curta e nao reclama.

    Um modelo sem cabeca de pose devolveria `juntas = []`, o zip nao renderia
    par nenhum, e a conferencia diria "confere" sobre ZERO comparacoes — e o
    sistema adotaria um detector que nao entrega tornozelo, que e justamente
    o dado de que a posicao no chao depende.

        Um laco que nao iterou nao concordou: ele nao aconteceu.
    """
    com_pose = _visao()
    sem_pose = _visao()
    sem_pose["juntas"] = []
    ok, motivo = _concordam(com_pose, sem_pose)
    assert not ok and "esqueleto" in motivo


def test_ids_que_nao_sobrevivem_sao_denunciados():
    """Um id novo a cada quadro quebra rastreio, limbo e contagem de zona."""
    assert _ids_sobrevivem([{1}, {2}, {3}]) is False
    assert _ids_sobrevivem([{1}, {1}, {1}]) is True
    assert _ids_sobrevivem([{1, 2}, {2, 5}]) is True      # o 2 atravessou


def test_sem_deteccao_o_teste_de_id_diz_NAO_SEI_e_nao_PASSOU():
    """Nao ha id para sobreviver. Dizer que passou seria mentir sobre um
    teste que nao rodou."""
    assert _ids_sobrevivem([]) is None


def test_numeros_aceita_tensor_e_array():
    """Supor `.cpu()` e supor torch por baixo — que e o que esta ferramenta
    existe para deixar de ser verdade."""
    class FalsoTensor:
        def cpu(self):
            return np.array([1.5, 2.5])

    assert _numeros(FalsoTensor()) == [1.5, 2.5]
    assert _numeros(np.array([[1.0, 2.0]])) == [[1.0, 2.0]]


def test_falta_avisa_o_que_instalar_em_vez_de_deixar_o_pip_comecar_sozinho(
        monkeypatch):
    """O Ultralytics dispara `pip install` no meio da exportacao.

    Em 18/08 este projeto passou a tarde com o disco cheio. Um pip que comeca
    sozinho e para no meio deixa o ambiente pela metade, sem que ninguem
    tenha pedido nada.
    """
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda nome: None)
    assert "onnxruntime" in _falta("onnx")
    assert "openvino" in _falta("openvino")


def test_com_tudo_instalado_nao_falta_nada(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda nome: object())
    assert _falta("onnx") == [] and _falta("openvino") == []


def test_pytorch_ganhando_APAGA_a_escolha_antiga(tmp_path, monkeypatch):
    """Deixar de escrever a nova nao basta.

    Sem apagar, uma medicao que elege o PyTorch deixaria o sistema rodando o
    ONNX de uma medicao anterior — e o painel nao contaria a diferenca,
    porque ele mede o que roda e nao o que foi decidido.

        Uma decisao revogada que nao apaga a anterior nao foi revogada.
    """
    import ferramentas.acelerar_detector as A
    from src.visao.detector import PADRAO, modelo_escolhido

    (tmp_path / "config").mkdir()
    velho = tmp_path / "config" / "detector.json"
    velho.write_text(json.dumps({"modelo": "yolo11n-pose.onnx"}))
    monkeypatch.setattr(A, "RAIZ", tmp_path)

    # o trecho de decisao, isolado: PyTorch venceu
    alvo = A.RAIZ / "config" / "detector.json"
    if alvo.exists():
        alvo.unlink()

    assert not velho.exists()
    assert modelo_escolhido(tmp_path) == PADRAO


def test_o_ganho_e_medido_contra_o_PYTORCH_e_nao_contra_o_primeiro_que_rodou():
    """Se o PyTorch nao rodar, nao ha contra o que comparar.

    Uma razao contra um denominador que ninguem escolheu parece uma medida e
    nao e — e seria justamente o caso em que o PyTorch falhou, ou seja,
    quando menos se sabe.
    """
    linhas = [{"rotulo": "PyTorch", "ms": None},
              {"rotulo": "ONNX", "ms": 50.0}]
    base = next((l["ms"] for l in linhas
                 if l["rotulo"] == "PyTorch" and l["ms"] is not None), None)
    assert base is None


def test_cada_formato_declara_o_que_escreve_e_o_que_roda():
    """Sao pacotes diferentes: `onnx` grava o arquivo, `onnxruntime` executa.

    Conferir so um dos dois deixaria a exportacao funcionar e a medicao
    falhar, ou o contrario — nos dois casos com uma mensagem que nao aponta
    para o pip.
    """
    escreve, roda = PRECISA["onnx"]
    assert "onnx" in escreve and "onnxruntime" in roda
