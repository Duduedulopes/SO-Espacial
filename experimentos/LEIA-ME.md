# Experimentos

Programas que cumpriram seu papel e foram superados. Ficam aqui porque estao
citados no caderno de laboratorio — apagar quebraria o registro.

- `detectar.py`   primeira deteccao com YOLO. Mediu o custo de inferencia
                  (87 ms) e revelou que a base da caixa nao marca os pes.
- `rastrear.py`   comparou base da caixa contra tornozelo lado a lado.
                  Produziu o desacordo de 97 px e achou o primeiro fantasma.
- `kalman_1d.py`  Kalman escrito do zero, criterio de dominio do bloco 2.
                  Dados sinteticos, com verdade conhecida.

Nada aqui e importado pelo sistema. Rodam sozinhos.
