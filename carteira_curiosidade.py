"""
CURIOSIDADE (análise ex-post, fora do resultado principal)
Mesma carteira combinada do trabalho, mas entrando em outras datas.

Atenção para o texto do trabalho: a lista de empresas foi escolhida com dados até 31/12/2023.
Por isso só fazem sentido datas de entrada DEPOIS de 28/03/2024 (antes disso haveria look-ahead).
Esta análise mostra todas as datas, boas e ruins, e não substitui o backtest oficial.

Uso: coloque este arquivo na mesma pasta de carteira.py (e do precos.csv, se usar) e rode:
  python curiosidade_datas.py
"""

import pandas as pd
from carteira import montar_carteira_combinada, carregar_precos, valor_da_carteira, retorno_acumulado
from carteira import FATOR_1, FATOR_2, FATOR_3

DATA_FIM = "2026-07-01"


def retorno_no_periodo(precos, pesos, inicio, fim):
    """Retorno da carteira (pesos reajustados na data de entrada) e do Ibovespa entre duas datas."""
    recorte = precos.loc[inicio:fim]
    cart = retorno_acumulado(valor_da_carteira(recorte, pesos))
    ibov = recorte["IBOV"].iloc[-1] / recorte["IBOV"].iloc[0] - 1
    return cart, ibov


def tabela_entradas(precos, pesos, datas_entrada, fim):
    """Uma linha por data de entrada, sempre terminando na mesma data final."""
    linhas = []
    for d in datas_entrada:
        cart, ibov = retorno_no_periodo(precos, pesos, d, fim)
        linhas.append({"Entrada": d.date(), "Fim": pd.Timestamp(fim).date(),
                       "Carteira": cart, "Ibovespa": ibov, "Diferença": cart - ibov,
                       "Bateu?": "sim" if cart > ibov else "não"})
    return pd.DataFrame(linhas)


def tabela_janelas_12m(precos, pesos, datas_entrada):
    """Janelas de 12 meses a partir de cada data de entrada (só as que cabem no período)."""
    linhas = []
    for d in datas_entrada:
        fim = d + pd.DateOffset(months=12)
        if fim > precos.index[-1]:
            break
        cart, ibov = retorno_no_periodo(precos, pesos, d, fim)
        linhas.append({"Entrada": d.date(), "Fim": fim.date(), "Carteira": cart, "Ibovespa": ibov,
                       "Diferença": cart - ibov, "Bateu?": "sim" if cart > ibov else "não"})
    return pd.DataFrame(linhas)


def main():
    _, _, _, pesos = montar_carteira_combinada()
    precos = carregar_precos(sorted(set(FATOR_1 + FATOR_2 + FATOR_3)))
    precos = precos.loc[:DATA_FIM]

    # primeira data de cada mês, a partir da formação oficial
    datas = precos.groupby([precos.index.year, precos.index.month]).apply(lambda g: g.index[0]).tolist()
    datas = [pd.Timestamp(d) for d in datas]

    t1 = tabela_entradas(precos, pesos, datas[:-3], DATA_FIM)
    t2 = tabela_janelas_12m(precos, pesos, datas)

    pd.set_option("display.width", 120)
    print("\n== Entrando em cada mês e saindo em", DATA_FIM, "==")
    print(t1.round(4).to_string(index=False))
    print(f"\nBateu o Ibovespa em {int((t1['Bateu?'] == 'sim').sum())} de {len(t1)} datas de entrada.")

    print("\n== Janelas de 12 meses ==")
    print(t2.round(4).to_string(index=False))
    print(f"\nBateu o Ibovespa em {int((t2['Bateu?'] == 'sim').sum())} de {len(t2)} janelas.")

    t1.to_csv("curiosidade_entradas.csv", index=False)
    t2.to_csv("curiosidade_janelas_12m.csv", index=False)


if __name__ == "__main__":
    main()