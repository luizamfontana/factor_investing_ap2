"""
Carteira de Factor Investing vs Ibovespa
Formação: 28/03/2024 | Backtest: 29/03/2024 a 01/07/2026

Como usar:
  1) pip install yfinance pandas numpy matplotlib
  2) python carteira_factor.py
Se o Yahoo não funcionar, coloque um arquivo "precos.csv" na mesma pasta
(coluna Date + uma coluna por ticker SEM ".SA" + coluna IBOV) e o script lê dele.
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1) PARÂMETROS
# ----------------------------------------------------------------------
DATA_FORMACAO = "2024-03-28"   # preço de compra = fechamento deste dia
DATA_FIM = "2026-07-01"        # fechamento final do backtest
DIAS_ANO = 252                 # pregões por ano (sem taxa livre de risco: Sharpe e alfa usam o retorno total)

# Empresas de cada fator (retiradas do documento do grupo)
FATOR_1 = ["SLCE3", "CMIG4", "SBSP3", "PETR4", "FLRY3", "DXCO3", "EGIE3", "KLBN11"]   # 8 empresas
FATOR_2 = ["TAEE11", "EGIE3", "VIVT3", "BBSE3", "ITSA4", "ABEV3", "TIMS3", "EQTL3", "ITUB4", "KLBN11"]  # 10
# AXIA3 = antiga ELET3 (Eletrobras mudou de nome e ticker em 10/11/2025; o Yahoo só tem o histórico sob AXIA3)
FATOR_3 = ["ITUB4", "ABEV3", "AXIA3", "BBAS3", "CMIG4", "RENT3"]  # só 6 posições preenchidas no documento


# ----------------------------------------------------------------------
# 2) PESOS DA CARTEIRA
# ----------------------------------------------------------------------
def pesos_do_fator(tickers):
    """Pesos iguais dentro de um fator (ex.: 8 empresas -> 12,5% cada)."""
    return pd.Series(1 / len(tickers), index=tickers)


def montar_carteira_combinada():
    """Cada fator vale 1/3 da carteira; empresa repetida em dois fatores soma os pesos."""
    p1, p2, p3 = pesos_do_fator(FATOR_1), pesos_do_fator(FATOR_2), pesos_do_fator(FATOR_3)
    combinada = p1.mul(1 / 3).add(p2.mul(1 / 3), fill_value=0).add(p3.mul(1 / 3), fill_value=0)
    return p1, p2, p3, combinada


# ----------------------------------------------------------------------
# 3) PREÇOS
# ----------------------------------------------------------------------
def carregar_precos(tickers):
    """Lê precos.csv se existir; senão baixa do Yahoo (preços ajustados por dividendos)."""
    if os.path.exists("precos.csv"):
        df = pd.read_csv("precos.csv", index_col="Date", parse_dates=True)
    else:
        import yfinance as yf
        simbolos = [t + ".SA" for t in tickers] + ["^BVSP"]
        df = yf.download(simbolos, start="2024-03-20", end="2026-07-03", auto_adjust=True)["Close"]
        df.columns = [c.replace(".SA", "").replace("^BVSP", "IBOV") for c in df.columns]
    df = df.sort_index().loc[DATA_FORMACAO:DATA_FIM]
    return df


# ----------------------------------------------------------------------
# 4) VALOR DA CARTEIRA (compra e mantém, sem rebalanceamento)
# ----------------------------------------------------------------------
def valor_da_carteira(precos, pesos):
    """Começa em 1,00 na formação. Cada ativo cresce com seu preço; o peso inicial vira o tamanho da posição."""
    base = precos[pesos.index] / precos[pesos.index].iloc[0]
    return (base * pesos).sum(axis=1)


# ----------------------------------------------------------------------
# 5) MÉTRICAS
# ----------------------------------------------------------------------
def retorno_acumulado(serie):
    return serie.iloc[-1] / serie.iloc[0] - 1


def volatilidade_anual(serie):
    return serie.pct_change().dropna().std() * np.sqrt(DIAS_ANO)


def sharpe(serie):
    r = serie.pct_change().dropna()
    return (r.mean() * DIAS_ANO) / (r.std() * np.sqrt(DIAS_ANO))


def alfa_anual(serie, ibov):
    """Alfa: regressão do retorno diário da carteira contra o do Ibovespa (o intercepto é o alfa)."""
    rc = serie.pct_change().dropna()
    rm = ibov.pct_change().dropna()
    beta, alfa_dia = np.polyfit(rm, rc, 1)
    return alfa_dia * DIAS_ANO, beta


# ----------------------------------------------------------------------
# 6) EXECUÇÃO
# ----------------------------------------------------------------------
def main():
    p1, p2, p3, combinada = montar_carteira_combinada()
    print("Pesos da carteira combinada:")
    print((combinada.sort_values(ascending=False) * 100).round(2).astype(str) + "%")
    print("Soma dos pesos:", round(combinada.sum(), 6))

    todos = sorted(set(FATOR_1 + FATOR_2 + FATOR_3))
    precos = carregar_precos(todos)
    print(f"\nPeríodo: {precos.index[0].date()} a {precos.index[-1].date()} ({len(precos)} pregões)")

    ibov = precos["IBOV"] / precos["IBOV"].iloc[0]
    series = {
        "Fator 1": valor_da_carteira(precos, p1),
        "Fator 2": valor_da_carteira(precos, p2),
        "Fator 3": valor_da_carteira(precos, p3),
        "Carteira combinada": valor_da_carteira(precos, combinada),
        "Ibovespa": ibov,
    }

    linhas = {}
    for nome, s in series.items():
        a, b = (alfa_anual(s, ibov) if nome != "Ibovespa" else (0.0, 1.0))
        linhas[nome] = {
            "Retorno acumulado": retorno_acumulado(s),
            "Volatilidade anualizada": volatilidade_anual(s),
            "Sharpe": sharpe(s),
            "Alfa anualizado": a,
            "Beta": b,
        }
    tabela = pd.DataFrame(linhas)
    print("\n", tabela.round(4))

    tabela.to_csv("resultados.csv")
    pd.DataFrame(series).to_csv("evolucao_carteiras.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.DataFrame(series).plot(figsize=(10, 5), title="Carteiras por fator vs Ibovespa (base 1,00 em 28/03/2024)")
    plt.ylabel("Valor da carteira")
    plt.tight_layout()
    plt.savefig("grafico.png", dpi=150)


if __name__ == "__main__":
    main()