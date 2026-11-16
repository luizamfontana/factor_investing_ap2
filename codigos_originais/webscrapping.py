# -*- coding: utf-8 -*-
"""
RANKING DE CAPITAL HUMANO POR PRODUTIVIDADE (dados abertos da CVM)

Como rodar:   py hcs_produtividade.py
Resultado:    pasta output/ (arquivos CSV, CSV para Excel pt-BR e Parquet) + tabela no terminal

Indicador único: produtividade do trabalho = receita / nº de empregados.
Fluxo, para cada ano do PERIODO:
    montar_base()  ->  calcular_ano()  (produtividade e z-score setorial)
Depois:
    calcular_periodo()  ->  média dos anos  ->  tabela_top()  ->  salvar()
"""
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# =============================================================================
# PARÂMETROS (tudo o que você pode mudar fica aqui)
# =============================================================================
PERIODO = [2022, 2023]          # anos analisados; o indicador final é a média entre eles
MIN_EMPREGADOS = 100            # empresas com menos empregados ficam sem nota
TOP_N = 10                      # quantas empresas entram na tabela de resultados
PASTA_CACHE = Path("cache_cvm")
PASTA_SAIDA = Path("output")
ARQ_EMPREGADOS_MANUAL = Path("empregados_manual.csv")   # opcional: colunas ticker, ano, empregados

URL_DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{ano}.zip"
URL_FRE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"

# ticker: (setor, regex do nome na CVM sem acento e em maiúsculas, palavra-chave para sugestões)
EMPRESAS = {
    "ITUB4":  ("Financeiro", r"^ITAU UNIBANCO HOLDING", "ITAU"),
    "BBDC4":  ("Financeiro", r"^(BCO|BANCO) BRADESCO S\.?\s?A", "BRADESCO"),
    "BBAS3":  ("Financeiro", r"^(BCO|BANCO) (DO )?BRASIL S\.?\s?A", "BRASIL"),
    "SANB11": ("Financeiro", r"^(BCO|BANCO) SANTANDER \(BRASIL\)", "SANTANDER"),
    "SBSP3":  ("Utilidades Públicas", r"SABESP|SANEAMENTO BASICO.*SAO PAULO", "SANEAMENTO"),
    "CMIG4":  ("Utilidades Públicas", r"^(CIA|COMPANHIA) ENERGETICA DE MINAS GERAIS", "ENERGETICA DE MINAS"),
    "ELET3":  ("Utilidades Públicas", r"ELETROBRAS|AXIA ENERGIA|^CENTRAIS ELET(RICAS)? BRAS", "ELET"),
    "CPLE6":  ("Utilidades Públicas", r"^(CIA|COMPANHIA) PARANAENSE DE ENERGIA", "PARANAENSE"),
    "ABEV3":  ("Consumo", r"^AMBEV S", "AMBEV"),
    "LREN3":  ("Consumo", r"^LOJAS RENNER", "RENNER"),
    "MGLU3":  ("Consumo", r"^MAGAZINE LUIZA", "MAGAZINE"),
    "RENT3":  ("Consumo", r"^LOCALIZA RENT A CAR", "LOCALIZA"),
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
def normalizar(texto):
    """Entrada: texto qualquer.  Saída: texto em MAIÚSCULAS e sem acentos."""
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).upper().strip()


def para_numero(serie):
    """Entrada: Series de texto ('1234.5' ou '1.234,5').  Saída: Series numérica (float)."""
    serie = serie.astype(str).str.strip()
    com_virgula = serie.str.contains(",", regex=False)
    serie = serie.where(~com_virgula, serie.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(serie, errors="coerce")


def achar_coluna(df, regex):
    """Entrada: DataFrame e regex.  Saída: nome da 1ª coluna que casa com o regex (ou None)."""
    for coluna in df.columns:
        if re.search(regex, normalizar(coluna)):
            return coluna
    return None


def baixar_zip(url, destino):
    """Entrada: url e caminho de destino.  Saída: caminho do ZIP (reaproveita se já existe)."""
    destino = Path(destino)
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} ...")
    resposta = requests.get(url, timeout=300, headers={"User-Agent": "hcs-pesquisa/1.0"})
    resposta.raise_for_status()
    destino.write_bytes(resposta.content)
    return destino


def ler_csv_do_zip(arquivo_zip, trecho_do_nome):
    """Entrada: caminho do ZIP e um trecho do nome do CSV.  Saída: DataFrame (ou None se não achar)."""
    with zipfile.ZipFile(arquivo_zip) as zf:
        achados = sorted(n for n in zf.namelist() if trecho_do_nome.lower() in n.lower())
        if not achados:
            return None
        with zf.open(achados[0]) as f:
            return pd.read_csv(f, sep=";", encoding="latin-1", dtype=str)


# =============================================================================
# 1) COLETA DE DADOS
# =============================================================================
def coletar_receita(ano):
    """
    Lê a receita na DVA (DFP) da CVM.
    Entrada: ano do exercício (ex.: 2023).
    Saída: DataFrame com [cnpj, nome, ano, receita] em R$, uma linha por empresa (receita = DVA conta 7.01).
    """
    zip_dfp = baixar_zip(URL_DFP.format(ano=ano), PASTA_CACHE / f"dfp_{ano}.zip")
    partes = []
    for escopo in ("con", "ind"):                       # consolidado e individual
        df = ler_csv_do_zip(zip_dfp, f"DVA_{escopo}_{ano}")
        if df is not None:
            df["escopo"] = escopo
            partes.append(df)
    dva = pd.concat(partes, ignore_index=True)

    dva = dva[dva["ORDEM_EXERC"].map(normalizar).str.startswith("ULT")].copy()      # exercício atual
    escala = np.where(dva["ESCALA_MOEDA"].map(normalizar).str.startswith("MIL"), 1000, 1)
    dva["valor"] = para_numero(dva["VL_CONTA"]) * escala                            # tudo em R$
    dva["VERSAO"] = pd.to_numeric(dva["VERSAO"], errors="coerce")
    dva = dva[dva["VERSAO"] == dva.groupby(["CNPJ_CIA", "escopo"])["VERSAO"].transform("max")]  # última versão
    tem_consolidado = dva.groupby("CNPJ_CIA")["escopo"].transform(lambda s: (s == "con").any())
    dva = dva[(dva["escopo"] == "con") | ~tem_consolidado]                          # prefere consolidado

    receita = dva[dva["CD_CONTA"] == "7.01"].groupby("CNPJ_CIA")["valor"].first().rename("receita")
    nome = dva.groupby("CNPJ_CIA")["DENOM_CIA"].first().rename("nome")
    saida = pd.concat([nome, receita], axis=1).reset_index().rename(columns={"CNPJ_CIA": "cnpj"})
    saida["ano"] = ano
    return saida


def coletar_empregados(ano_fre):
    """
    Lê o nº de empregados no Formulário de Referência (FRE, item 10.1A).
    Entrada: ano de ENTREGA do FRE (o FRE de 2024 reporta o exercício 2023).
    Saída: DataFrame com [cnpj, ano, empregados]; vazio se o arquivo não existir.
    O arquivo traz uma coluna de quantidade por região (Norte, Nordeste, ..., Exterior);
    o total de empregados é a soma de todas elas, em todas as linhas (posições) do documento mais recente.
    Só existe a partir do FRE de 2023 (exercício 2022); antes disso o layout é outro e não é lido.
    """
    vazio = pd.DataFrame(columns=["cnpj", "ano", "empregados"])
    try:
        zip_fre = baixar_zip(URL_FRE.format(ano=ano_fre), PASTA_CACHE / f"fre_{ano_fre}.zip")
        df = ler_csv_do_zip(zip_fre, "empregado_posicao_local")
        if df is None:
            print(f"[FRE {ano_fre}] não há arquivo de empregados neste ano (layout antigo): exercício {ano_fre - 1} sem dado.")
            return vazio
        c_cnpj, c_data = achar_coluna(df, "CNPJ"), achar_coluna(df, "DATA.*REFER|DT_REFER")
        c_versao = achar_coluna(df, "^VERS")
        c_qtds = [c for c in df.columns if normalizar(c).startswith("QUANTIDADE")]      # uma por região
        print(f"[FRE {ano_fre}] somando as colunas: {c_qtds}")
        chave = df[c_data].fillna("") + "|" + df[c_versao].fillna("0").str.zfill(4)
        df = df[chave == chave.groupby(df[c_cnpj]).transform("max")]               # documento mais recente
        por_linha = df[c_qtds].apply(para_numero).sum(axis=1, min_count=1)          # total da linha
        total = por_linha.groupby(df[c_cnpj]).sum().rename("empregados")
        saida = total.reset_index().rename(columns={c_cnpj: "cnpj"})
        saida["ano"] = ano_fre - 1
        return saida
    except Exception as erro:                       # não derruba o script: o aviso aparece e segue
        print(f"[FRE {ano_fre}] não foi possível ler os empregados: {erro}")
        return vazio


def identificar_empresas(financeiro):
    """
    Liga cada ticker ao CNPJ da CVM pelo nome da empresa (regex em EMPRESAS).
    Entrada: DataFrame de coletar_receita (precisa de cnpj e nome).
    Saída: dict {ticker: cnpj}. Tickers que não casam aparecem no terminal, com nomes parecidos.
    """
    nomes = financeiro[["cnpj", "nome"]].drop_duplicates()
    nomes["n"] = nomes["nome"].map(normalizar)
    mapa = {}
    for ticker, (_, regex, palavra) in EMPRESAS.items():
        achados = nomes[nomes["n"].map(lambda n: bool(re.search(regex, n)))]
        if achados.empty:
            parecidos = nomes.loc[nomes["n"].str.contains(palavra, regex=False), "nome"].head(5).tolist()
            print(f"[AVISO] {ticker}: nenhuma empresa casou com o regex. Ajuste em EMPRESAS. "
                  f"Nomes com '{palavra}' no arquivo: {parecidos}")
            continue
        escolhida = achados.loc[achados["n"].str.len().idxmin()]          # nome mais curto se houver dúvida
        mapa[ticker] = escolhida["cnpj"]
        print(f"{ticker} -> {escolhida['nome']}")
    return mapa


def montar_base(ano):
    """
    Junta receita e empregados de UM ano numa tabela.
    Entrada: ano do exercício.
    Saída: DataFrame com [ticker, empresa, setor, ano, receita, empregados], uma linha por ticker
           (dados faltantes ficam NaN, então nenhuma empresa some da tabela).
    """
    financeiro = coletar_receita(ano)
    mapa = identificar_empresas(financeiro)
    empregados = coletar_empregados(ano + 1)                     # o FRE de ano+1 reporta o exercício 'ano'

    nome_por_cnpj = financeiro.drop_duplicates("cnpj").set_index("cnpj")["nome"]
    cadastro = pd.DataFrame([{"ticker": t, "setor": EMPRESAS[t][0], "cnpj": mapa.get(t)} for t in EMPRESAS])
    cadastro["empresa"] = cadastro["cnpj"].map(nome_por_cnpj).fillna("n/d")        # razão social na CVM
    cadastro["ano"] = ano
    base = (cadastro.merge(financeiro[["cnpj", "receita"]], on="cnpj", how="left")
            .merge(empregados[["cnpj", "empregados"]], on="cnpj", how="left"))
    base["empregados"] = pd.to_numeric(base["empregados"], errors="coerce")

    # Opcional: empregados_manual.csv (ticker, ano, empregados) tem prioridade sobre o FRE
    if ARQ_EMPREGADOS_MANUAL.exists():
        manual = pd.read_csv(ARQ_EMPREGADOS_MANUAL, sep=None, engine="python")
        manual = manual[manual["ano"] == ano].rename(columns={"empregados": "emp_manual"})
        base = base.merge(manual[["ticker", "emp_manual"]], on="ticker", how="left")
        base["empregados"] = base["emp_manual"].combine_first(base["empregados"])
        base = base.drop(columns="emp_manual")
    return base[["ticker", "empresa", "setor", "ano", "receita", "empregados"]]


# =============================================================================
# 2) INDICADOR DE PRODUTIVIDADE
# =============================================================================
def winsorizar(serie):
    """Entrada: Series.  Saída: Series com extremos cortados nos percentis 1% e 99% (precisa de 5+ valores)."""
    if serie.notna().sum() < 5:
        return serie
    return serie.clip(serie.quantile(0.01), serie.quantile(0.99))


def zscore_setorial(df, coluna):
    """
    Entrada: DataFrame (com coluna 'setor') e nome da coluna numérica.
    Saída: Series com (valor - média do setor) / desvio-padrão do setor.
    """
    grupo = df.groupby("setor")[coluna]
    return (df[coluna] - grupo.transform("mean")) / grupo.transform("std")


def calcular_ano(ano):
    """
    Calcula o indicador de UM ano.
    Entrada: ano do exercício.
    Saída: DataFrame com [ticker, empresa, setor, ano, receita, empregados, produtividade, z_produtividade]
        produtividade   = receita / empregados (R$ por empregado); vazia se empregados < MIN_EMPREGADOS
        z_produtividade = z-score setorial do log da produtividade, winsorizado em 1% e 99%
    """
    b = montar_base(ano)
    b.loc[b["empregados"] < MIN_EMPREGADOS, "empregados"] = np.nan          # filtro de qualidade
    b["produtividade"] = b["receita"] / b["empregados"]
    log_prod = np.log(b["produtividade"].where(b["produtividade"] > 0))     # log: ratio é muito assimétrico
    b["log_produtividade"] = winsorizar(log_prod)
    b["z_produtividade"] = zscore_setorial(b, "log_produtividade")
    return b.drop(columns="log_produtividade")


# =============================================================================
# 3) PERÍODO E RANKING
# =============================================================================
def calcular_periodo(anos):
    """
    Roda o indicador para cada ano e tira a média.
    Entrada: lista de anos (ex.: [2022, 2023]).
    Saída: dois DataFrames
        por_ano: resultado de calcular_ano para cada ano, empilhado
        media:   uma linha por ticker com produtividade_media (R$/empregado), z_produtividade_medio
                 (média dos z-scores dos anos com dado), anos_com_dado e ranking (1 = maior produtividade)
    """
    por_ano = pd.concat([calcular_ano(a) for a in anos], ignore_index=True)
    grupo = por_ano.groupby(["ticker", "empresa", "setor"])
    media = pd.DataFrame({
        "produtividade_media": grupo["produtividade"].mean(),
        "z_produtividade_medio": grupo["z_produtividade"].mean(),
        "anos_com_dado": grupo["z_produtividade"].count(),
    }).reset_index()
    media["ranking"] = media["z_produtividade_medio"].rank(ascending=False, method="min")
    media = media.sort_values("z_produtividade_medio", ascending=False, na_position="last").reset_index(drop=True)
    return por_ano, media


def tabela_top(media, n=TOP_N):
    """
    Monta a tabela de resultados no formato do trabalho.
    Entrada: DataFrame 'media' de calcular_periodo e a quantidade de empresas (n).
    Saída: DataFrame com [Posição, Empresa, Ticker, Indicador], só com as n maiores notas
           (Indicador = z-score setorial médio da produtividade; empresas sem nota ficam de fora).
    """
    top = media.dropna(subset=["z_produtividade_medio"]).head(n)
    return pd.DataFrame({"Posição": range(1, len(top) + 1),
                         "Empresa": top["empresa"].values,
                         "Ticker": top["ticker"].values,
                         "Indicador": top["z_produtividade_medio"].round(3).values})


# =============================================================================
# 4) SAÍDA
# =============================================================================
def salvar(resultado, nome):
    """
    Entrada: DataFrame e o nome-base dos arquivos (ex.: "produtividade_periodo").
    Saída: 3 arquivos em output/ -> <nome>.csv, <nome>_excel.csv (sep ';' e decimal ',') e <nome>.parquet.
    """
    PASTA_SAIDA.mkdir(exist_ok=True)
    resultado.to_csv(PASTA_SAIDA / f"{nome}.csv", index=False, encoding="utf-8-sig")
    resultado.to_csv(PASTA_SAIDA / f"{nome}_excel.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    try:
        resultado.to_parquet(PASTA_SAIDA / f"{nome}.parquet", index=False)
    except Exception as erro:
        print(f"[AVISO] Parquet não salvo ({erro}). Instale pyarrow.")
    print(f"Arquivos salvos em {PASTA_SAIDA.resolve()}")


# =============================================================================
if __name__ == "__main__":
    por_ano, media = calcular_periodo(PERIODO)          # 1-3) coleta, indicador por ano e média
    top = tabela_top(media)                             # 4) tabela de resultados
    salvar(por_ano, "produtividade_por_ano")
    salvar(media, "produtividade_periodo")
    salvar(top, "top10")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n===== PRODUTIVIDADE MÉDIA DO PERÍODO =====")
    print(media.round(3).to_string(index=False))
    print(f"\n===== TOP {TOP_N} (tabela de resultados) =====")
    print(top.to_string(index=False))