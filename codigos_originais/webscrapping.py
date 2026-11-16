# -*- coding: utf-8 -*-
"""
RANKING DE CAPITAL HUMANO POR PRODUTIVIDADE (dados abertos da CVM)
Universo: carteira do Ibovespa vigente de 04/09/2023 a 29/12/2023 (86 papéis de 83 empresas; um ticker por empresa).

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
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# =============================================================================
# PARÂMETROS (tudo o que você pode mudar fica aqui)
# =============================================================================
PERIODO = [2022, 2023]          # anos analisados; o indicador final é a média entre eles
MIN_EMPREGADOS = 100            # empresas com menos empregados ficam sem nota
MIN_SETOR = 3                   # setor com menos empresas válidas usa o z-score do universo inteiro
TOP_N = 10                      # quantas empresas entram na tabela de resultados
PASTA_CACHE = Path("cache_cvm")
PASTA_SAIDA = Path("output")
ARQ_EMPREGADOS_MANUAL = Path("empregados_manual.csv")   # opcional: colunas ticker, ano, empregados

URL_DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{ano}.zip"
URL_FRE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"

# Universo: Ibovespa de set-dez/2023, um ticker por empresa (ações ON/PN da mesma empresa contam uma vez).
# ticker: (setor, regex do nome da empresa na CVM, sem acento e em maiúsculas)
# Setores: classificação setorial da B3 (atribuída manualmente; confira se necessário).
EMPRESAS = {
    # --- Bens Industriais
    "AZUL4":  ("Bens Industriais", r"^AZUL S"),
    "CCRO3":  ("Bens Industriais", r"^CCR S"),
    "EMBR3":  ("Bens Industriais", r"^EMBRAER S"),
    "GOLL4":  ("Bens Industriais", r"^GOL LINHAS"),
    "RAIL3":  ("Bens Industriais", r"^RUMO S"),
    "VAMO3":  ("Bens Industriais", r"^VAMOS"),
    "WEGE3":  ("Bens Industriais", r"^WEG S"),
    # --- Comunicações
    "TIMS3":  ("Comunicações", r"^TIM S"),
    "VIVT3":  ("Comunicações", r"^TELEFONICA BRASIL"),
    # --- Consumo Cíclico
    "ALPA4":  ("Consumo Cíclico", r"^ALPARGATAS"),
    "ARZZ3":  ("Consumo Cíclico", r"^AREZZO|^AZZAS"),
    "COGN3":  ("Consumo Cíclico", r"^COGNA"),
    "CVCB3":  ("Consumo Cíclico", r"^CVC BRASIL"),
    "CYRE3":  ("Consumo Cíclico", r"^CYRELA"),
    "EZTC3":  ("Consumo Cíclico", r"^EZ ?TEC"),
    "LREN3":  ("Consumo Cíclico", r"^LOJAS RENNER"),
    "MGLU3":  ("Consumo Cíclico", r"^MAGAZINE LUIZA"),
    "MRVE3":  ("Consumo Cíclico", r"^MRV ENGENHARIA"),
    "PETZ3":  ("Consumo Cíclico", r"^PET CENTER|^PETZ"),
    "RENT3":  ("Consumo Cíclico", r"^LOCALIZA RENT A CAR"),
    "SOMA3":  ("Consumo Cíclico", r"^GRUPO DE MODA SOMA|^SOMA S"),
    "VIIA3":  ("Consumo Cíclico", r"^VIA S\.?\s?A|^VIA VAREJO|^GRUPO CASAS BAHIA"),
    "YDUQ3":  ("Consumo Cíclico", r"^YDUQS"),
    # --- Consumo não Cíclico
    "ABEV3":  ("Consumo não Cíclico", r"^AMBEV S"),
    "ASAI3":  ("Consumo não Cíclico", r"^SENDAS DISTRIBUIDORA"),
    "BEEF3":  ("Consumo não Cíclico", r"^MINERVA S"),
    "BRFS3":  ("Consumo não Cíclico", r"^BRF S"),
    "CRFB3":  ("Consumo não Cíclico", r"^ATACADAO"),
    "JBSS3":  ("Consumo não Cíclico", r"^JBS S"),
    "MRFG3":  ("Consumo não Cíclico", r"^MARFRIG"),
    "NTCO3":  ("Consumo não Cíclico", r"^NATURA"),
    "PCAR3":  ("Consumo não Cíclico", r"^(CIA|COMPANHIA) BRASILEIRA DE DISTRIBUICAO"),
    "SLCE3":  ("Consumo não Cíclico", r"^SLC AGRICOLA"),
    "SMTO3":  ("Consumo não Cíclico", r"^SAO MARTINHO"),
    # --- Financeiro
    "ALSO3":  ("Financeiro", r"^ALIANSCE"),
    "B3SA3":  ("Financeiro", r"^B3 S"),
    "BBAS3":  ("Financeiro", r"^(BCO|BANCO) (DO )?BRASIL S\.?\s?A"),
    "BBDC4":  ("Financeiro", r"^(BCO|BANCO) BRADESCO S\.?\s?A"),
    "BBSE3":  ("Financeiro", r"^BB SEGURIDADE"),
    "BPAC11": ("Financeiro", r"^(BCO|BANCO) BTG PACTUAL"),
    "BRAP4":  ("Financeiro", r"^BRADESPAR"),
    "CIEL3":  ("Financeiro", r"^CIELO"),
    "IGTI11": ("Financeiro", r"^IGUATEMI"),
    "IRBR3":  ("Financeiro", r"^IRB"),
    "ITSA4":  ("Financeiro", r"^ITAUSA"),
    "ITUB4":  ("Financeiro", r"^ITAU UNIBANCO HOLDING"),
    "MULT3":  ("Financeiro", r"^MULTIPLAN"),
    "SANB11": ("Financeiro", r"^(BCO|BANCO) SANTANDER \(BRASIL\)"),
    # --- Materiais Básicos
    "BRKM5":  ("Materiais Básicos", r"^BRASKEM S"),
    "CMIN3":  ("Materiais Básicos", r"^CSN MINERACAO"),
    "CSNA3":  ("Materiais Básicos", r"^(CIA|COMPANHIA) SIDERURGICA NACIONAL"),
    "DXCO3":  ("Materiais Básicos", r"^DEXCO|^DURATEX"),
    "GGBR4":  ("Materiais Básicos", r"^GERDAU S"),
    "GOAU4":  ("Materiais Básicos", r"^METALURGICA GERDAU"),
    "KLBN11": ("Materiais Básicos", r"^KLABIN S"),
    "SUZB3":  ("Materiais Básicos", r"^SUZANO S"),
    "USIM5":  ("Materiais Básicos", r"^USINAS SIDERURGICAS DE MINAS GERAIS"),
    "VALE3":  ("Materiais Básicos", r"^VALE S\.?\s?A"),
    # --- Petróleo, Gás e Biocombustíveis
    "CSAN3":  ("Petróleo, Gás e Biocombustíveis", r"^COSAN S"),
    "PETR4":  ("Petróleo, Gás e Biocombustíveis", r"^PETROLEO BRASILEIRO"),
    "PRIO3":  ("Petróleo, Gás e Biocombustíveis", r"^PETRO ?RIO|^PRIO S"),
    "RAIZ4":  ("Petróleo, Gás e Biocombustíveis", r"^RAIZEN S"),
    "RECV3":  ("Petróleo, Gás e Biocombustíveis", r"^PETRORECONCAVO"),
    "RRRP3":  ("Petróleo, Gás e Biocombustíveis", r"^3R PETROLEUM"),
    "UGPA3":  ("Petróleo, Gás e Biocombustíveis", r"^ULTRAPAR"),
    "VBBR3":  ("Petróleo, Gás e Biocombustíveis", r"^VIBRA ENERGIA"),
    # --- Saúde
    "FLRY3":  ("Saúde", r"^FLEURY"),
    "HAPV3":  ("Saúde", r"^HAPVIDA"),
    "HYPE3":  ("Saúde", r"^HYPERA"),
    "RADL3":  ("Saúde", r"^RAIA DROGASIL"),
    "RDOR3":  ("Saúde", r"^REDE D"),
    # --- Tecnologia da Informação
    "LWSA3":  ("Tecnologia da Informação", r"^LOCAWEB"),
    "TOTS3":  ("Tecnologia da Informação", r"^TOTVS"),
    # --- Utilidade Pública
    "CMIG4":  ("Utilidade Pública", r"^(CIA|COMPANHIA) ENERGETICA DE MINAS GERAIS"),
    "CPFE3":  ("Utilidade Pública", r"^CPFL ENERGIA"),
    "CPLE6":  ("Utilidade Pública", r"^(CIA|COMPANHIA) PARANAENSE DE ENERGIA"),
    "EGIE3":  ("Utilidade Pública", r"^ENGIE BRASIL"),
    "ELET3":  ("Utilidade Pública", r"ELETROBRAS|AXIA ENERGIA|^CENTRAIS ELET(RICAS)? BRAS"),
    "ENEV3":  ("Utilidade Pública", r"^ENEVA"),
    "ENGI11": ("Utilidade Pública", r"^ENERGISA S\.?\s?A"),
    "EQTL3":  ("Utilidade Pública", r"^EQUATORIAL"),
    "SBSP3":  ("Utilidade Pública", r"SABESP|SANEAMENTO BASICO.*SAO PAULO"),
    "TAEE11": ("Utilidade Pública", r"^TRANSMISSORA ALIANCA"),
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
@lru_cache(maxsize=None)
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
    Entrada: DataFrame de coletar_receita (precisa de cnpj e nome); pode juntar vários anos,
             porque o nome da empresa na CVM muda com o tempo (ex.: Eletrobras -> Axia).
    Saída: dict {ticker: cnpj}. Tickers que não casam aparecem no terminal, com nomes parecidos.
    """
    nomes = financeiro[["cnpj", "nome"]].drop_duplicates()
    nomes["n"] = nomes["nome"].map(normalizar)
    mapa = {}
    for ticker, (_, regex) in EMPRESAS.items():
        achados = nomes[nomes["n"].map(lambda n: bool(re.search(regex, n)))]
        if achados.empty:
            palavra = max(re.findall(r"[A-Z0-9]{4,}", regex), key=len, default="")     # palavra-chave do regex
            parecidos = nomes.loc[nomes["n"].str.contains(palavra, regex=False), "nome"].head(5).tolist() if palavra else []
            print(f"[AVISO] {ticker}: nenhuma empresa casou com o regex. Ajuste em EMPRESAS. "
                  f"Nomes parecidos no arquivo: {parecidos}")
            continue
        escolhida = achados.loc[achados["n"].str.len().idxmin()]          # nome mais curto se houver dúvida
        mapa[ticker] = escolhida["cnpj"]
        print(f"{ticker} -> {escolhida['nome']}")
    print(f"Empresas identificadas: {len(mapa)} de {len(EMPRESAS)}")
    return mapa


def montar_base(ano, mapa):
    """
    Junta receita e empregados de UM ano numa tabela.
    Entrada: ano do exercício e o dicionário {ticker: cnpj} de identificar_empresas.
    Saída: DataFrame com [ticker, empresa, setor, ano, receita, empregados], uma linha por ticker
           (dados faltantes ficam NaN, então nenhuma empresa some da tabela).
    """
    financeiro = coletar_receita(ano)
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
    Setores com menos de MIN_SETOR empresas válidas usam o z-score do universo inteiro,
    porque com 2 empresas o z-score seria sempre +-0,71 e não diria nada.
    """
    grupo = df.groupby("setor")[coluna]
    z_setor = (df[coluna] - grupo.transform("mean")) / grupo.transform("std")
    z_universo = (df[coluna] - df[coluna].mean()) / df[coluna].std()
    return z_setor.where(grupo.transform("count") >= MIN_SETOR, z_universo)


def calcular_ano(ano, mapa):
    """
    Calcula o indicador de UM ano.
    Entrada: ano do exercício e o dicionário {ticker: cnpj}.
    Saída: DataFrame com [ticker, empresa, setor, ano, receita, empregados, produtividade, z_produtividade]
        produtividade   = receita / empregados (R$ por empregado); vazia se empregados < MIN_EMPREGADOS
        z_produtividade = z-score setorial do log da produtividade, winsorizado em 1% e 99%
    """
    b = montar_base(ano, mapa)
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
    nomes_todos = pd.concat([coletar_receita(a) for a in anos], ignore_index=True)
    mapa = identificar_empresas(nomes_todos)                         # ticker -> CNPJ (uma vez só)
    por_ano = pd.concat([calcular_ano(a, mapa) for a in anos], ignore_index=True)
    # O nome na CVM muda com o tempo (ex.: Via -> Grupo Casas Bahia): usa o nome mais recente em todos os anos
    por_ano["empresa"] = por_ano.groupby("ticker")["empresa"].transform(
        lambda nomes: nomes[nomes != "n/d"].iloc[-1] if (nomes != "n/d").any() else "n/d")
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