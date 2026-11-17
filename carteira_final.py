# -*- coding: utf-8 -*-
"""
FACTOR INVESTING vs IBOVESPA  —  código único do trabalho
Programação para Análise de Dados (Python) · Prof. Laerte Takeuti · 2026.2
Grupo: Davi Pepe, João Marcos Zanetti e Luiza Fontana

O QUE ESTE ARQUIVO FAZ
  Fator 1  Escolaridade do CEO   (CVM, FRE 2017–2023)          -> 10 maiores notas
  Fator 2  Baixa volatilidade     (Yahoo Finance, jan/22–dez/23) -> 10 ações menos voláteis
  Fator 3  Capital humano         (CVM, DFP + FRE, 2022–2023)    -> 10 maiores índices de produtividade vs mediana do setor
  Carteira Cada fator com 1/3 do capital, formada em 28/03/2024 e comparada ao Ibovespa até 01/07/2026

UNIVERSO (igual para os três fatores)
  ibovespa_2024.csv: carteira teórica do Ibovespa de jan–abr/2024 (87 tickers, 84 empresas).

COMO RODAR (na pasta do projeto, no VS Code)
  pip install pandas numpy requests yfinance matplotlib openpyxl pyarrow
  python carteira_final.py

ONDE FICAM OS RESULTADOS
  outputs/      -> resultados de cada fator (o nome do arquivo começa com fator1_, fator2_ ou fator3_)
  pasta raiz    -> resultados da carteira final (carteira_*.csv e carteira_*.png)

ARQUIVOS NECESSÁRIOS NA PASTA
  ibovespa_2024.csv (obrigatório) · ajustes_manuais_fator1.csv (opcional, Fator 1)
  dados_cvm/ e cache_cvm/ (opcional: arquivos da CVM já baixados; se não existirem, o código baixa)

OBSERVAÇÃO
  Os métodos de cada fator são os mesmos dos códigos originais (ceo_escolaridade.py, factor_low_vol.py,
  webscrapping.py e carteira.py, guardados em codigos_originais/). A única mudança de conteúdo é o
  universo: todos os fatores agora usam o mesmo ibovespa_2024.csv.
  Fator 2: o filtro de histórico completo passou a olhar só a janela do critério (2022–2023), sem dados
  posteriores à formação; quem mudou de ticker usa o histórico do ticker novo (TICKER_NOVO_YAHOO).
"""
import io
import os
import re
import unicodedata
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ######################################################################################################
# 0) CONFIGURAÇÃO GERAL
# ######################################################################################################
ARQ_IBOV = "ibovespa_2024.csv"          # universo comum: Ibovespa de jan–abr/2024 (ticker; empresa; cnpj)
PASTA_OUTPUTS = Path("outputs")         # resultados de cada fator
DATA_FORMACAO = "2024-03-28"            # formação da carteira (preço de compra = fechamento deste dia)
DATA_FIM = "2026-07-01"                 # fim da comparação com o Ibovespa

# Empresas com mais de um ticker no Ibovespa: a nota é da empresa, então entra um ticker só (o mais líquido).
# Vale para os Fatores 1 e 3 (o Fator 2 trabalha com preços, ticker a ticker, como no código original).
TICKER_PRINCIPAL = {"Petrobras": "PETR4", "Bradesco": "BBDC4", "Eletrobras": "ELET3"}

# Ticker que mudou depois da formação (mesma empresa, código novo): se o Yahoo não tiver preço no ticker
# antigo, usa o novo. Vale para o Fator 2 (janela do critério) e para a carteira (seção 4).
# Empresas compradas por outra ou que fecharam o capital (BRFS3, SOMA3, CIEL3, CRFB3) não entram aqui:
# o ticker da compradora seria o preço de outra empresa.
TICKER_NOVO_YAHOO = {
    "ELET3": "AXIA3",   # Eletrobras mudou de nome e ticker em 10/11/2025; o Yahoo só tem o histórico sob AXIA3
    "ELET6": "AXIA6",
    "CPLE6": "CPLE3",   # Copel unificou as classes de ações; o CPLE6 deixou de ter cotação no Yahoo
    "MRFG3": "MBRF3",   # Marfrig incorporou a BRF em 2025 e passou a negociar como MBRF3
    "TRPL4": "ISAE4",   # ISA CTEEP virou ISA Energia Brasil e mudou de ticker
    "CCRO3": "MOTV3",   # CCR virou Motiva
    "EMBR3": "EMBJ3",   # Embraer mudou de ticker
    "ARZZ3": "AZZA3",   # Arezzo incorporou o Grupo Soma e virou Azzas 2154
    "NTCO3": "NATU3",   # Natura &Co foi incorporada pela Natura
    "RRRP3": "BRAV3",   # 3R Petroleum virou Brava Energia
    "PETZ3": "AUAU3",   # Petz (fusão com a Cobasi): conferir se o histórico é mesmo o da Petz
}


def carregar_universo():
    """Lê o ibovespa_2024.csv. Saída: DataFrame com ticker, empresa e cnpj.
    Obs.: 87 tickers de 84 empresas; 4 empresas aparecem com dois CNPJs (Iguatemi, Natura, Raízen e TIM
    trocaram de CNPJ por reorganização societária), para o Fator 1 achar os CEOs nos dois cadastros."""
    ibov = pd.read_csv(ARQ_IBOV, sep=";", dtype=str, encoding="utf-8-sig")
    print(f"Universo: {ibov['ticker'].nunique()} tickers de {ibov['empresa'].nunique()} empresas ({ARQ_IBOV})")
    return ibov


# ######################################################################################################
# 1) FATOR 1 — ESCOLARIDADE DO CEO
#    Critério: 2017 a dez/2023 | Fonte: CVM, Formulário de Referência (FRE)
#
#    IDEIA EM 5 PASSOS
#      1. Baixar da CVM o FRE de cada ano, de 2017 a 2023.
#      2. Ficar só com os CEOs (Diretor Presidente).
#      3. Ler o currículo de cada CEO e dar pontos pela formação:
#             graduação = 1 | MBA ou especialização = 1 | mestrado = 2 | doutorado = 3   (máximo 7)
#      4. Nota da empresa em cada ano = média dos CEOs daquele ano.  Nota final = média dos anos.
#      5. Ordenar as empresas pela nota final. Entram na carteira as 10 maiores notas.
# ######################################################################################################
F1_ANOS = list(range(2017, 2024))                                     # 2017, 2018, ..., 2023
F1_PESOS = {"graduacao": 1, "mba": 1, "mestrado": 2, "doutorado": 3}  # pontos de cada nível de formação
F1_TOP_N = 10                                                         # quantas empresas mostrar no terminal
F1_N_SELECAO = 10                                                     # entram as 10 maiores notas, como nos Fatores 2 e 3
F1_ARQ_AJUSTES = "ajustes_manuais_fator1.csv"                              # correções feitas à mão (opcional)
F1_PASTA_ZIPS = Path("dados_cvm")                                     # onde ficam os arquivos baixados
F1_SAIDA = PASTA_OUTPUTS / "fator1_ranking_escolaridade_ceo.xlsx"
F1_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"


# ----------------------------------------------------------------------------------------------------
# 1.1 Limpar texto
# ----------------------------------------------------------------------------------------------------
def sem_acento(texto):
    """Deixa o texto em minúsculas e sem acento: 'Doutorado' -> 'doutorado'. Assim a busca de palavras
    não erra por causa de acento ou maiúscula."""
    if pd.isna(texto):
        return ""
    texto = unicodedata.normalize("NFKD", str(texto))                 # separa a letra do acento
    texto = "".join(c for c in texto if not unicodedata.combining(c))  # joga o acento fora
    return re.sub(r"\s+", " ", texto).strip().lower()


def achar_coluna(tabela, *nomes):
    """A CVM às vezes muda o nome das colunas. Procura a coluna pelo nome exato; se não achar, procura uma
    coluna que CONTENHA o nome (sem pegar colunas da 'companhia', para não confundir nome da pessoa com
    nome da empresa)."""
    colunas = {sem_acento(c).replace(" ", "_"): c for c in tabela.columns}
    for nome in nomes:                                  # 1ª tentativa: nome exato
        if nome in colunas:
            return colunas[nome]
    for nome in nomes:                                  # 2ª tentativa: nome que contém o texto
        for chave, original in colunas.items():
            if nome in chave and "companhia" not in chave:
                return original
    raise KeyError(f"Coluna não encontrada: {nomes}. Colunas do arquivo: {list(tabela.columns)}")


# ----------------------------------------------------------------------------------------------------
# 1.2 Baixar os dados da CVM
# ----------------------------------------------------------------------------------------------------
def baixar_fre():
    """Para cada ano: baixa o .zip (se ainda não estiver na pasta dados_cvm), abre a tabela de administradores
    e junta tudo numa tabela só. A coluna 'ano' guarda de qual ano veio cada linha."""
    F1_PASTA_ZIPS.mkdir(exist_ok=True)
    tabelas = []
    for ano in F1_ANOS:
        arquivo = F1_PASTA_ZIPS / f"fre_cia_aberta_{ano}.zip"
        if not arquivo.exists():
            print(f"Baixando FRE {ano}...")
            resposta = requests.get(F1_URL.format(ano=ano), timeout=300)
            resposta.raise_for_status()
            arquivo.write_bytes(resposta.content)
        zip_cvm = zipfile.ZipFile(arquivo)
        # dentro do zip há vários arquivos; o dos administradores tem este pedaço no nome
        for nome in zip_cvm.namelist():
            if "administrador_membro_conselho_fiscal" in nome.lower():
                t = pd.read_csv(io.BytesIO(zip_cvm.read(nome)), sep=";", encoding="latin-1", dtype=str)
                t["ano"] = ano
                tabelas.append(t)
    return pd.concat(tabelas, ignore_index=True).drop_duplicates()


# ----------------------------------------------------------------------------------------------------
# 1.3 Achar os CEOs
# ----------------------------------------------------------------------------------------------------
# padrões de texto (expressões regulares): "diretor presidente", "dir. presidente"...
RE_CEO = re.compile(r"(?:diretor|dir\.?)[\s\-]*presidente|presidente\s*/\s*superintendente")
RE_CEO_COMPLEMENTO = re.compile(r"(?:diretor|dir\.?)[\s\-]*presidente|\bceo\b|chief executive|diretor[\s\-]*geral")
# NÃO é o CEO da empresa: vice, adjunto, ou CEO de uma unidade/subsidiária ("CEO - Plataforma Digital")
RE_NAO_CEO = re.compile(r"\bvice\b|\badjunt[oa]\b|(?:\bceo\b|diretor[\s\-]*(?:geral|presidente))\s*(?:-|–|\bda\b|\bdo\b|\bde\b)\s*\w")


def achar_ceos(adm):
    """Recebe a tabela de administradores e devolve só as linhas dos CEOs, com estas colunas:
    cnpj, ano, pessoa (nome limpo), profissao e curriculo.

    Regra: é CEO quem tem cargo 'Diretor Presidente' no campo oficial do FRE. Se uma entrega do FRE não tem
    ninguém assim, olhamos o complemento do cargo (ex.: 'Outros Diretores | CEO')."""
    c = {"cnpj": achar_coluna(adm, "cnpj_companhia", "cnpj"),
         "data": achar_coluna(adm, "data_referencia"),
         "versao": achar_coluna(adm, "versao"),
         "nome": achar_coluna(adm, "nome_administrador", "nome"),
         "cargo": achar_coluna(adm, "cargo_eletivo_ocupado", "cargo"),
         "complemento": achar_coluna(adm, "complemento_cargo"),
         "profissao": achar_coluna(adm, "profissao"),
         "curriculo": achar_coluna(adm, "experiencia_profissional", "curriculo")}
    print("Colunas usadas do FRE:", c)                       # conferência: 'nome' deve ser a PESSOA
    d = pd.DataFrame({novo: adm[original] for novo, original in c.items()})
    d["ano"] = adm["ano"].values
    d["cnpj"] = d["cnpj"].str.replace(r"\D", "", regex=True).str.zfill(14)   # só números, 14 dígitos

    cargo = d["cargo"].map(sem_acento)
    complemento = d["complemento"].map(sem_acento)
    pelo_cargo = cargo.str.contains(RE_CEO) & ~cargo.str.contains(RE_NAO_CEO)
    pelo_complemento = complemento.str.contains(RE_CEO_COMPLEMENTO) & ~complemento.str.contains(RE_NAO_CEO)
    entrega = d["cnpj"] + "|" + d["data"] + "|" + d["versao"]                 # identifica cada entrega do FRE
    entrega_tem_ceo_oficial = pelo_cargo.groupby(entrega).transform("any")    # True se a entrega tem CEO pelo cargo
    d["eh_ceo"] = pelo_cargo | (pelo_complemento & ~entrega_tem_ceo_oficial)

    ceos = d[d["eh_ceo"]].copy()
    # chave da pessoa: só letras maiúsculas ('João  da Silva' e 'JOAO DA SILVA' viram a mesma pessoa)
    nome = ceos["nome"].map(sem_acento).str.upper()
    ceos["pessoa"] = nome.str.replace(r"[^A-Z ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return ceos[ceos["pessoa"] != ""]


# ----------------------------------------------------------------------------------------------------
# 1.4 Ler a formação no currículo
# ----------------------------------------------------------------------------------------------------
# palavras que indicam cada nível (o texto já está sem acento e em minúsculas)
PALAVRAS = {
    "graduacao": r"graduou[\s\-]?se|formou[\s\-]?se|bacharelou[\s\-]?se|graduad[oa]s?|graduacao|bacharel(?:ad[oa])?|"
                 r"bachelor(?:'s)?|formad[oa]|formacao|licenciad[oa]|licenciatura|diplomad[oa]|curso superior|degree",
    "mba": r"mba|pos[\s\-]?graduad[oa]|pos[\s\-]?graduacao|especializacao|especialista|lato sensu|extensao|"
           r"executive education|advanced management|pmd|amp|opm",
    "mestrado": r"mestrad[oa]|mestre|master(?:'s)?|msc|m\.sc",
    "doutorado": r"doutorad[oa]|doutor|ph\.?\s?d|phd",
}
# "Master of Business Administration" é um MBA, não um mestrado: trocamos por 'mba' antes de procurar
MBA_POR_EXTENSO = r"master(?:'s)?\s+(?:of|in|em|de|degree in)?\s*business\s+administration|" \
                  r"mestre\s+em\s+administracao\s+de\s+empresas|master(?:'s)?\s+(?:in\s+)?business"
# profissões que exigem diploma: quem declara 'engenheiro' tem graduação
PROFISSOES = r"economista|engenheir[oa]|advogad[oa]|contador|contadora|administrador|administradora|medic[oa]|" \
             r"arquitet[oa]|atuari[oa]|geolog[oa]|quimic[oa]"


def ler_formacao(curriculo, profissao):
    """Recebe o texto do currículo (e a profissão) de UMA pessoa e devolve um dicionário com 4 valores
    True/False: graduacao, mba, mestrado, doutorado."""
    texto = re.sub(MBA_POR_EXTENSO, " mba ", sem_acento(curriculo))
    nivel = {nome: bool(re.search(r"\b(" + palavras + r")\b", texto)) for nome, palavras in PALAVRAS.items()}
    tem_pos = nivel["mba"] or nivel["mestrado"] or nivel["doutorado"]
    # quem tem pós-graduação (ou profissão de nível superior) necessariamente tem graduação
    nivel["graduacao"] = nivel["graduacao"] or tem_pos or bool(re.search(PROFISSOES, sem_acento(profissao)))
    return nivel


# ----------------------------------------------------------------------------------------------------
# 1.5 Nota e ranking das empresas
# ----------------------------------------------------------------------------------------------------
def pontuar_empresas(tabela, anos, pesos):
    """tabela: uma linha por (empresa, ano, CEO), com as colunas graduacao, mba, mestrado, doutorado.
    Devolve uma linha por empresa com: nota, n_anos (anos com CEO identificado), valida e posicao.

      1) nota da pessoa    = soma dos pontos dos níveis que ela tem (ex.: graduação + mestrado = 1 + 2 = 3)
      2) nota empresa-ano  = média das notas dos CEOs daquele ano (2 CEOs no ano: cada um vale metade)
      3) nota da empresa   = média das notas dos anos
    A empresa só é 'valida' (entra no ranking) se tiver CEO identificado em pelo menos metade dos anos."""
    t = tabela[tabela["ano"].isin(anos)].copy()
    t["nota_pessoa"] = sum(t[nivel] * pontos for nivel, pontos in pesos.items())             # etapa 1
    nota_por_ano = t.groupby(["empresa", "ano"])["nota_pessoa"].mean()                         # etapa 2
    resultado = nota_por_ano.groupby("empresa").agg(nota="mean", n_anos="count")               # etapa 3
    resultado["valida"] = resultado["n_anos"] >= (len(anos) + 1) // 2
    # posição 1 = maior nota; em caso de empate, vem primeiro quem tem mais anos com dado
    ordem = resultado[resultado["valida"]].sort_values(["nota", "n_anos"], ascending=False)
    resultado["posicao"] = pd.Series(range(1, len(ordem) + 1), index=ordem.index)
    return resultado


# ----------------------------------------------------------------------------------------------------
# 1.6 Execução do Fator 1
# ----------------------------------------------------------------------------------------------------
def rodar_fator1(ibov):
    """Roda o Fator 1 inteiro. Entrada: universo (ibovespa_2024.csv).
    Saída: lista dos 10 tickers selecionados + arquivos em outputs/."""
    print("\n" + "=" * 100 + "\nFATOR 1 — ESCOLARIDADE DO CEO\n" + "=" * 100)

    # --- 1. dados de entrada
    adm = baixar_fre()
    ceos = achar_ceos(adm)

    # --- 2. formação de cada pessoa (lendo TODOS os currículos que ela tem, de todos os anos)
    textos = ceos.groupby("pessoa").agg(curriculo=("curriculo", lambda s: " || ".join(s.dropna().unique())),
                                        profissao=("profissao", lambda s: " ".join(s.dropna().unique())))
    formacao = pd.DataFrame([ler_formacao(l.curriculo, l.profissao) for l in textos.itertuples()],
                            index=textos.index)

    # --- 3. ajustes manuais (opcional): só ACRESCENTAM títulos confirmados pelo grupo
    if Path(F1_ARQ_AJUSTES).exists():
        ajustes = pd.read_csv(F1_ARQ_AJUSTES, sep=";", dtype=str, encoding="utf-8-sig", encoding_errors="replace")
        aplicados = 0
        for linha in ajustes.itertuples():
            pessoa = sem_acento(linha.pessoa).upper()
            if pessoa in formacao.index:
                for nivel in F1_PESOS:
                    if getattr(linha, nivel) == "1":
                        formacao.loc[pessoa, nivel] = True
                aplicados += 1
            else:
                print("Ajuste NÃO aplicado (nome não encontrado):", linha.pessoa)
        print(f"Ajustes manuais aplicados: {aplicados} de {len(ajustes)}")
    else:
        print("Arquivo de ajustes manuais não encontrado: nenhuma correção aplicada.")

    # --- 4. tabela (empresa, ano, CEO): fica só quem trabalhou em empresa do Ibovespa de 2024
    lista = ibov[["cnpj", "empresa"]].drop_duplicates()
    tabela = ceos.merge(lista, on="cnpj")                                    # junta pelo CNPJ
    tabela = tabela[["empresa", "ano", "pessoa"]].drop_duplicates()          # 1 linha por empresa-ano-CEO
    tabela = tabela.join(formacao, on="pessoa")                              # acrescenta a formação

    # --- 5. ranking do caso-base
    base = pontuar_empresas(tabela, F1_ANOS, F1_PESOS)
    tickers = ibov.groupby("empresa")["ticker"].agg(lambda s: ", ".join(sorted(set(s))))
    ranking = base.join(tickers, how="right").sort_values("posicao")         # 'right' mantém empresas sem dado
    ranking["n_anos"] = ranking["n_anos"].fillna(0).astype(int)
    ranking["valida"] = ranking["valida"].fillna(False).astype(bool)

    # --- 6. quem é cada CEO (para conferir) e quem ficou sem nenhum título detectado (para revisar)
    tabela["nota_pessoa"] = sum(tabela[nivel] * pontos for nivel, pontos in F1_PESOS.items())
    ceos_por_empresa = tabela.groupby(["empresa", "pessoa"]).agg(
        anos=("ano", lambda s: ", ".join(str(a) for a in sorted(set(s)))),
        graduacao=("graduacao", "max"), mba=("mba", "max"), mestrado=("mestrado", "max"),
        doutorado=("doutorado", "max"), nota_pessoa=("nota_pessoa", "max")).reset_index()
    revisar = ceos_por_empresa[ceos_por_empresa["nota_pessoa"] == 0]

    # --- 7. seleção da carteira: as 10 maiores notas entre as empresas válidas (mesmo número dos Fatores 2 e 3)
    #        Empresa com dois tickers (ex.: PETR3 e PETR4) entra com um só, o de TICKER_PRINCIPAL.
    selecionadas = ranking[ranking["valida"]].sort_values("posicao").head(F1_N_SELECAO).copy()
    selecionadas["ticker_carteira"] = [TICKER_PRINCIPAL.get(emp, t.split(", ")[0])
                                       for emp, t in zip(selecionadas.index, selecionadas["ticker"])]

    # --- 8. gravar os resultados
    with pd.ExcelWriter(F1_SAIDA) as arquivo:
        ranking.reset_index().to_excel(arquivo, sheet_name="ranking_empresas", index=False)
        ceos_por_empresa.to_excel(arquivo, sheet_name="ceos_por_empresa", index=False)
        revisar.to_excel(arquivo, sheet_name="revisar_sem_titulo", index=False)
    selecionadas.reset_index()[["posicao", "empresa", "ticker_carteira", "nota", "n_anos"]].to_csv(
        PASTA_OUTPUTS / "fator1_selecionadas.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")

    print(f"\nTop {F1_TOP_N} (critério 2017-2023, Ibovespa jan/2024):")
    print(ranking.head(F1_TOP_N)[["posicao", "ticker", "nota", "n_anos"]].to_string())
    print(f"\nSelecionadas (top {F1_N_SELECAO}): {len(selecionadas)} -> {selecionadas['ticker_carteira'].tolist()}")
    print(f"Arquivos gerados: {F1_SAIDA} e {PASTA_OUTPUTS / 'fator1_selecionadas.csv'}")
    return selecionadas["ticker_carteira"].tolist()


# ######################################################################################################
# 2) FATOR 2 — BAIXA VOLATILIDADE
#    Critério: jan/2022 a dez/2023 | Fonte: Yahoo Finance (preços ajustados a dividendos)
#
#    IDEIA
#      1. Baixar os preços diários de todos os tickers do universo, SÓ na janela do critério (2022–2023).
#      2. Se o Yahoo não tiver o histórico no ticker antigo, usar o ticker novo da mesma empresa
#         (TICKER_NOVO_YAHOO, seção 0). Empresas compradas ou que fecharam o capital não têm ticker novo.
#      3. Manter só quem tem histórico completo em 2022–2023 (dropna).
#      4. Volatilidade anualizada de 2022–2023 = desvio padrão dos retornos diários × √252.
#      5. Entram as 10 ações MENOS voláteis, com pesos iguais.
#    Obs.: a versão anterior exigia preços completos até dez/2025. Isso usava informação posterior à
#          formação (quem ainda existiria em 2025) e excluía quem saiu da bolsa depois de 28/03/2024.
#          Agora a seleção usa só dados disponíveis na formação. A avaliação é a da seção 4 (Carteira),
#          igual para todos os fatores: 28/03/2024 a 01/07/2026.
# ######################################################################################################
F2_INICIO_IN_SAMPLE = "2022-01-01"     # início da janela do critério
F2_FIM_IN_SAMPLE = "2023-12-31"        # fim da janela do critério (nenhum dado depois disso entra na seleção)
F2_TOP_N = 10


def baixar_precos_criterio(tickers, inicio, fim):
    """Baixa o fechamento ajustado da janela do critério. Entrada: tickers sem '.SA'.
    Saída: (DataFrame com uma coluna por ticker do universo, dicionário ticker -> série usada no Yahoo)."""
    import yfinance as yf
    fim_exclusivo = (pd.Timestamp(fim) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # o 'end' do Yahoo não inclui o dia
    novos = [TICKER_NOVO_YAHOO[t] for t in tickers if t in TICKER_NOVO_YAHOO]
    simbolos = [t + ".SA" for t in sorted(set(tickers) | set(novos))]
    bruto = yf.download(simbolos, start=inicio, end=fim_exclusivo, auto_adjust=True)["Close"]
    bruto.columns = [str(c).replace(".SA", "") for c in bruto.columns]
    bruto = bruto.reindex(columns=[s.replace(".SA", "") for s in simbolos])

    dados, serie_usada = {}, {}
    for t in tickers:
        if bruto[t].notna().sum() == 0 and t in TICKER_NOVO_YAHOO and bruto[TICKER_NOVO_YAHOO[t]].notna().sum() > 0:
            serie_usada[t] = TICKER_NOVO_YAHOO[t]          # ex.: ELET3 sem preço -> histórico de AXIA3
        else:
            serie_usada[t] = t
        dados[t] = bruto[serie_usada[t]]
    return pd.DataFrame(dados), serie_usada


def run_low_vol_factor(tickers, inicio_in_sample, fim_in_sample, top_n=10):
    """
    Seleção do Fator de Baixa Volatilidade usando só a janela do critério.
    Saída: lista das ações selecionadas, ranking completo de volatilidade e a série usada para cada ticker.
    """
    print(f"1. Baixando dados de {inicio_in_sample} até {fim_in_sample}...")
    dados, serie_usada = baixar_precos_criterio(tickers, inicio_in_sample, fim_in_sample)
    trocas = {t: s for t, s in serie_usada.items() if s != t and dados[t].notna().any()}
    for antigo, novo in trocas.items():
        print(f"   [AVISO] {antigo} sem preço no Yahoo: usando o histórico de {novo}")

    # Limpeza: remove ações sem histórico completo na janela do critério
    completos = dados.dropna(axis=1)
    fora = sorted(set(tickers) - set(completos.columns))
    print(f"   Tickers com histórico completo em 2022–2023: {completos.shape[1]} de {len(tickers)}")
    print(f"   Fora (sem preço completo no Yahoo): {fora}")
    retornos = completos.pct_change().dropna()

    print("2. Calculando volatilidade e ranqueando as empresas...")
    vol_anualizada = retornos.std() * np.sqrt(252)                  # volatilidade anualizada de 2022–2023
    ranking_low_vol = vol_anualizada.sort_values(ascending=True)    # do MENOR para o MAIOR risco
    carteira_selecionada = ranking_low_vol.head(top_n).index.tolist()
    return carteira_selecionada, ranking_low_vol, serie_usada


def rodar_fator2(ibov):
    """Roda o Fator 2 com os tickers do ibovespa_2024.csv.
    Saída: lista de tickers selecionados + arquivos em outputs/."""
    print("\n" + "=" * 100 + "\nFATOR 2 — BAIXA VOLATILIDADE\n" + "=" * 100)
    tickers_ibovespa = ibov["ticker"].unique().tolist()   # mesmo universo dos outros fatores (87 tickers)

    selecionadas, ranking, serie_usada = run_low_vol_factor(
        tickers=tickers_ibovespa,
        inicio_in_sample=F2_INICIO_IN_SAMPLE,
        fim_in_sample=F2_FIM_IN_SAMPLE,
        top_n=F2_TOP_N,
    )

    # --- gravar os resultados
    tabela = ranking.rename("volatilidade_anual_2022_2023").reset_index()
    tabela.columns = ["ticker", "volatilidade_anual_2022_2023"]
    tabela.insert(0, "posicao", range(1, len(tabela) + 1))
    tabela["serie_yahoo"] = tabela["ticker"].map(serie_usada)
    tabela["selecionada"] = tabela["ticker"].isin(selecionadas)
    tabela.to_csv(PASTA_OUTPUTS / "fator2_ranking_volatilidade.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    pd.DataFrame({"ticker": selecionadas}).to_csv(PASTA_OUTPUTS / "fator2_selecionadas.csv", index=False, encoding="utf-8-sig")

    print("\n-> AS 10 AÇÕES SELECIONADAS (Com base em 2022-2023):")
    print(tabela.head(F2_TOP_N + 5).to_string(index=False))
    print(f"Arquivos salvos em {PASTA_OUTPUTS.resolve()}")
    return selecionadas

# ######################################################################################################
# 3) FATOR 3 — CAPITAL HUMANO (PRODUTIVIDADE DO TRABALHO)
#    Critério: 2022 e 2023 | Fonte: CVM (DFP: receita na DVA; FRE: número de empregados)
#
#    Indicador único: produtividade relativa = produtividade da empresa / mediana do setor.
#        produtividade = receita (DVA 7.01) / nº de empregados (FRE 10.1A)
#        índice 1,20 = produtividade 20% acima da mediana do setor
#    Fluxo, para cada ano do PERIODO:
#        montar_base()  ->  calcular_ano()  (produtividade, mediana setorial e índice anual)
#    Depois:
#        calcular_periodo()  ->  média dos índices anuais  ->  tabela_top()  ->  salvar()
#    Universo: o mesmo ibovespa_2024.csv, um ticker por empresa (ações ON/PN da mesma empresa contam uma vez).
# ######################################################################################################
F3_PERIODO = [2022, 2023]          # anos analisados; o indicador final é a média entre eles
F3_MIN_EMPREGADOS = 100            # empresas com menos empregados ficam sem nota
F3_MIN_SETOR = 3                   # setor com menos empresas válidas fica sem índice naquele ano
F3_TOP_N = 10                      # quantas empresas entram na carteira

# Exclusões do Fator 3 (decisão do responsável pelo fator).
# Holding com receita CONSOLIDADA (soma das controladas) e empregados só da HOLDING: a divisão
# receita / empregados não mede a produtividade do trabalho e gera um outlier extremo.
# A empresa sai do cálculo inteiro: não recebe índice e não entra na mediana do seu setor.
F3_EXCLUIR = {
    "CSAN3": "Cosan: holding (consolida Rumo, Compass e Moove); receita consolidada ÷ ~140 empregados da holding; "
             "índice 21,7, mais que o dobro do 2º colocado",
}
F3_PASTA_CACHE = Path("cache_cvm")
F3_ARQ_EMPREGADOS_MANUAL = Path("empregados_manual.csv")   # opcional: colunas ticker, ano, empregados

URL_DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{ano}.zip"
URL_FRE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"

# Universo do Fator 3: empresas do ibovespa_2024.csv, um ticker por empresa.
# ticker: (setor, regex do nome da empresa na CVM, sem acento e em maiúsculas)
# Setores: classificação setorial da B3 (atribuída manualmente; confira se necessário).
# Diferenças em relação ao webscrapping.py original (que usava o Ibovespa de set–dez/2023):
#   ALSO3 -> ALOS3 (Aliansce Sonae virou Allos), VIIA3 -> BHIA3 (Via virou Grupo Casas Bahia),
#   e entra TRPL4 (ISA CTEEP), que passou a fazer parte do índice em jan/2024.
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
    "BHIA3":  ("Consumo Cíclico", r"^VIA S\.?\s?A|^VIA VAREJO|^GRUPO CASAS BAHIA"),
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
    "ALOS3":  ("Financeiro", r"^ALLOS|^ALIANSCE"),
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
    "TRPL4":  ("Utilidade Pública", r"^ISA ENERGIA|^ISA CTEEP|^(CIA|COMPANHIA) DE TRANSMISSAO DE ENERGIA ELETRICA PAULISTA"),
}


def conferir_universo_fator3(ibov):
    """Confere se o dicionário EMPRESAS cobre exatamente as empresas do ibovespa_2024.csv
    (um ticker por empresa). Só imprime avisos; não muda nenhum cálculo."""
    no_csv = set(ibov["ticker"])
    fora_do_csv = sorted(set(EMPRESAS) - no_csv)
    por_empresa = ibov.groupby("empresa")["ticker"].apply(set)
    sem_ticker = [sorted(t) for t in por_empresa if not t & set(EMPRESAS)]
    duplicadas = [sorted(t & set(EMPRESAS)) for t in por_empresa if len(t & set(EMPRESAS)) > 1]
    print(f"Fator 3: {len(EMPRESAS)} empresas no dicionário; {len(por_empresa)} empresas no {ARQ_IBOV}")
    if fora_do_csv:
        print("[AVISO] tickers do Fator 3 que não estão no CSV:", fora_do_csv)
    if sem_ticker:
        print("[AVISO] empresas do CSV sem ticker no Fator 3:", sem_ticker)
    if duplicadas:
        print("[AVISO] empresas com mais de um ticker no Fator 3:", duplicadas)


# ----------------------------------------------------------------------------------------------------
# 3.1 Funções auxiliares
# ----------------------------------------------------------------------------------------------------
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


def achar_coluna_regex(df, regex):
    """Entrada: DataFrame e regex.  Saída: nome da 1ª coluna que casa com o regex (ou None).
    (No webscrapping.py original se chamava achar_coluna; o nome mudou só para não colidir com a do Fator 1.)"""
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


# ----------------------------------------------------------------------------------------------------
# 3.2 Coleta de dados
# ----------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def coletar_receita(ano):
    """
    Lê a receita na DVA (DFP) da CVM.
    Entrada: ano do exercício (ex.: 2023).
    Saída: DataFrame com [cnpj, nome, ano, receita] em R$, uma linha por empresa (receita = DVA conta 7.01).
    """
    zip_dfp = baixar_zip(URL_DFP.format(ano=ano), F3_PASTA_CACHE / f"dfp_{ano}.zip")
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
        zip_fre = baixar_zip(URL_FRE.format(ano=ano_fre), F3_PASTA_CACHE / f"fre_{ano_fre}.zip")
        df = ler_csv_do_zip(zip_fre, "empregado_posicao_local")
        if df is None:
            print(f"[FRE {ano_fre}] não há arquivo de empregados neste ano (layout antigo): exercício {ano_fre - 1} sem dado.")
            return vazio
        c_cnpj, c_data = achar_coluna_regex(df, "CNPJ"), achar_coluna_regex(df, "DATA.*REFER|DT_REFER")
        c_versao = achar_coluna_regex(df, "^VERS")
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
    if F3_ARQ_EMPREGADOS_MANUAL.exists():
        manual = pd.read_csv(F3_ARQ_EMPREGADOS_MANUAL, sep=None, engine="python")
        manual = manual[manual["ano"] == ano].rename(columns={"empregados": "emp_manual"})
        base = base.merge(manual[["ticker", "emp_manual"]], on="ticker", how="left")
        base["empregados"] = base["emp_manual"].combine_first(base["empregados"])
        base = base.drop(columns="emp_manual")
    return base[["ticker", "empresa", "setor", "ano", "receita", "empregados"]]


# ----------------------------------------------------------------------------------------------------
# 3.3 Indicador de produtividade
# ----------------------------------------------------------------------------------------------------
def calcular_mediana_setorial(df):
    """
    Retorna a mediana da produtividade e a quantidade de empresas válidas de cada setor.
    A mediana só é liberada como referência quando o setor possui pelo menos F3_MIN_SETOR
    empresas com produtividade válida no ano.
    """
    grupo = df.groupby("setor")["produtividade"]
    quantidade = grupo.transform("count")
    mediana = grupo.transform("median").where(quantidade >= F3_MIN_SETOR)
    return mediana, quantidade


def calcular_ano(ano, mapa):
    """
    Calcula o indicador de um ano:
        produtividade = receita / empregados;
        indice_produtividade = produtividade / mediana da produtividade do setor.

    Empresas com menos de F3_MIN_EMPREGADOS, receita não positiva ou dado ausente ficam
    sem produtividade. Setores com menos de F3_MIN_SETOR empresas válidas ficam sem índice.
    Empresas em F3_EXCLUIR também ficam sem produtividade (e, portanto, fora da mediana do setor).
    """
    b = montar_base(ano, mapa)
    b["excluida"] = b["ticker"].isin(F3_EXCLUIR)                          # holdings/outliers (ver F3_EXCLUIR)

    empregados_validos = b["empregados"].where(b["empregados"] >= F3_MIN_EMPREGADOS)
    receita_valida = b["receita"].where((b["receita"] > 0) & ~b["excluida"])
    b["produtividade"] = receita_valida / empregados_validos

    b["mediana_produtividade_setor"], b["empresas_validas_setor"] = calcular_mediana_setorial(b)
    b["indice_produtividade"] = b["produtividade"] / b["mediana_produtividade_setor"]
    return b


# ----------------------------------------------------------------------------------------------------
# 3.4 Período e ranking
# ----------------------------------------------------------------------------------------------------
def calcular_periodo(anos):
    """
    Calcula cada ano e tira a média dos índices anuais disponíveis.
    Se a empresa tiver somente um ano válido, esse único índice será o indicador final.
    A produtividade média em R$ por empregado é mantida apenas para consulta.
    """
    nomes_todos = pd.concat([coletar_receita(a) for a in anos], ignore_index=True)
    mapa = identificar_empresas(nomes_todos)                         # ticker -> CNPJ (uma vez só)
    por_ano = pd.concat([calcular_ano(a, mapa) for a in anos], ignore_index=True)

    # Usa em todos os anos o nome mais recente encontrado na CVM (ex.: Via -> Grupo Casas Bahia).
    por_ano["empresa"] = por_ano.groupby("ticker")["empresa"].transform(
        lambda nomes: nomes[nomes != "n/d"].iloc[-1] if (nomes != "n/d").any() else "n/d")

    grupo = por_ano.groupby(["ticker", "empresa", "setor"])
    media = pd.DataFrame({
        "produtividade_media": grupo["produtividade"].mean(),
        "indicador_produtividade": grupo["indice_produtividade"].mean(),
        "anos_com_dado": grupo["indice_produtividade"].count(),
    }).reset_index()

    media["ranking"] = media["indicador_produtividade"].rank(ascending=False, method="min")
    media = media.sort_values(
        "indicador_produtividade", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return por_ano, media


def tabela_top(media, n=F3_TOP_N):
    """
    Retorna as n maiores notas válidas.
    Exemplo: indicador 1,20 significa produtividade 20% acima da mediana do setor.
    """
    top = media.dropna(subset=["indicador_produtividade"]).head(n)
    return pd.DataFrame({
        "Posição": range(1, len(top) + 1),
        "Empresa": top["empresa"].values,
        "Ticker": top["ticker"].values,
        "Indicador": top["indicador_produtividade"].round(3).values,
    })


# ----------------------------------------------------------------------------------------------------
# 3.5 Saída e execução do Fator 3
# ----------------------------------------------------------------------------------------------------
def salvar(resultado, nome):
    """
    Entrada: DataFrame e o nome-base dos arquivos (ex.: "fator3_produtividade_periodo").
    Saída: 3 arquivos em outputs/ -> <nome>.csv, <nome>_excel.csv (sep ';' e decimal ',') e <nome>.parquet.
    """
    resultado.to_csv(PASTA_OUTPUTS / f"{nome}.csv", index=False, encoding="utf-8-sig")
    resultado.to_csv(PASTA_OUTPUTS / f"{nome}_excel.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    try:
        resultado.to_parquet(PASTA_OUTPUTS / f"{nome}.parquet", index=False)
    except Exception as erro:
        print(f"[AVISO] Parquet não salvo ({erro}). Instale pyarrow.")


def rodar_fator3(ibov):
    """Roda o Fator 3 inteiro. Saída: lista dos 10 tickers selecionados + arquivos em outputs/."""
    print("\n" + "=" * 100 + "\nFATOR 3 — CAPITAL HUMANO (PRODUTIVIDADE)\n" + "=" * 100)
    conferir_universo_fator3(ibov)
    for ticker, motivo in F3_EXCLUIR.items():
        print(f"[EXCLUÍDA] {ticker}: {motivo}")
    por_ano, media = calcular_periodo(F3_PERIODO)       # coleta, indicador por ano e média
    top = tabela_top(media)                             # tabela de resultados
    salvar(por_ano, "fator3_produtividade_por_ano")
    salvar(media, "fator3_produtividade_periodo")
    salvar(top, "fator3_top10")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n===== INDICADOR DE PRODUTIVIDADE DO PERÍODO =====")
    print(media.round(3).to_string(index=False))
    print(f"\n===== TOP {F3_TOP_N} (tabela de resultados) =====")
    print(top.to_string(index=False))
    print(f"Arquivos salvos em {PASTA_OUTPUTS.resolve()}")
    return top["Ticker"].tolist()


# ######################################################################################################
# 4) CARTEIRA DE FACTOR INVESTING vs IBOVESPA
#    Formação: 28/03/2024 (preço de compra = fechamento do dia) | Comparação até 01/07/2026
#
#    Regras
#      - Pesos iguais dentro de cada fator (ex.: 8 empresas -> 12,5% cada).
#      - Carteira combinada: cada fator vale 1/3; ação repetida em dois fatores soma os pesos.
#      - Compra e mantém (sem rebalanceamento), preços ajustados por dividendos (Yahoo Finance).
#      - Métricas: retorno acumulado, volatilidade anualizada, Sharpe (sem taxa livre de risco) e drawdown.
#    Os resultados desta seção ficam direto na pasta do projeto (carteira_*.csv e carteira_*.png).
# ######################################################################################################
DIAS_ANO = 252                 # pregões por ano

# TICKER_NOVO_YAHOO fica na seção 0 (usado também pelo Fator 2).


# ----------------------------------------------------------------------------------------------------
# 4.1 Pesos
# ----------------------------------------------------------------------------------------------------
def pesos_do_fator(tickers):
    """Pesos iguais dentro de um fator (ex.: 8 empresas -> 12,5% cada)."""
    return pd.Series(1 / len(tickers), index=tickers)


def montar_carteira_combinada(fator_1, fator_2, fator_3):
    """Cada fator vale 1/3 da carteira; empresa repetida em dois fatores soma os pesos."""
    p1, p2, p3 = pesos_do_fator(fator_1), pesos_do_fator(fator_2), pesos_do_fator(fator_3)
    combinada = p1.mul(1 / 3).add(p2.mul(1 / 3), fill_value=0).add(p3.mul(1 / 3), fill_value=0)
    return p1, p2, p3, combinada


# ----------------------------------------------------------------------------------------------------
# 4.2 Preços
# ----------------------------------------------------------------------------------------------------
def baixar_yahoo(tickers):
    """Baixa o fechamento ajustado do Yahoo. Entrada: tickers sem '.SA'. Saída: DataFrame (colunas = tickers)."""
    import yfinance as yf
    simbolos = [t + ".SA" for t in tickers]
    df = yf.download(simbolos, start="2024-03-20", end="2026-07-03", auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(simbolos[0])
    df.columns = [c.replace(".SA", "") for c in df.columns]
    return df


def carregar_precos(tickers):
    """Lê precos.csv se existir; senão baixa do Yahoo (preços ajustados por dividendos).
    precos.csv: coluna Date + uma coluna por ticker SEM '.SA' + coluna IBOV."""
    usar_csv = False
    if os.path.exists("precos.csv"):
        df = pd.read_csv("precos.csv", index_col="Date", parse_dates=True)
        usar_csv = all(t in df.columns for t in tickers + ["IBOV"])
        if not usar_csv:
            print("[AVISO] precos.csv não tem todos os tickers da carteira: baixando do Yahoo")
    if not usar_csv:
        import yfinance as yf
        df = baixar_yahoo(tickers)
        ibov = yf.download("^BVSP", start="2024-03-20", end="2026-07-03", auto_adjust=True)["Close"]
        df["IBOV"] = ibov.squeeze()
        # tickers sem preço na formação: tenta o ticker novo da mesma empresa
        sem_preco = [t for t in tickers if t not in df.columns or df.loc[:DATA_FORMACAO, t].dropna().empty]
        trocas = {t: TICKER_NOVO_YAHOO[t] for t in sem_preco if t in TICKER_NOVO_YAHOO}
        if trocas:
            novos = baixar_yahoo(list(trocas.values()))
            for antigo, novo in trocas.items():
                df[antigo] = novos[novo]
                print(f"[AVISO] {antigo} sem preço no Yahoo: usando o histórico de {novo}")
    df = df.sort_index().loc[DATA_FORMACAO:DATA_FIM]
    df = df.dropna(subset=["IBOV"])                     # só pregões da B3
    # ação que deixou de ser negociada no meio do período: o valor fica parado no último preço (vira caixa)
    for t in tickers:
        ultimo = df[t].last_valid_index() if t in df else None
        if ultimo is not None and ultimo < df.index[-1]:
            print(f"[AVISO] {t}: último preço em {ultimo.date()}; daí em diante o valor fica parado")
    df = df.ffill()
    return df


# ----------------------------------------------------------------------------------------------------
# 4.3 Valor da carteira (compra e mantém, sem rebalanceamento)
# ----------------------------------------------------------------------------------------------------
def valor_da_carteira(precos, pesos):
    """Começa em 1,00 na formação. Cada ativo cresce com seu preço; o peso inicial vira o tamanho da posição."""
    base = precos[pesos.index] / precos[pesos.index].iloc[0]
    return (base * pesos).sum(axis=1)


# ----------------------------------------------------------------------------------------------------
# 4.4 Métricas
# ----------------------------------------------------------------------------------------------------
def retorno_acumulado(serie):
    return serie.iloc[-1] / serie.iloc[0] - 1


def volatilidade_anual(serie):
    return serie.pct_change().dropna().std() * np.sqrt(DIAS_ANO)


def sharpe(serie):
    """Sharpe sem taxa livre de risco: retorno médio diário anualizado ÷ volatilidade anualizada."""
    r = serie.pct_change().dropna()
    return (r.mean() * DIAS_ANO) / (r.std() * np.sqrt(DIAS_ANO))


def drawdown(serie):
    """Queda em relação ao maior valor já atingido até cada dia (0% = no pico)."""
    return serie / serie.cummax() - 1


def drawdown_maximo(serie):
    """Pior queda a partir de um pico em todo o período."""
    return drawdown(serie).min()


# ----------------------------------------------------------------------------------------------------
# 4.5 Execução da carteira
# ----------------------------------------------------------------------------------------------------
def rodar_carteira(fator_1, fator_2, fator_3):
    """Monta a carteira com as listas que saíram de cada fator e compara com o Ibovespa.
    Resultados na pasta do projeto: carteira_pesos.csv, carteira_resultados.csv, carteira_evolucao.csv,
    carteira_grafico.png e carteira_drawdown.png."""
    print("\n" + "=" * 100 + "\nCARTEIRA DE FACTOR INVESTING vs IBOVESPA\n" + "=" * 100)
    print("Fator 1:", fator_1)
    print("Fator 2:", fator_2)
    print("Fator 3:", fator_3)

    p1, p2, p3, combinada = montar_carteira_combinada(fator_1, fator_2, fator_3)
    print("\nPesos da carteira combinada:")
    print((combinada.sort_values(ascending=False) * 100).round(2).astype(str) + "%")
    print("Soma dos pesos:", round(combinada.sum(), 6))
    pesos = pd.DataFrame({"Fator 1": p1, "Fator 2": p2, "Fator 3": p3, "Carteira combinada": combinada}).fillna(0)
    pesos.sort_values("Carteira combinada", ascending=False).to_csv(
        "carteira_pesos.csv", sep=";", decimal=",", encoding="utf-8-sig", index_label="ticker")

    todos = sorted(set(fator_1 + fator_2 + fator_3))
    precos = carregar_precos(todos)
    faltando = precos.columns[precos.iloc[0].isna()]
    if len(faltando) > 0:
        print("ATENÇÃO: sem preço em 28/03:", list(faltando))
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
        linhas[nome] = {
            "Retorno acumulado": retorno_acumulado(s),
            "Volatilidade anualizada": volatilidade_anual(s),
            "Sharpe": sharpe(s),
            "Drawdown máximo": drawdown_maximo(s),
        }
    tabela = pd.DataFrame(linhas)
    print("\n", tabela.round(4))

    tabela.to_csv("carteira_resultados.csv")
    pd.DataFrame(series).to_csv("carteira_evolucao.csv")

    # retorno de cada ação no período (usado no slide de explicações do Fator 3)
    retorno_acoes = (precos[todos].iloc[-1] / precos[todos].iloc[0] - 1).rename("retorno_no_periodo")
    retorno_acoes = pd.concat([retorno_acoes, pesos], axis=1).sort_values("retorno_no_periodo")
    retorno_acoes.to_csv("carteira_retorno_por_acao.csv", sep=";", decimal=",", encoding="utf-8-sig", index_label="ticker")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.DataFrame(series).plot(figsize=(10, 5), title="Carteiras por fator vs Ibovespa (base 1,00 em 28/03/2024)")
    plt.ylabel("Valor da carteira")
    plt.tight_layout()
    plt.savefig("carteira_grafico.png", dpi=150)
    plt.close()

    pd.DataFrame({n: drawdown(s) for n, s in series.items()}).plot(figsize=(10, 5), title="Drawdown: queda em relação ao pico anterior")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig("carteira_drawdown.png", dpi=150)
    plt.close()
    print("\nArquivos da carteira salvos na pasta do projeto: carteira_pesos.csv, carteira_resultados.csv, "
          "carteira_evolucao.csv, carteira_retorno_por_acao.csv, carteira_grafico.png e carteira_drawdown.png")


# ######################################################################################################
# EXECUÇÃO: roda os três fatores e depois a carteira
# ######################################################################################################
def main():
    PASTA_OUTPUTS.mkdir(exist_ok=True)
    ibov = carregar_universo()
    fator_1 = rodar_fator1(ibov)
    fator_2 = rodar_fator2(ibov)
    fator_3 = rodar_fator3(ibov)
    rodar_carteira(fator_1, fator_2, fator_3)


if __name__ == "__main__":
    main()