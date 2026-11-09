"""
RANKING DE ESCOLARIDADE DOS CEOs  (versão simples)
Critério: 2017 a dez/2023  |  Empresas: as que estavam no Ibovespa no início de 2024

IDEIA EM 5 PASSOS
  1. Baixar da CVM o Formulário de Referência (FRE) de cada ano, de 2017 a 2023.
  2. Ficar só com os CEOs (Diretor Presidente).
  3. Ler o currículo de cada CEO e dar pontos pela formação:
         graduação = 1 | MBA ou especialização = 1 | mestrado = 2 | doutorado = 3   (máximo 7)
  4. Nota da empresa em cada ano = média dos CEOs daquele ano.  Nota final = média dos anos.
  5. Ordenar as empresas pela nota final e guardar o resultado numa planilha.

ARQUIVOS NA MESMA PASTA:  ibovespa_2024.csv (obrigatório)  e  ajustes_manuais.csv (opcional).
COMO RODAR NO JUPYTER:    %run ranking_ceo_simples.py     (ou colar tudo numa célula e rodar)

O QUE FOI USADO (para estudar)
  pandas:  read_csv, DataFrame, groupby, agg, merge, join, drop_duplicates, sort_values, rank, isin, str.contains,
           map, to_excel, ExcelWriter
  re:      busca de palavras no texto (expressões regulares simples)
  zipfile: abrir o arquivo .zip da CVM
  requests: baixar o arquivo da internet
  Python:  def, for, if, dict (dicionário), list (lista), f-string, sum, sorted, set
"""
import io
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------------------------------
# CONFIGURAÇÃO (o que o grupo pode mudar)
# ----------------------------------------------------------------------------------------------------
ANOS = list(range(2017, 2024))                                     # 2017, 2018, ..., 2023
PESOS = {"graduacao": 1, "mba": 1, "mestrado": 2, "doutorado": 3}  # pontos de cada nível de formação
TOP_N = 10                                                         # tamanho da carteira
ARQ_IBOV = "ibovespa_2024.csv"                                     # lista das empresas do Ibovespa em jan/2024
ARQ_AJUSTES = "ajustes_manuais.csv"                                # correções feitas à mão (opcional)
PASTA_ZIPS = Path("dados_cvm")                                     # onde ficam os arquivos baixados
SAIDA = "ranking_escolaridade_ceo.xlsx"
URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"


# ----------------------------------------------------------------------------------------------------
# FUNÇÃO 1: limpar texto
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
# FUNÇÃO 2: baixar os dados da CVM
# ----------------------------------------------------------------------------------------------------
def baixar_fre():
    """Para cada ano: baixa o .zip (se ainda não estiver na pasta dados_cvm), abre a tabela de administradores
    e junta tudo numa tabela só. A coluna 'ano' guarda de qual ano veio cada linha."""
    PASTA_ZIPS.mkdir(exist_ok=True)
    tabelas = []
    for ano in ANOS:
        arquivo = PASTA_ZIPS / f"fre_cia_aberta_{ano}.zip"
        if not arquivo.exists():
            import requests
            print(f"Baixando FRE {ano}...")
            resposta = requests.get(URL.format(ano=ano), timeout=300)
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
# FUNÇÃO 3: achar os CEOs
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
# FUNÇÃO 4: ler a formação no currículo
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
# FUNÇÃO 5: nota e ranking das empresas
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
# FUNÇÃO 6: programa principal
# ----------------------------------------------------------------------------------------------------
def main():
    # --- 1. dados de entrada
    ibov = pd.read_csv(ARQ_IBOV, sep=";", dtype=str, encoding="utf-8-sig")   # ticker; empresa; cnpj
    adm = baixar_fre()
    ceos = achar_ceos(adm)

    # --- 2. formação de cada pessoa (lendo TODOS os currículos que ela tem, de todos os anos)
    textos = ceos.groupby("pessoa").agg(curriculo=("curriculo", lambda s: " || ".join(s.dropna().unique())),
                                        profissao=("profissao", lambda s: " ".join(s.dropna().unique())))
    formacao = pd.DataFrame([ler_formacao(l.curriculo, l.profissao) for l in textos.itertuples()],
                            index=textos.index)

    # --- 3. ajustes manuais (opcional): só ACRESCENTAM títulos confirmados pelo grupo
    if Path(ARQ_AJUSTES).exists():
        ajustes = pd.read_csv(ARQ_AJUSTES, sep=";", dtype=str, encoding="utf-8-sig", encoding_errors="replace")
        aplicados = 0
        for linha in ajustes.itertuples():
            pessoa = sem_acento(linha.pessoa).upper()
            if pessoa in formacao.index:
                for nivel in PESOS:
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
    base = pontuar_empresas(tabela, ANOS, PESOS)
    tickers = ibov.groupby("empresa")["ticker"].agg(lambda s: ", ".join(sorted(set(s))))
    ranking = base.join(tickers, how="right").sort_values("posicao")         # 'right' mantém empresas sem dado
    ranking["n_anos"] = ranking["n_anos"].fillna(0).astype(int)
    ranking["valida"] = ranking["valida"].fillna(False)

    # --- 6. quem é cada CEO (para conferir) e quem ficou sem nenhum título detectado (para revisar)
    tabela["nota_pessoa"] = sum(tabela[nivel] * pontos for nivel, pontos in PESOS.items())
    ceos_por_empresa = tabela.groupby(["empresa", "pessoa"]).agg(
        anos=("ano", lambda s: ", ".join(str(a) for a in sorted(set(s)))),
        graduacao=("graduacao", "max"), mba=("mba", "max"), mestrado=("mestrado", "max"),
        doutorado=("doutorado", "max"), nota_pessoa=("nota_pessoa", "max")).reset_index()
    revisar = ceos_por_empresa[ceos_por_empresa["nota_pessoa"] == 0]

    # --- 7. gravar a planilha
    with pd.ExcelWriter(SAIDA) as arquivo:
        ranking.reset_index().to_excel(arquivo, sheet_name="ranking_empresas", index=False)
        ceos_por_empresa.to_excel(arquivo, sheet_name="ceos_por_empresa", index=False)
        revisar.to_excel(arquivo, sheet_name="revisar_sem_titulo", index=False)

    print(f"\nTop {TOP_N} (critério 2017-2023, Ibovespa jan/2024):")
    print(ranking.head(TOP_N)[["posicao", "ticker", "nota", "n_anos"]].to_string())
    print(f"\nPlanilha gerada: {SAIDA}")


if __name__ == "__main__":
    main()