"""
Teste estatístico dos 3 fatores do grupo sobre o desempenho das ações listadas (acima do Ibovespa?)
====================================================================================================
Fatores
  1) Escolaridade do CEO   – pontuação do CEO vigente (base gerada por base_ceo_cvm.py)
  2) Baixa volatilidade    – desvio-padrão dos retornos diários nos 12 meses anteriores
  3) Capital humano        – valor adicionado por empregado (DVA ÷ nº de empregados) e
                             participação do pessoal no valor adicionado (DVA)

Desenho (anual, sem look-ahead grosseiro)
  Em 30/jun de cada ano t (2017–2025) os sinais são medidos com o que já era público:
    escolaridade = CEO vigente em 31/mai/t; volatilidade = jul/(t-1)–jun/t;
    capital humano = DVA do exercício t-1 (DFP entregue até mar/t) e empregados do FRE de t.
  Variável dependente = retorno da ação de jul/t a jun/(t+1) MENOS o retorno do Ibovespa
  no mesmo período (retorno em excesso ao Ibovespa).

Testes
  A) Regressão em painel (empresa-ano), fatores padronizados (z-score por ano), efeitos fixos
     de ano e setor, erros-padrão clusterizados por empresa: modelos com cada fator sozinho e
     com os três juntos.
  B) Fama-MacBeth: uma regressão cross-section por ano; média dos coeficientes.
  C) Carteiras: top 20% de cada fator e do score combinado (pesos iguais, rebalanceamento anual);
     retorno mensal em excesso ao Ibovespa, t de Newey-West e alfa/beta contra o Ibovespa.

Uso (na mesma pasta de base_ceo_cvm.py, depois de rodá-lo)
    pip install pandas numpy requests openpyxl statsmodels yfinance matplotlib
    python regressao_fatores.py                       # lê a base em saida_ceo/
    python regressao_fatores.py --base ceo_escolaridade
    No Jupyter:  %run regressao_fatores.py --base ceo_escolaridade
Opcional: --depara arquivo.csv (colunas cnpj;ticker) para usar um de-para próprio.
"""

from __future__ import annotations

import argparse
import io
import pickle
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VERSAO = "2026-10-20b regressões dos 3 fatores + top 10 pela média do período"
CACHE = Path("dados_cvm")
URL_FCA = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ano}.zip"
URL_DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{ano}.zip"
URL_FRE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"
URL_CDI = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json"  # CDI acumulado no mês (%)
LOG: list[str] = []


def log(msg):
    print(msg)
    LOG.append(msg)


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------
def norm(t) -> str:
    if pd.isna(t):
        return ""
    t = unicodedata.normalize("NFKD", str(t))
    return re.sub(r"\s+", " ", "".join(c for c in t if not unicodedata.combining(c))).strip().lower()


def col(df, cands, obrig=True):
    mapa = {norm(c).replace(" ", "_"): c for c in df.columns}
    for c in cands:
        k = norm(c).replace(" ", "_")
        if k in mapa:
            return mapa[k]
    for c in cands:
        k = norm(c).replace(" ", "_")
        for kk, orig in mapa.items():
            if k in kk:
                return orig
    if obrig:
        raise KeyError(f"coluna {cands} não encontrada em {list(df.columns)}")
    return None


def ler_csv(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), sep=";", encoding="latin-1", dtype=str, on_bad_lines="skip", low_memory=False)


def cnpj14(s):
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(14)


def baixar_zip(url: str, nome: str, ano: int) -> zipfile.ZipFile | None:
    """Baixa (ou lê da cópia local em dados_cvm/) um zip da CVM."""
    CACHE.mkdir(exist_ok=True)
    arq = CACHE / nome
    if not (arq.exists() and ano < pd.Timestamp.today().year):
        import requests
        print(f"Baixando {url} ...")
        try:
            r = requests.get(url, timeout=300)
            ok = r.status_code == 200
        except Exception as e:  # noqa: BLE001
            ok, r = False, None
            log(f"  {nome}: falha no download ({type(e).__name__})")
        if ok:
            arq.write_bytes(r.content)
        elif not arq.exists():
            if r is not None:
                log(f"  {nome}: HTTP {r.status_code} – ignorado")
            return None
    return zipfile.ZipFile(arq)


def ultima_versao(df, c_cnpj, c_ref, c_ver):
    df = df.copy()
    df["_v"] = pd.to_numeric(df[c_ver], errors="coerce")
    mx = df.groupby([c_cnpj, c_ref])["_v"].transform("max")
    return df[df["_v"] == mx].drop(columns="_v")


# --------------------------------------------------------------------------
# 1. Base do grupo (escolaridade, setor, empresas listadas)
# --------------------------------------------------------------------------
def carregar_base(pasta: Path):
    def rd(n):
        p = pasta / f"{n}.csv"
        if not p.exists():
            sys.exit(f"Arquivo {p} não encontrado. Rode antes o base_ceo_cvm.py (ou indique --base).")
        return pd.read_csv(p, sep=";", encoding="utf-8-sig", dtype={"cnpj": str, "pessoa_id": str}, low_memory=False)

    painel, form, emp = rd("painel_mensal"), rd("ceo_formacao"), rd("empresas")
    if "pontuacao_escolaridade" not in form:
        sys.exit("ceo_formacao.csv não tem 'pontuacao_escolaridade': rode a versão nova do base_ceo_cvm.py.")
    for d in (painel, emp):
        d["cnpj"] = cnpj14(d["cnpj"])
    painel["data"] = pd.to_datetime(painel["data"])
    if "listada_bolsa" in emp:
        emp["listada_bolsa"] = emp["listada_bolsa"].astype(str).str.lower().isin(["true", "1"])
    else:
        emp["listada_bolsa"] = emp.get("tipo_mercado", pd.Series("", index=emp.index)).fillna("").str.upper().str.contains("BOLSA")
    log(f"Base do grupo: {len(emp):,} companhias, {emp['listada_bolsa'].sum():,} listadas em bolsa")
    return painel, form, emp


def sinal_escolaridade(painel, form, anos):
    pts = form.set_index("pessoa_id")["pontuacao_escolaridade"]
    out = []
    for t in anos:
        s = painel[(painel["data"] == pd.Timestamp(t, 5, 31)) & painel["pessoa_id"].notna()][["cnpj", "pessoa_id"]]
        s = s.assign(ano=t, escolaridade=s["pessoa_id"].map(pts))
        out.append(s)
    return pd.concat(out)[["cnpj", "ano", "escolaridade"]].dropna()


# --------------------------------------------------------------------------
# 2. De-para CNPJ -> ticker (FCA da CVM)
# --------------------------------------------------------------------------
RE_TICKER = re.compile(r"^[A-Z]{4}(3|4|5|6|11)$")


def depara_fca(anos_fca):
    linhas = []
    for ano in anos_fca:
        z = baixar_zip(URL_FCA.format(ano=ano), f"fca_cia_aberta_{ano}.zip", ano)
        if z is None:
            continue
        nomes = [n for n in z.namelist() if "valor_mobiliario" in n.lower()]
        if not nomes:
            log(f"  FCA {ano}: tabela de valores mobiliários não encontrada ({z.namelist()})")
            continue
        df = ler_csv(z.read(nomes[0]))
        c_cnpj = col(df, ["CNPJ_Companhia", "CNPJ_CIA"])
        c_tk = col(df, ["Codigo_Negociacao"])
        c_merc = col(df, ["Mercado"], obrig=False)
        d = df[[c_cnpj, c_tk] + ([c_merc] if c_merc else [])].copy()
        if c_merc:  # só bolsa
            d = d[d[c_merc].map(norm).str.contains("bolsa")]
        d = d.assign(ticker=d[c_tk].fillna("").str.upper().str.split(r"[,;/ ]+")).explode("ticker")
        d["ticker"] = d["ticker"].str.strip()
        d = d[d["ticker"].str.match(RE_TICKER)]
        linhas.append(pd.DataFrame({"cnpj": cnpj14(d[c_cnpj]), "ticker": d["ticker"]}))
    if not linhas:
        sys.exit("Não foi possível montar o de-para CNPJ → ticker pelo FCA. Use --depara arquivo.csv (cnpj;ticker).")
    dp = pd.concat(linhas).drop_duplicates()
    log(f"De-para FCA: {dp['cnpj'].nunique():,} companhias, {len(dp):,} tickers candidatos")
    return dp


# --------------------------------------------------------------------------
# 3. Preços (Yahoo Finance)
# --------------------------------------------------------------------------
def baixar_precos(tickers, inicio="2015-06-01"):
    import yfinance as yf
    arq = CACHE / "precos_yahoo.pkl"
    if arq.exists():
        px, vol = pickle.loads(arq.read_bytes())
        faltam = sorted(set(tickers) - set(px.columns))
        if not faltam and px.index.max() >= pd.Timestamp.today().normalize() - pd.Timedelta(days=7):
            return px, vol
    px_l, vol_l = [], []
    lista = sorted(set(tickers)) + ["^BVSP"]
    for i in range(0, len(lista), 80):
        lote = lista[i:i + 80]
        print(f"Baixando preços {i + 1}–{i + len(lote)} de {len(lista)} ...")
        d = yf.download([t if t.startswith("^") else t + ".SA" for t in lote], start=inicio,
                        auto_adjust=True, progress=False, threads=True, group_by="column")
        if d is None or d.empty:
            continue
        c, v = d["Close"], d["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(lote[0]), v.to_frame(lote[0])
        c.columns = [str(x).replace(".SA", "") for x in c.columns]
        v.columns = [str(x).replace(".SA", "") for x in v.columns]
        px_l.append(c)
        vol_l.append(v)
    px = pd.concat(px_l, axis=1)
    vol = pd.concat(vol_l, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].dropna(how="all", axis=1)
    vol = vol.loc[:, ~vol.columns.duplicated()].reindex(columns=px.columns)
    px.index = pd.to_datetime(px.index).tz_localize(None)
    vol.index = px.index
    arq.write_bytes(pickle.dumps((px, vol)))
    return px, vol


def escolher_ticker(dp, px, vol):
    """Para cada CNPJ, o ticker com maior volume financeiro mediano (ON, PN ou Unit)."""
    vf = (px * vol).median()
    d = dp[dp["ticker"].isin(px.columns)].copy()
    d["vf"] = d["ticker"].map(vf)
    d = d.sort_values("vf", ascending=False).drop_duplicates("cnpj")
    log(f"Tickers com preço no Yahoo: {len(d):,} companhias")
    return d[["cnpj", "ticker"]]


def sinais_preco(px, vol, anos, min_obs=200):
    r = px.pct_change(fill_method=None)
    out = []
    for t in anos:
        ini, fim = pd.Timestamp(t - 1, 6, 30), pd.Timestamp(t, 6, 30)
        j = r[(r.index > ini) & (r.index <= fim)]
        n = j.notna().sum()
        vola = (j.std() * np.sqrt(252)).where(n >= min_obs)
        jv = (px * vol)[(px.index > ini) & (px.index <= fim)]
        liq = jv.mean()
        out.append(pd.DataFrame({"ticker": vola.index, "ano": t, "volatilidade": vola.values,
                                 "liq_media_rs": liq.reindex(vola.index).values}))
    return pd.concat(out)


def retorno_anual(px, anos):
    """Retorno buy-and-hold de 30/jun/t a 30/jun/t+1 (última cotação disponível se a ação parar de negociar)."""
    out = []
    ult = px.index.max()
    for t in anos:
        a, b = pd.Timestamp(t, 6, 30), pd.Timestamp(t + 1, 6, 30)
        if b > ult:
            continue
        p0 = px[px.index <= a].ffill().iloc[-1]
        jan = px[(px.index > a) & (px.index <= b)]
        p1 = jan.ffill().iloc[-1]
        ret = p1 / p0 - 1
        out.append(pd.DataFrame({"ticker": ret.index, "ano": t, "ret": ret.values}))
    return pd.concat(out)


# --------------------------------------------------------------------------
# 4. Capital humano: DVA (DFP) e empregados (FRE)
# --------------------------------------------------------------------------
def dva_anual(anos_exerc):
    out = []
    for ano in anos_exerc:
        z = baixar_zip(URL_DFP.format(ano=ano), f"dfp_cia_aberta_{ano}.zip", ano)
        if z is None:
            continue
        nomes = z.namelist()
        con = [n for n in nomes if "dva_con" in n.lower()]
        ind = [n for n in nomes if "dva_ind" in n.lower()]
        partes = []
        for nm, tipo in [(con, "con"), (ind, "ind")]:
            if not nm:
                continue
            df = ler_csv(z.read(nm[0]))
            c = {k: col(df, v) for k, v in {"cnpj": ["CNPJ_CIA"], "ref": ["DT_REFER"], "ver": ["VERSAO"],
                                            "ordem": ["ORDEM_EXERC"], "conta": ["CD_CONTA"], "ds": ["DS_CONTA"],
                                            "vl": ["VL_CONTA"], "esc": ["ESCALA_MOEDA"]}.items()}
            df = df[df[c["ordem"]].map(norm).str.startswith("ultimo")]
            df = ultima_versao(df, c["cnpj"], c["ref"], c["ver"])
            vl = pd.to_numeric(df[c["vl"]].str.replace(",", "."), errors="coerce")
            vl = vl * np.where(df[c["esc"]].map(norm).str.startswith("mil"), 1000, 1)
            ds = df[c["ds"]].map(norm)
            conta = df[c["conta"]].astype(str)
            is_va = (conta == "7.07") | ds.str.contains("valor adicionado total a distribuir")
            is_pes = (conta == "7.08.01") | ((conta.str.count(r"\.") == 2) & conta.str.startswith("7.08")
                                             & ds.str.match(r"^pessoal"))
            d = pd.DataFrame({"cnpj": cnpj14(df[c["cnpj"]]), "va": vl.where(is_va), "pessoal": vl.where(is_pes)})
            d = d.groupby("cnpj").agg(va=("va", "first"), pessoal=("pessoal", "first")).reset_index()
            d["tipo_dva"] = tipo
            partes.append(d)
        if partes:  # consolidado tem preferência
            d = pd.concat(partes).dropna(subset=["va"]).drop_duplicates("cnpj", keep="first")
            d["exercicio"] = ano
            out.append(d)
            log(f"  DVA {ano}: {len(d):,} companhias")
    return pd.concat(out) if out else pd.DataFrame(columns=["cnpj", "va", "pessoal", "exercicio"])


RE_QTD = re.compile(r"quantidade|numero|qtd|total|empregados")
RE_NAO_QTD = re.compile(r"id_|percent|indice|taxa|versao|codigo|cnpj|ano|data")


def empregados_fre(anos):
    """Nº total de empregados por companhia e ano do FRE (tabelas de recursos humanos)."""
    out = []
    for ano in anos:
        z = baixar_zip(URL_FRE.format(ano=ano), f"fre_cia_aberta_{ano}.zip", ano)
        if z is None:
            continue
        cands = [n for n in z.namelist() if re.search(r"empregad|recursos_humanos", n.lower())]
        melhor = None
        for nm in cands:
            df = ler_csv(z.read(nm))
            c_cnpj = col(df, ["CNPJ_Companhia", "CNPJ_CIA"], obrig=False)
            c_ref = col(df, ["Data_Referencia", "DT_REFER"], obrig=False)
            c_ver = col(df, ["Versao"], obrig=False)
            if not (c_cnpj and c_ref and c_ver):
                continue
            num = [c for c in df.columns if RE_QTD.search(norm(c)) and not RE_NAO_QTD.search(norm(c))]
            num = [c for c in num if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
            if not num:
                continue
            tot = [c for c in num if "total" in norm(c)]
            usar = tot[:1] or num
            df = ultima_versao(df, c_cnpj, c_ref, c_ver)
            q = df[usar].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            d = pd.DataFrame({"cnpj": cnpj14(df[c_cnpj]), "empregados": q}).groupby("cnpj")["empregados"].sum()
            d = d[d > 0]
            if melhor is None or len(d) > len(melhor[0]):
                melhor = (d, nm, usar)
        if melhor is None:
            log(f"  FRE {ano}: tabela de empregados não encontrada (arquivos: {cands or 'nenhum'})")
            continue
        d, nm, usar = melhor
        pb = d.get("33000167000101", np.nan)
        log(f"  FRE {ano}: empregados de {len(d):,} companhias – {Path(nm).name}, colunas {usar} "
            f"(conferência: Petrobras = {pb:,.0f})")
        out.append(pd.DataFrame({"cnpj": d.index, "ano": ano, "empregados": d.values}))
    return pd.concat(out) if out else pd.DataFrame(columns=["cnpj", "ano", "empregados"])


def sinal_capital_humano(dva, emp, anos):
    out = []
    for t in anos:
        d = dva[dva["exercicio"] == t - 1][["cnpj", "va", "pessoal"]]
        e = emp[emp["ano"] == t][["cnpj", "empregados"]]
        d = d.merge(e, on="cnpj", how="left").assign(ano=t)
        out.append(d)
    ch = pd.concat(out)
    ch["va_por_empregado"] = ch["va"] / ch["empregados"].where(ch["empregados"] >= 20)
    ch["ln_va_por_empregado"] = np.log(ch["va_por_empregado"].where(ch["va_por_empregado"] > 0))
    ch["pessoal_sobre_va"] = (ch["pessoal"] / ch["va"].where(ch["va"] > 0)).clip(-1, 2)
    ch["ln_va"] = np.log(ch["va"].where(ch["va"] > 0))
    return ch


# --------------------------------------------------------------------------
# 5. Testes estatísticos
# --------------------------------------------------------------------------
def zscore_ano(df, c):
    g = df.groupby("ano")[c]
    return (df[c] - g.transform("mean")) / g.transform("std")


def winsor(s, p=0.01):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


FATORES = {"z_escolaridade": "Escolaridade do CEO (+1 dp)",
           "z_volatilidade": "Volatilidade (+1 dp)",
           "z_capital_humano": "Capital humano: ln(VA/empregado) (+1 dp)",
           "z_pessoal_va": "Capital humano: pessoal/VA (+1 dp)"}
SINAL_ESPERADO = {"z_escolaridade": "+", "z_volatilidade": "−", "z_capital_humano": "+", "z_pessoal_va": "+"}


def regressoes_painel(base):
    import statsmodels.formula.api as smf
    modelos = {
        "(1) Escolaridade": ["z_escolaridade"],
        "(2) Volatilidade": ["z_volatilidade"],
        "(3) Capital humano": ["z_capital_humano"],
        "(3b) Pessoal/VA": ["z_pessoal_va"],
        "(4) Três fatores": ["z_escolaridade", "z_volatilidade", "z_capital_humano"],
        "(5) Três fatores + controles": ["z_escolaridade", "z_volatilidade", "z_capital_humano", "ln_liq", "ln_va"],
    }
    linhas, resumo = [], []
    for nome, xs in modelos.items():
        d = base.dropna(subset=["exc_ret"] + xs + ["setor"])
        if len(d) < 50 or d["cnpj"].nunique() < 20:
            log(f"  {nome}: poucas observações ({len(d)}) – modelo não estimado")
            continue
        f = "exc_ret ~ " + " + ".join(xs) + " + C(ano) + C(setor)"
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cnpj"])[0]})
        for x in xs:
            linhas.append({"modelo": nome, "variavel": FATORES.get(x, x), "coeficiente": m.params[x],
                           "erro_padrao": m.bse[x], "t": m.tvalues[x], "p_valor": m.pvalues[x],
                           "sinal_esperado": SINAL_ESPERADO.get(x, ""), "significativo_5pct": m.pvalues[x] < 0.05})
        resumo.append({"modelo": nome, "n_obs": int(m.nobs), "n_empresas": d["cnpj"].nunique(),
                       "anos": f"{d['ano'].min()}–{d['ano'].max()}", "r2": m.rsquared, "r2_ajustado": m.rsquared_adj})
    return pd.DataFrame(linhas), pd.DataFrame(resumo)


def fama_macbeth(base, xs=("z_escolaridade", "z_volatilidade", "z_capital_humano")):
    import statsmodels.api as sm
    xs = list(xs)
    coefs = []
    for t, d in base.dropna(subset=["exc_ret"] + xs).groupby("ano"):
        if len(d) < 20:
            continue
        m = sm.OLS(d["exc_ret"], sm.add_constant(d[xs])).fit()
        coefs.append(m.params.rename(t))
    if len(coefs) < 3:
        return pd.DataFrame()
    c = pd.DataFrame(coefs)
    T = len(c)
    res = pd.DataFrame({"coef_medio": c.mean(), "erro_padrao": c.std(ddof=1) / np.sqrt(T)})
    res["t"] = res["coef_medio"] / res["erro_padrao"]
    from scipy import stats
    res["p_valor"] = 2 * stats.t.sf(res["t"].abs(), df=T - 1)
    res["n_anos"] = T
    res.index = [FATORES.get(i, i) for i in res.index]
    anual = c.copy()
    anual.columns = [FATORES.get(i, i) for i in anual.columns]
    return res.reset_index(names="variavel"), anual.reset_index(names="ano")


def carteiras(base, ret_m, ibov_m, cdi_m, top=0.2):
    import statsmodels.api as sm
    b = base.copy()
    b["z_combinado"] = b[["z_escolaridade", "z_baixa_vol", "z_capital_humano"]].mean(axis=1, skipna=True)
    b.loc[b[["z_escolaridade", "z_baixa_vol", "z_capital_humano"]].notna().sum(axis=1) < 2, "z_combinado"] = np.nan
    criterios = {"Escolaridade do CEO (top 20%)": "z_escolaridade", "Baixa volatilidade (top 20%)": "z_baixa_vol",
                 "Capital humano (top 20%)": "z_capital_humano", "Combinado 3 fatores (top 20%)": "z_combinado"}
    series, membros, spreads = {}, [], {}
    for nome, c in criterios.items():
        partes, sp = [], []
        for t, d in b.dropna(subset=[c]).groupby("ano"):
            if len(d) < 15:
                continue
            q_hi, q_lo = d[c].quantile(1 - top), d[c].quantile(top)
            alto, baixo = d[d[c] >= q_hi]["ticker"], d[d[c] <= q_lo]["ticker"]
            meses = ret_m.loc[(ret_m.index > pd.Timestamp(t, 6, 30)) & (ret_m.index <= pd.Timestamp(t + 1, 6, 30))]
            ra, rb = meses[alto].mean(axis=1), meses[baixo].mean(axis=1)
            partes.append(ra)
            sp.append(ra - rb)
            membros += [{"carteira": nome, "ano_formacao": t, "ticker": x} for x in alto]
        if partes:
            series[nome] = pd.concat(partes)
            spreads[nome] = pd.concat(sp)
    rm = pd.DataFrame(series)
    rm["Ibovespa"] = ibov_m.reindex(rm.index)
    resumo = []
    for nome in criterios:
        if nome not in rm:
            continue
        r = rm[nome].dropna()
        ib = rm.loc[r.index, "Ibovespa"]
        exc = r - ib
        n = len(r)
        mt = sm.OLS(exc, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        capm = sm.OLS(r, sm.add_constant(ib)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        spr = spreads[nome].dropna()
        ms = sm.OLS(spr, np.ones(len(spr))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        linha = {"carteira": nome, "meses": n, "inicio": r.index.min(), "fim": r.index.max(),
                 "retorno_aa_carteira": (1 + r).prod() ** (12 / n) - 1,
                 "retorno_aa_ibovespa": (1 + ib).prod() ** (12 / n) - 1,
                 "excesso_medio_mensal": exc.mean(), "t_excesso (Newey-West)": mt.tvalues.iloc[0],
                 "p_excesso": mt.pvalues.iloc[0],
                 "alfa_mensal_vs_ibov": capm.params.iloc[0], "p_alfa": capm.pvalues.iloc[0],
                 "beta_vs_ibov": capm.params.iloc[1],
                 "vol_aa": r.std() * np.sqrt(12),
                 "pct_meses_acima_ibov": (exc > 0).mean() * 100,
                 "max_drawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),
                 "spread_top_menos_bottom_mensal": spr.mean(), "p_spread": ms.pvalues.iloc[0]}
        if cdi_m is not None:
            ex_rf = r - cdi_m.reindex(r.index).fillna(0)
            linha["sharpe"] = ex_rf.mean() / ex_rf.std() * np.sqrt(12)
        resumo.append(linha)
    ib = rm["Ibovespa"].dropna()
    resumo.append({"carteira": "Ibovespa", "meses": len(ib), "retorno_aa_carteira": (1 + ib).prod() ** (12 / len(ib)) - 1,
                   "vol_aa": ib.std() * np.sqrt(12),
                   "max_drawdown": ((1 + ib).cumprod() / (1 + ib).cumprod().cummax() - 1).min(),
                   **({"sharpe": (ib - cdi_m.reindex(ib.index).fillna(0)).mean() /
                       (ib - cdi_m.reindex(ib.index).fillna(0)).std() * np.sqrt(12)} if cdi_m is not None else {})})
    return pd.DataFrame(resumo), rm, pd.DataFrame(membros)


def top_n_media_periodo(base, ret_m, ibov_m, cdi_m, emp_nome, n=10, min_anos=5, min_meses_pct=0.6):
    """Top N empresas válidas de cada fator pela MÉDIA do fator no período inteiro.

    Válida = fator medido em pelo menos `min_anos` anos (com liquidez mínima) e cotação em pelo menos
    `min_meses_pct` dos meses da janela de retornos. Carteira de pesos iguais, rebalanceada todo mês entre as
    empresas que ainda negociam, mantida durante toda a janela (jul do 1º ano até jun do último + 1).
    Atenção: usa a média de anos futuros para escolher as empresas (análise descritiva, não estratégia investível).
    """
    import statsmodels.api as sm
    anos = sorted(base["ano"].unique())
    ini, fim = pd.Timestamp(anos[0], 6, 30), pd.Timestamp(anos[-1] + 1, 6, 30)
    jan = ret_m.loc[(ret_m.index > ini) & (ret_m.index <= fim)]
    cobertura = jan.notna().mean()
    g = base.groupby(["cnpj", "ticker"])
    med = g.agg(escolaridade=("escolaridade", "mean"), volatilidade=("volatilidade", "mean"),
                ln_va_por_empregado=("ln_va_por_empregado", "mean"),
                z_escolaridade=("z_escolaridade", "mean"), z_baixa_vol=("z_baixa_vol", "mean"),
                z_capital_humano=("z_capital_humano", "mean"),
                n_anos_esc=("escolaridade", "count"), n_anos_vol=("volatilidade", "count"),
                n_anos_ch=("ln_va_por_empregado", "count")).reset_index()
    med["cobertura_meses_pct"] = med["ticker"].map(cobertura).fillna(0) * 100
    med["z_combinado"] = med[["z_escolaridade", "z_baixa_vol", "z_capital_humano"]].mean(axis=1)
    med["n_anos_comb"] = med[["n_anos_esc", "n_anos_vol", "n_anos_ch"]].min(axis=1)
    criterios = [  # nome, coluna de ordenação, maior é melhor?, coluna de nº de anos, valor exibido
        ("Escolaridade do CEO – top 10", "escolaridade", False, "n_anos_esc", "escolaridade"),
        ("Baixa volatilidade – top 10", "volatilidade", True, "n_anos_vol", "volatilidade"),
        ("Capital humano – top 10", "ln_va_por_empregado", False, "n_anos_ch", "ln_va_por_empregado"),
        ("Combinado 3 fatores – top 10", "z_combinado", False, "n_anos_comb", "z_combinado"),
    ]
    listas, series = [], {}
    for nome, c, asc, c_n, c_show in criterios:
        v = med[(med[c_n] >= min_anos) & (med["cobertura_meses_pct"] >= min_meses_pct * 100) & med[c].notna()]
        # desempate: mais anos com o atributo medido, depois maior cobertura de cotações
        v = v.sort_values([c, c_n, "cobertura_meses_pct"], ascending=[asc, False, False])
        if len(v) > n:
            corte = v[c].iloc[n - 1]
            empatadas = int((v[c] == corte).sum())
            if empatadas > 1 and (v[c].iloc[n:] == corte).any():
                log(f"  {nome}: {empatadas} empresas empatadas com {c} = {corte:.3f} na linha de corte; "
                    f"desempate por anos com dado e cobertura de cotações")
        v = v.head(n)
        listas.append(pd.DataFrame({"carteira": nome, "posicao": range(1, len(v) + 1),
                                    "empresa": v["cnpj"].map(emp_nome).values, "ticker": v["ticker"].values,
                                    "media_do_fator_no_periodo": v[c_show].round(4).values,
                                    "anos_com_dado": v[c_n].values,
                                    "cobertura_meses_pct": v["cobertura_meses_pct"].round(1).values,
                                    "cnpj": v["cnpj"].values}))
        if len(v):
            series[nome] = jan[v["ticker"]].mean(axis=1)
    rm = pd.DataFrame(series)
    rm["Ibovespa"] = ibov_m.reindex(rm.index)
    resumo = []
    for nome in series:
        r, ib = rm[nome].dropna(), rm["Ibovespa"]
        ib = ib.loc[r.index]
        exc = r - ib
        k = len(r)
        mt = sm.OLS(exc, np.ones(k)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        capm = sm.OLS(r, sm.add_constant(ib)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        linha = {"carteira": nome, "n_empresas": int(len(listas[[x[0] for x in criterios].index(nome)])), "meses": k,
                 "retorno_acumulado": (1 + r).prod() - 1, "retorno_acumulado_ibov": (1 + ib).prod() - 1,
                 "retorno_aa_carteira": (1 + r).prod() ** (12 / k) - 1,
                 "retorno_aa_ibovespa": (1 + ib).prod() ** (12 / k) - 1,
                 "excesso_medio_mensal": exc.mean(), "t_excesso (Newey-West)": mt.tvalues.iloc[0],
                 "p_excesso": mt.pvalues.iloc[0], "alfa_mensal_vs_ibov": capm.params.iloc[0],
                 "p_alfa": capm.pvalues.iloc[0], "beta_vs_ibov": capm.params.iloc[1],
                 "vol_aa": r.std() * np.sqrt(12), "pct_meses_acima_ibov": (exc > 0).mean() * 100,
                 "max_drawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()}
        if cdi_m is not None:
            ex_rf = r - cdi_m.reindex(r.index).fillna(0)
            linha["sharpe"] = ex_rf.mean() / ex_rf.std() * np.sqrt(12)
        resumo.append(linha)
    log(f"Top {n} pela média do período: válidas = ≥{min_anos} anos com dado e cotação em ≥{min_meses_pct:.0%} "
        f"dos meses de {ini:%m/%Y} a {fim:%m/%Y}")
    return pd.concat(listas, ignore_index=True), pd.DataFrame(resumo), rm


def cdi_mensal():
    try:
        import requests
        j = requests.get(URL_CDI, timeout=60).json()
        s = pd.Series({pd.to_datetime(x["data"], dayfirst=True) + pd.offsets.MonthEnd(0): float(x["valor"]) / 100
                       for x in j})
        return s.sort_index()
    except Exception as e:  # noqa: BLE001
        log(f"CDI não obtido ({e}); Sharpe não calculado")
        return None


# --------------------------------------------------------------------------
def interpretar(coefs, cart):
    txt = []
    if not coefs.empty:
        m = coefs[coefs["modelo"] == "(5) Três fatores + controles"]
        if m.empty:
            m = coefs[coefs["modelo"] == "(4) Três fatores"]
        for _, r in m[m["variavel"].isin(FATORES.values())].iterrows():
            sig = "SIGNIFICATIVO a 5%" if r["p_valor"] < 0.05 else ("significativo só a 10%" if r["p_valor"] < 0.10
                                                                   else "NÃO significativo")
            txt.append(f"{r['variavel']}: {r['coeficiente'] * 100:+.2f} p.p. de retorno anual acima do Ibovespa "
                       f"por desvio-padrão (p = {r['p_valor']:.3f}) → {sig}; sinal esperado {r['sinal_esperado']}")
    for _, r in cart.iterrows():
        if r["carteira"] == "Ibovespa":
            continue
        txt.append(f"{r['carteira']}: {r['retorno_aa_carteira'] * 100:.1f}% a.a. vs Ibovespa "
                   f"{r['retorno_aa_ibovespa'] * 100:.1f}% a.a.; alfa mensal {r['alfa_mensal_vs_ibov'] * 100:+.2f}% "
                   f"(p = {r['p_alfa']:.3f})")
    return txt


LEIA_ME = """Teste dos fatores do grupo sobre o desempenho das ações listadas – {versao}
Variável dependente: retorno da ação de 30/jun/t a 30/jun/t+1 menos o retorno do Ibovespa no mesmo período
(winsorizado a 1%/99%). Anos de formação: {anos}.
Sinais medidos em 30/jun/t: escolaridade do CEO vigente em 31/mai/t (base_ceo_cvm.py); volatilidade dos retornos
diários de jul/t-1 a jun/t; capital humano = ln(valor adicionado/empregados) com a DVA do exercício t-1 (DFP) e
empregados do FRE de t; pessoal/VA = parcela do valor adicionado distribuída a pessoal (DVA).
Fatores padronizados por ano (z-score): o coeficiente é o efeito, em retorno anual acima do Ibovespa, de 1 desvio-padrão.
Abas:
- regressao_painel: MQO empresa-ano com efeitos fixos de ano e setor; erros-padrão clusterizados por empresa.
  Modelos (1)–(3b): cada fator sozinho; (4): os três juntos; (5): + controles de liquidez (ln volume financeiro) e
  tamanho (ln valor adicionado).
- fama_macbeth: média dos coeficientes das regressões anuais (robustez; poucos anos = pouco poder estatístico).
- carteiras: top 20% de cada fator (baixa volatilidade = 20% MENOS voláteis) e do score combinado, pesos iguais,
  rebalanceamento anual em 30/jun. t e p-valores com erros de Newey-West (6 defasagens). spread = top 20% − bottom 20%.
- top10_carteiras / top10_empresas / top10_retornos_mensais: para cada fator, as 10 empresas válidas com a melhor
  MÉDIA do fator no período (escolaridade mais alta, volatilidade mais baixa, VA/empregado mais alto, score combinado
  mais alto). Válida = fator medido em ≥ 5 anos (com liquidez mínima) e cotação em ≥ 60% dos meses. Carteira de pesos
  iguais mantida no período inteiro. Como a escolha usa a média de anos futuros, o resultado é DESCRITIVO (mostra
  como foram as empresas que mais tiveram o atributo), não uma estratégia que poderia ter sido seguida em tempo real.
- retornos_mensais, membros_carteiras, base_empresa_ano, correlacao_fatores: dados usados nos testes.
Como ler: p_valor < 0,05 = efeito estatisticamente significativo a 5%. Sinal esperado: escolaridade (+),
volatilidade (−, ou seja, menos volatilidade → mais retorno), capital humano (+).
Limitações: amostra de ~9 anos; preços do Yahoo Finance podem faltar para empresas que saíram da bolsa
(viés de sobrevivência); de-para CNPJ→ticker pelo FCA escolhe o ticker mais líquido; nº de empregados lido das
tabelas de RH do FRE (conferir a linha de verificação da Petrobras no log); correlação não implica causalidade."""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--base", type=Path, default=Path("saida_ceo"), help="pasta de saída do base_ceo_cvm.py")
    ap.add_argument("--saida", type=Path, default=None, help="pasta dos resultados (padrão: a mesma da base)")
    ap.add_argument("--inicio", type=int, default=2017, help="1º ano de formação das carteiras")
    ap.add_argument("--fim", type=int, default=2025, help="último ano de formação")
    ap.add_argument("--liq-min", type=float, default=500_000, help="volume financeiro médio diário mínimo (R$)")
    ap.add_argument("--depara", type=Path, default=None, help="CSV próprio cnpj;ticker (opcional)")
    ap.add_argument("--top-n", type=int, default=10, help="nº de empresas na análise pela média do período")
    ap.add_argument("--min-anos", type=int, default=5, help="mín. de anos com dado para a empresa ser válida no top N")
    if argv is None and "ipykernel" in sys.modules:
        argv = [a for a in sys.argv[1:] if not a.startswith("--f=") and not a.endswith(".json")]
    a, _ = ap.parse_known_args(argv)
    saida = a.saida or a.base
    anos = list(range(a.inicio, a.fim + 1))
    LOG.clear()
    log(f"Versão do script: {VERSAO}")

    painel, form, emp = carregar_base(a.base)
    listadas = set(emp.loc[emp["listada_bolsa"], "cnpj"])
    setor = emp.set_index("cnpj").get("setor_atividade", pd.Series(dtype=str))

    esc = sinal_escolaridade(painel, form, anos)
    dp = (pd.read_csv(a.depara, sep=";", dtype=str).assign(cnpj=lambda d: cnpj14(d["cnpj"]))
          if a.depara else depara_fca(range(a.inicio - 1, a.fim + 2)))
    dp = dp[dp["cnpj"].isin(listadas)]
    px, vol = baixar_precos(dp["ticker"].unique(), inicio=f"{a.inicio - 1}-06-01")
    if "^BVSP" not in px:
        sys.exit("Série do Ibovespa (^BVSP) não baixada do Yahoo.")
    tk = escolher_ticker(dp, px, vol)
    sp = sinais_preco(px, vol, anos)
    ret = retorno_anual(px, anos)
    ibov_ano = ret[ret["ticker"] == "^BVSP"].set_index("ano")["ret"]
    log("Capital humano: DVA (DFP) e empregados (FRE)")
    dva = dva_anual(range(a.inicio - 1, a.fim))
    empreg = empregados_fre(anos)
    ch = sinal_capital_humano(dva, empreg, anos)

    # base empresa-ano
    base = tk.merge(pd.DataFrame({"ano": anos}), how="cross")
    base = (base.merge(esc, on=["cnpj", "ano"], how="left")
                .merge(sp, on=["ticker", "ano"], how="left")
                .merge(ch[["cnpj", "ano", "va", "pessoal", "empregados", "va_por_empregado", "ln_va_por_empregado",
                           "pessoal_sobre_va", "ln_va"]], on=["cnpj", "ano"], how="left")
                .merge(ret, on=["ticker", "ano"], how="left"))
    base["ret_ibov"] = base["ano"].map(ibov_ano)
    base["setor"] = base["cnpj"].map(setor).fillna("Sem setor")
    base = base[base["liq_media_rs"] >= a.liq_min].dropna(subset=["ret"]).copy()
    base["exc_ret"] = winsor(base["ret"] - base["ret_ibov"])
    base["ln_liq"] = np.log(base["liq_media_rs"])
    base["z_escolaridade"] = zscore_ano(base, "escolaridade")
    base["z_volatilidade"] = zscore_ano(base, "volatilidade")
    base["z_baixa_vol"] = -base["z_volatilidade"]
    base["z_capital_humano"] = zscore_ano(base, "ln_va_por_empregado")
    base["z_pessoal_va"] = zscore_ano(base, "pessoal_sobre_va")
    base.insert(1, "empresa", base["cnpj"].map(emp.set_index("cnpj")["empresa"]))
    log(f"Base empresa-ano: {len(base):,} observações, {base['cnpj'].nunique():,} empresas "
        f"(liquidez mínima R$ {a.liq_min:,.0f}/dia); com escolaridade: {base['escolaridade'].notna().sum():,}; "
        f"com capital humano (VA/empregado): {base['ln_va_por_empregado'].notna().sum():,}")

    log("A) Regressões em painel")
    coefs, resumo = regressoes_painel(base)
    log("B) Fama-MacBeth")
    fm = fama_macbeth(base)
    fm_res, fm_anual = fm if isinstance(fm, tuple) else (pd.DataFrame(), pd.DataFrame())
    log("C) Carteiras")
    m = px.resample("ME").last()
    ret_m = m.pct_change(fill_method=None)
    cdi = cdi_mensal()
    cart, rm, membros = carteiras(base, ret_m, ret_m["^BVSP"], cdi)
    log("D) Top 10 de cada fator pela média do período")
    nomes_emp = emp.set_index("cnpj")["empresa"]
    t10_lista, t10_cart, t10_rm = top_n_media_periodo(base, ret_m, ret_m["^BVSP"], cdi, nomes_emp,
                                                      n=a.top_n, min_anos=a.min_anos)
    corr = base[["escolaridade", "volatilidade", "ln_va_por_empregado", "pessoal_sobre_va"]].corr(method="spearman")

    interp = interpretar(coefs, cart)
    for _, r in t10_cart.iterrows():
        interp.append(f"{r['carteira']} (média do período): {r['retorno_aa_carteira'] * 100:.1f}% a.a. vs Ibovespa "
                      f"{r['retorno_aa_ibovespa'] * 100:.1f}% a.a.; alfa mensal {r['alfa_mensal_vs_ibov'] * 100:+.2f}% "
                      f"(p = {r['p_alfa']:.3f}) – descritivo, escolhe as empresas olhando o período inteiro")
    log("\nRESUMO")
    for t in interp:
        log("  " + t)

    saida.mkdir(parents=True, exist_ok=True)
    xlsx = saida / "resultados_fatores.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as w:
        pd.DataFrame({"Leia-me": LEIA_ME.format(versao=VERSAO, anos=f"{anos[0]}–{anos[-1]}").split("\n")}).to_excel(
            w, sheet_name="LEIA-ME", index=False)
        pd.DataFrame({"interpretacao": interp}).to_excel(w, sheet_name="resumo", index=False)
        coefs.to_excel(w, sheet_name="regressao_painel", index=False)
        resumo.to_excel(w, sheet_name="regressao_painel", index=False, startrow=len(coefs) + 3)
        fm_res.to_excel(w, sheet_name="fama_macbeth", index=False)
        fm_anual.to_excel(w, sheet_name="fama_macbeth", index=False, startrow=len(fm_res) + 3)
        cart.to_excel(w, sheet_name="carteiras", index=False)
        rm.reset_index(names="mes").to_excel(w, sheet_name="retornos_mensais", index=False)
        t10_cart.to_excel(w, sheet_name="top10_carteiras", index=False)
        t10_lista.to_excel(w, sheet_name="top10_empresas", index=False)
        t10_rm.reset_index(names="mes").to_excel(w, sheet_name="top10_retornos_mensais", index=False)
        membros.to_excel(w, sheet_name="membros_carteiras", index=False)
        corr.reset_index(names="spearman").to_excel(w, sheet_name="correlacao_fatores", index=False)
        base.to_excel(w, sheet_name="base_empresa_ano", index=False)
        pd.DataFrame({"log": LOG}).to_excel(w, sheet_name="log", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws.columns:
                ws.column_dimensions[c[0].column_letter].width = min(max(10, max(len(str(x.value or "")) for x in list(c)[:100]) + 2), 70)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for dados, titulo, arq in [
                (rm, "Carteiras top 20% (rebalanceamento anual) vs Ibovespa – R$ 1 investido", "retorno_acumulado_fatores.png"),
                (t10_rm, "Top 10 de cada fator pela média do período vs Ibovespa – R$ 1 investido",
                 "retorno_acumulado_top10.png")]:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for c in dados.columns:
                s = (1 + dados[c].fillna(0)).cumprod()
                ax.plot(s.index, s.values, lw=2.4 if c == "Ibovespa" else 1.6,
                        color="black" if c == "Ibovespa" else None, ls="--" if c == "Ibovespa" else "-", label=c)
            ax.set_title(titulo)
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(saida / arq, dpi=150)
            plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"Gráfico não gerado: {e}")
    print(f"\nResultados em {xlsx.resolve()} e {saida / 'retorno_acumulado_fatores.png'}")


if __name__ == "__main__":
    main()