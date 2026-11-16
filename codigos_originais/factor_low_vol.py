import yfinance as yf
import pandas as pd
import numpy as np

def run_low_vol_factor(tickers, inicio_in_sample, fim_in_sample, fim_out_of_sample, top_n=10, risk_free_rate=0.105):
    """
    Executa o backtest do Fator de Baixa Volatilidade com separação estrita de janelas temporais.
    """
    print(f"1. Baixando dados de {inicio_in_sample} até {fim_out_of_sample}...")
    
    # Extração de Dados (auto_adjust=True garante que dividendos não distorçam a volatilidade)
    dados = yf.download(tickers, start=inicio_in_sample, end=fim_out_of_sample, auto_adjust=True)['Close']
    
    # Limpeza: remove ações que não têm o histórico completo nesses 4 anos
    dados = dados.dropna(axis=1)
    retornos = dados.pct_change().dropna()
    
    # Divisão das Janelas (In-Sample vs Out-of-Sample)
    # .loc fatia o DataFrame exatamente nas datas estipuladas
    retornos_in_sample = retornos.loc[:fim_in_sample]
    retornos_out_of_sample = retornos.loc[fim_in_sample:]
    
    # ==========================================
    # ETAPA DE FORMAÇÃO (IN-SAMPLE: 2022 - 2023)
    # ==========================================
    print("2. Calculando volatilidade e ranqueando as empresas (In-Sample)...")
    
    # Calcula a volatilidade anualizada baseada em 2 anos de pregões diários
    vol_anualizada_in_sample = retornos_in_sample.std() * np.sqrt(252)
    
    # Cria o Ranking: Ordena do MENOR risco para o MAIOR risco
    ranking_low_vol = vol_anualizada_in_sample.sort_values(ascending=True)
    
    # Seleciona as N ações menos voláteis
    carteira_selecionada = ranking_low_vol.head(top_n).index.tolist()
    
    # ==========================================
    # ETAPA DE TESTE (OUT-OF-SAMPLE: 2024 - 2026)
    # ==========================================
    print("3. Testando o portfólio no mundo real (Out-of-Sample)...")
    
    # Isola os retornos apenas das 10 ações selecionadas no período de teste
    retornos_carteira_oos = retornos_out_of_sample[git add ap_2.pycarteira_selecionada]
    
    # Calcula o retorno diário do portfólio (Pesos iguais = 10% para cada ação)
    retorno_diario_portfolio = retornos_carteira_oos.mean(axis=1)
    
    # Métricas Finais do Portfólio (2024 e 2025)
    retorno_acumulado = (1 + retorno_diario_portfolio).cumprod().iloc[-1] - 1
    
    # Ajuste dinâmico de anos (caso o pregão tenha menos de 252 dias úteis no período)
    anos_oos = len(retorno_diario_portfolio) / 252
    cagr = (1 + retorno_acumulado) ** (1 / anos_oos) - 1
    
    # Volatilidade e Sharpe do período de teste
    vol_portfolio = retorno_diario_portfolio.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / vol_portfolio
    
    metricas = {
        'Retorno Acumulado (%)': round(retorno_acumulado * 100, 2),
        'Retorno Anualizado - CAGR (%)': round(cagr * 100, 2),
        'Volatilidade Anualizada (%)': round(vol_portfolio * 100, 2),
        'Índice Sharpe': round(sharpe, 2)
    }
    
    return carteira_selecionada, metricas

# ==========================================
# EXECUÇÃO DO CÓDIGO
# ==========================================
if __name__ == "__main__":
    # Lista completa das ações do Ibovespa
    tickers_ibovespa = [
        'RRRP3.SA', 'ALOS3.SA', 'ALPA4.SA', 'ABEV3.SA', 'ARZZ3.SA', 'ASAI3.SA', 'AZUL4.SA', 
        'B3SA3.SA', 'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA', 
        'BRFS3.SA', 'BPAC11.SA', 'CRFB3.SA', 'CCRO3.SA', 'CMIG4.SA', 'COGN3.SA', 'CPLE6.SA', 
        'CSAN3.SA', 'CPFE3.SA', 'CMIN3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DXCO3.SA', 'ELET3.SA', 
        'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA', 'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'EZTC3.SA', 
        'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA', 'NTCO3.SA', 'SOMA3.SA', 'HAPV3.SA', 'HYPE3.SA', 
        'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'JBSS3.SA', 'KLBN11.SA', 'RENT3.SA', 
        'LREN3.SA', 'LWSA3.SA', 'MGLU3.SA', 'MRFG3.SA', 'BEEF3.SA', 'MRVE3.SA', 'MULT3.SA', 
        'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'PRIO3.SA', 'PETZ3.SA', 'RADL3.SA', 
        'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA', 'SMTO3.SA', 'CSNA3.SA', 
        'SLCE3.SA', 'SUZB3.SA', 'TAEE11.SA', 'VIVT3.SA', 'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 
        'USIM5.SA', 'VALE3.SA', 'VAMO3.SA', 'VBBR3.SA', 'WEGE3.SA', 'YDUQ3.SA'
    ]
    
    acoes_vencedoras, resultados = run_low_vol_factor(
        tickers=tickers_ibovespa,
        inicio_in_sample='2022-01-01',   # Início da coleta de dados e formação
        fim_in_sample='2023-12-31',      # Fim da formação / Início do Teste Real
        fim_out_of_sample='2026-01-01',  # Fim do Teste Real
        top_n=10
    )
    
    print("\n=======================================================")
    print(" RESULTADO DO FATOR: BAIXA VOLATILIDADE (LOW VOLATILITY)")
    print("=======================================================")
    print("\n-> AS 10 AÇÕES SELECIONADAS (Com base em 2022-2023):")
    # Limpa o '.SA' apenas para exibir no console de forma mais profissional
    print([acao.replace('.SA', '') for acao in acoes_vencedoras])
    
    print("\n-> DESEMPENHO DA CARTEIRA (No período de 2024-2025):")
    for metrica, valor in resultados.items():
        print(f"{metrica}: {valor}")


