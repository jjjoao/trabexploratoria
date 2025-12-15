import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind, norm
import numpy as np

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise Spotify (1991-2020)", 
    layout="wide",
    page_icon="🎵"
)

# --- 1. Carregamento e Processamento de Dados ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spotify_songs.csv")
    except FileNotFoundError:
        st.error("Arquivo 'spotify_songs.csv' não encontrado. Por favor, coloque-o na mesma pasta do script.")
        return None

    # Processamento de Data
    df['year'] = pd.to_numeric(df['track_album_release_date'].astype(str).str[:4], errors='coerce')
    
    # Criar Períodos
    def get_period(year):
        if 1991 <= year <= 2000:
            return "1991 - 2000"
        elif 2001 <= year <= 2010:
            return "2001 - 2010"
        elif 2011 <= year <= 2020:
            return "2011 - 2020"
        else:
            return "Outros"

    df['periodo'] = df['year'].apply(get_period)
    
    # Filtrar apenas o intervalo desejado
    df_filtered = df[df['periodo'] != "Outros"].copy()
    
    # Converter Mode para Categórico (Legível)
    df_filtered['mode_categoria'] = df_filtered['mode'].map({0: 'Menor', 1: 'Maior'})
    
    return df_filtered

# Função auxiliar para teste de proporção (Z-test)
def z_test_proportions(count1, nobs1, count2, nobs2):
    # Proporções
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    # Proporção combinada
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    # Erro padrão
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs1 + 1/nobs2))
    # Estatística Z
    z = (p1 - p2) / se
    # Valor-p (bilateral)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value, p1, p2

df = load_data()

# --- NAVEGAÇÃO LATERAL ---
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["🏠 Apresentação", "📊 Dashboard de Análise"])
st.sidebar.markdown("---")
st.sidebar.info("Dados extraídos via Spotifyr Package / TidyTuesday.")

if df is not None:
    # Dataset de músicas únicas
    df_unique = df.drop_duplicates(subset=['track_id'])

    # --- PÁGINA 1: APRESENTAÇÃO ---
    if pagina == "🏠 Apresentação":
        st.title("🎵 Evolução Musical no Spotify (1991 - 2020)")
        st.markdown("""
        ### 🎯 Objetivo da Análise
        Este projeto traça o perfil das músicas mais escutadas nas últimas três décadas.
        
        **Nesta aplicação você encontrará:**
        1.  **Dashboard:** Visualizações gráficas de tendências.
        2.  **Ferramenta de Teste:** Uma calculadora estatística para validar se as mudanças (médias ou gêneros) são reais ou fruto do acaso.
        """)
        
        st.info("Utilize o menu lateral para navegar até o Dashboard.")

    # --- PÁGINA 2: DASHBOARD ---
    elif pagina == "📊 Dashboard de Análise":
        st.title("📊 Dashboard Analítico")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📉 Estatísticas Gerais", 
            "🎸 Gêneros", 
            "🎛️ Características de Áudio", 
            "⭐ Popularidade",
            "🧪 Ferramenta de Testes"
        ])

        # ... (Abas 1 a 4 permanecem iguais ao código anterior) ...
        # Para economizar espaço na resposta, vou focar na ABA 5 que mudou.
        # AS ABAS 1, 2, 3 e 4 DO CÓDIGO ANTERIOR DEVEM SER MANTIDAS AQUI.
        # SE VOCÊ PRECISAR DO CÓDIGO COMPLETO DAS OUTRAS ABAS NOVAMENTE, ME AVISE.
        # VOU REPETIR APENAS A ESTRUTURA BÁSICA DELAS ABAIXO PARA O CÓDIGO RODAR:

        with tab1:
            st.header("Resumo por Década")
            resumo = df_unique.groupby('periodo').agg({'duration_ms': lambda x: x.mean()/60000, 'energy': 'mean', 'valence': 'mean'}).reset_index()
            st.dataframe(resumo, use_container_width=True)
            fig = px.bar(resumo, x='periodo', y='duration_ms', title="Duração Média")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Gêneros")
            cnt = df.groupby(['periodo', 'playlist_genre']).size().reset_index(name='n')
            fig = px.bar(cnt, x='periodo', y='n', color='playlist_genre', barmode='fill', title="Distribuição de Gêneros")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.header("Áudio")
            stats = df_unique.groupby('year')[['energy', 'valence']].mean().reset_index()
            fig = px.line(stats, x='year', y=['energy', 'valence'], title="Evolução Temporal")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.header("Popularidade")
            pop = df_unique.groupby('year')['track_popularity'].mean().reset_index()
            fig = px.line(pop, x='year', y='track_popularity', title="Popularidade")
            st.plotly_chart(fig, use_container_width=True)

        # --- ABA 5: FERRAMENTA DE TESTES (NOVA IMPLEMENTAÇÃO) ---
        with tab5:
            st.header("🧪 Ferramenta de Testes Estatísticos")
            st.markdown("""
            Esta ferramenta permite comparar duas décadas para verificar se houve mudanças significativas.
            * **Variáveis Contínuas:** Usa Teste t de Welch (Comparação de Médias).
            * **Variáveis Categóricas:** Usa Teste Z de Proporções (Comparação de Frequência).
            * **Significância (α):** Fixada em 5% (0.05).
            """)
            st.divider()

            # 1. Escolha do Tipo de Teste
            tipo_teste = st.radio("O que você quer comparar?", 
                                ["Variável Numérica (ex: Energia, Duração)", 
                                 "Variável Categórica (ex: Gênero, Tonalidade)"], horizontal=True)

            col_a, col_b = st.columns(2)
            decadas = sorted(df_unique['periodo'].unique())
            
            with col_a:
                decada_1 = st.selectbox("Década A (Grupo 1)", decadas, index=0)
            with col_b:
                decada_2 = st.selectbox("Década B (Grupo 2)", decadas, index=1)

            # --- LÓGICA DO TESTE ---
            if decada_1 == decada_2:
                st.error("⚠️ Escolha duas décadas diferentes para comparar.")
            
            else:
                # Filtrar dados das décadas
                df_d1 = df_unique[df_unique['periodo'] == decada_1]
                df_d2 = df_unique[df_unique['periodo'] == decada_2]

                # CASO 1: NUMÉRICO (Médias)
                if "Numérica" in tipo_teste:
                    vars_num = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness', 'track_popularity', 'duration_ms', 'loudness', 'tempo']
                    variavel = st.selectbox("Escolha a Variável", vars_num)

                    if st.button("🚀 Calcular Teste t"):
                        # Dados
                        dados1 = df_d1[variavel].dropna()
                        dados2 = df_d2[variavel].dropna()
                        
                        # Médias
                        m1, m2 = dados1.mean(), dados2.mean()
                        
                        # Teste t de Welch
                        stat, p_val = ttest_ind(dados1, dados2, equal_var=False)
                        
                        # Resultados
                        col1, col2 = st.columns(2)
                        col1.metric(f"Média {decada_1}", f"{m1:.4f}")
                        col2.metric(f"Média {decada_2}", f"{m2:.4f}", delta=f"{m2-m1:.4f}")
                        
                        st.markdown("### Interpretação")
                        st.write(f"**Valor-p:** `{p_val:.10f}`")
                        
                        if p_val < 0.05:
                            st.success("✅ **Diferença Significativa!**")
                            maior = decada_1 if m1 > m2 else decada_2
                            st.write(f"O teste estatístico indica (com 95% de confiança) que a média de **{variavel}** mudou. A década de **{maior}** apresenta valores maiores.")
                        else:
                            st.warning("❌ **Sem Diferença Significativa.**")
                            st.write(f"Não há evidências estatísticas suficientes para afirmar que a média de **{variavel}** mudou entre essas décadas. A diferença observada pode ser acaso.")

                # CASO 2: CATEGÓRICO (Proporções)
                else:
                    vars_cat = ['playlist_genre', 'playlist_subgenre', 'mode_categoria', 'key']
                    variavel_cat = st.selectbox("Escolha a Categoria", vars_cat)
                    
                    # O usuário precisa escolher qual valor específico ele quer testar (ex: "Rock" dentro de "Genre")
                    # Usamos o df completo (não unique) para gêneros pois a mesma música pode estar em playlists de gêneros diferentes? 
                    # O usuário pediu unique antes, vamos manter df_unique para consistência estatística de amostras independentes.
                    valores_possiveis = sorted(df_unique[variavel_cat].unique().astype(str))
                    alvo = st.selectbox(f"Qual valor de '{variavel_cat}' você quer testar?", valores_possiveis)

                    if st.button("🚀 Calcular Teste de Proporção"):
                        # Contagens
                        total1 = len(df_d1)
                        count1 = len(df_d1[df_d1[variavel_cat].astype(str) == alvo])
                        
                        total2 = len(df_d2)
                        count2 = len(df_d2[df_d2[variavel_cat].astype(str) == alvo])
                        
                        # Teste Z
                        z_stat, p_val, prop1, prop2 = z_test_proportions(count1, total1, count2, total2)
                        
                        # Resultados Visual
                        col1, col2 = st.columns(2)
                        col1.metric(f"% em {decada_1}", f"{prop1:.2%}", help=f"{count1} músicas de {total1}")
                        col2.metric(f"% em {decada_2}", f"{prop2:.2%}", delta=f"{(prop2-prop1)*100:.2f} p.p.")
                        
                        st.markdown("### Interpretação")
                        st.write(f"**Valor-p:** `{p_val:.10f}`")
                        
                        if p_val < 0.05:
                            st.success("✅ **Mudança Significativa na Proporção!**")
                            tendencia = "aumentou" if prop2 > prop1 else "diminuiu"
                            st.write(f"Com 95% de confiança, podemos afirmar que a presença de **'{alvo}'** {tendencia} significativamente de {decada_1} para {decada_2}.")
                        else:
                            st.warning("❌ **Proporção Estável.**")
                            st.write(f"A variação na porcentagem de **'{alvo}'** entre as décadas não é estatisticamente relevante.")


