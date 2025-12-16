import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, norm
import numpy as np

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise Spotify (1991-2020)", 
    layout="wide",
    page_icon="🎵"
)

# --- FUNÇÕES AUXILIARES ---

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spotify_songs.csv")
    except FileNotFoundError:
        st.error("Arquivo 'spotify_songs.csv' não encontrado.")
        return None

    # Processamento
    df['year'] = pd.to_numeric(df['track_album_release_date'].astype(str).str[:4], errors='coerce')
    
    def get_period(year):
        if 1991 <= year <= 2000: return "1991 - 2000"
        elif 2001 <= year <= 2010: return "2001 - 2010"
        elif 2011 <= year <= 2020: return "2011 - 2020"
        else: return "Outros"

    df['periodo'] = df['year'].apply(get_period)
    df_filtered = df[df['periodo'] != "Outros"].copy()
    df_filtered['mode_categoria'] = df_filtered['mode'].map({0: 'Menor', 1: 'Maior'})
    
    return df_filtered

def z_test_proportions(count1, nobs1, count2, nobs2):
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs1 + 1/nobs2))
    if se == 0: return 0, 1.0, p1, p2
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value, p1, p2

df = load_data()

# --- NAVEGAÇÃO LATERAL ---
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["🏠 Apresentação", "📊 Dashboard de Análise"])
st.sidebar.markdown("---")
st.sidebar.info("Dados extraídos via Spotifyr Package / TidyTuesday.")

if df is not None:
    df_unique = df.drop_duplicates(subset=['track_id'])

    # === PÁGINA 1: APRESENTAÇÃO ===
    if pagina == "🏠 Apresentação":
        st.title("🎵 Evolução Musical no Spotify (1991 - 2020)")
        
        st.markdown("""
        ### 🎯 Objetivo da Análise
        Este projeto tem como objetivo traçar um perfil das músicas mais escutadas no Spotify ao longo das últimas três décadas.
        A análise busca responder: **O que mudou na música popular?**
        
        Investigamos mudanças em:
        * ⏱️ **Duração:** As músicas estão ficando mais curtas?
        * 🎸 **Gêneros:** Qual estilo dominou cada época?
        * 🎛️ **Características Técnicas:** A música ficou mais rápida, mais dançante ou mais triste?
        
        ---
        """)

        st.warning("### 📖 Dicionário de Variáveis do Banco de Dados")
        st.markdown("""
        Abaixo estão as descrições de todas as variáveis utilizadas nesta análise:

        * **Danceability (Dançabilidade):** Descreve o quão adequada uma música é para dançar (0.0 a 1.0).
        * **Energy (Energia):** Medida de intensidade e atividade. Músicas rápidas e barulhentas têm alta energia.
        * **Valence (Positividade):** Descreve a positividade musical. Alto = Feliz/Eufórico, Baixo = Triste/Depressivo.
        * **Acousticness (Acústico):** Nível de confiança de que a faixa é acústica.
        * **Instrumentalness (Instrumental):** Probabilidade da música não conter vocais.
        * **Speechiness (Fala):** Detecta a presença de palavras faladas.
        * **Loudness (Volume):** O volume médio da faixa em decibéis (dB).
        * **Tempo (BPM):** Velocidade da música em batidas por minuto.
        * **Popularity (Popularidade):** Índice de 0 a 100 baseado na reprodução atual.
        * **Duration_ms (Duração):** Duração da música em milissegundos.
        * **Mode (Modo/Tonalidade):** Indica a escala da música (Maior = geralmente alegre, Menor = geralmente sério).
        * **Playlist Genre:** O gênero principal da playlist onde a música foi encontrada.
        """)

    # === PÁGINA 2: DASHBOARD ===
    elif pagina == "📊 Dashboard de Análise":
        st.title("📊 Dashboard Analítico")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⏱️ Duração das Músicas", 
            "🎸 Gêneros", 
            "🎛️ Características de Áudio", 
            "⭐ Popularidade",
            "🧪 Teste de Hipótese"
        ])

        # --- ABA 1: DURAÇÃO ---
        with tab1:
            st.header("Análise de Duração")
            
            resumo = df_unique.groupby('periodo').agg({
                'duration_ms': lambda x: (x.mean() / 60000),
                'energy': 'mean', 'valence': 'mean', 'danceability': 'mean', 'track_id': 'count'
            }).reset_index()
            resumo.columns = ['Período', 'Duração (min)', 'Energia', 'Positividade', 'Dançabilidade', 'Nº Músicas']
            
            st.dataframe(resumo.style.format({'Duração (min)': '{:.2f}', 'Energia': '{:.3f}', 'Positividade': '{:.3f}', 'Dançabilidade': '{:.3f}'}), use_container_width=True)

            st.subheader("A Queda na Duração das Músicas")
            fig_duracao = px.bar(resumo, x='Período', y='Duração (min)', color='Período', text_auto='.2f', title="Duração Média (Minutos) por Década")
            fig_duracao.update_traces(textposition='outside')
            st.plotly_chart(fig_duracao, use_container_width=True)

        # --- ABA 2: GÊNEROS ---
        with tab2:
            st.header("Dominância de Gêneros")
            genre_counts = df.groupby(['periodo', 'playlist_genre']).size().reset_index(name='n')
            genre_counts['total'] = genre_counts.groupby('periodo')['n'].transform('sum')
            genre_counts['proporcao'] = genre_counts['n'] / genre_counts['total']
            
            fig_genre = px.bar(genre_counts, x="periodo", y="proporcao", color="playlist_genre", title="Distribuição de Gêneros (% nas Playlists)", barmode="group")
            fig_genre.layout.yaxis.tickformat = ',.0%'
            st.plotly_chart(fig_genre, use_container_width=True)

        # --- ABA 3: ÁUDIO (ATUALIZADA) ---
        with tab3:
            st.header("Tendências de Áudio")
            
            # Layout em duas colunas: Radar à esquerda, Linhas à direita
            col_radar, col_line = st.columns([1, 1])
            
            # --- COLUNA 1: PERFIL SONORO (RADAR CHART) ---
            with col_radar:
                st.subheader("📸 Perfil Sonoro (Radar)")
                st.markdown("Compare a 'forma' das décadas nas variáveis de 0 a 1.")
                
                features_radar = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness']
                radar_df = df_unique.groupby('periodo')[features_radar].mean().reset_index()
                
                fig_radar = go.Figure()
                
                # Cores fixas para cada década para manter consistência
                colors = ['#636EFA', '#EF553B', '#00CC96'] 
                
                for i, row in radar_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row[features_radar].values,
                        theta=features_radar,
                        fill='toself',
                        name=row['periodo'],
                        line_color=colors[i % len(colors)]
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=450
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # --- COLUNA 2: EVOLUÇÃO TEMPORAL INTERATIVA ---
            with col_line:
                st.subheader("📈 Evolução Temporal Interativa")
                st.markdown("Selecione quais variáveis você quer visualizar no tempo.")
                
                # Todas as variáveis numéricas possíveis
                all_metrics = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness', 'loudness']
                
                # Multiselect para o usuário escolher
                metrics_selected = st.multiselect(
                    "Escolha as variáveis:", 
                    all_metrics, 
                    default=['energy', 'valence'] # Padrão inicial
                )
                
                if metrics_selected:
                    yearly = df_unique.groupby('year')[metrics_selected].mean().reset_index()
                    yearly_melted = yearly.melt(id_vars='year', var_name='Métrica', value_name='Valor')
                    
                    fig_line = px.line(
                        yearly_melted, x='year', y='Valor', color='Métrica',
                        markers=True,
                        title="Evolução Ano a Ano"
                    )
                    # Adicionar linhas verticais
                    fig_line.add_vline(x=2000.5, line_dash="dash", line_color="gray")
                    fig_line.add_vline(x=2010.5, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("Selecione pelo menos uma variável acima para gerar o gráfico.")

        # --- ABA 4: POPULARIDADE ---
        with tab4:
            st.header("Popularidade Atual (2020)")
            
            pop_periodo = df_unique.groupby('periodo')['track_popularity'].mean().reset_index()
            fig_pop_bar = px.bar(pop_periodo, x='periodo', y='track_popularity', color='periodo', color_discrete_sequence=px.colors.sequential.YlOrBr, text_auto='.1f', title="Média por Década")
            fig_pop_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_pop_bar, use_container_width=True)

            st.divider()

            st.subheader("Evolução Detalhada (Ano a Ano)")
            pop_ano = df_unique.groupby('year')['track_popularity'].mean().reset_index()
            fig_pop_line = px.line(pop_ano, x='year', y='track_popularity', title="Trajetória da Popularidade Temporal", markers=True, color_discrete_sequence=['gold'])
            fig_pop_line.add_vline(x=2000.5, line_dash="dash", line_color="gray")
            fig_pop_line.add_vline(x=2010.5, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pop_line, use_container_width=True)

        # --- ABA 5: FERRAMENTA DE TESTES ---
        with tab5:
            st.header("🧪 Teste de Hipótese (Comparação)")
            st.markdown("Compare duas décadas para verificar se a diferença é estatisticamente significativa (Significância de 5%).")
            st.divider()

            tipo_teste = st.radio("Tipo de Variável:", ["Numérica (ex: Energia, Duração)", "Categórica (ex: Gênero, Tonalidade)"], horizontal=True)

            col_a, col_b = st.columns(2)
            decadas = sorted(df_unique['periodo'].unique())
            decada_1 = col_a.selectbox("Década A", decadas, index=0)
            decada_2 = col_b.selectbox("Década B", decadas, index=1)

            if decada_1 == decada_2:
                st.warning("Selecione décadas diferentes.")
            else:
                df_d1 = df_unique[df_unique['periodo'] == decada_1]
                df_d2 = df_unique[df_unique['periodo'] == decada_2]

                if "Numérica" in tipo_teste:
                    mapa_variaveis_num = {
                        "Dançabilidade": "danceability", "Energia": "energy", "Positividade (Valence)": "valence",
                        "Acústico": "acousticness", "Instrumental": "instrumentalness", "Fala (Speechiness)": "speechiness",
                        "Popularidade": "track_popularity", "Duração (ms)": "duration_ms", "Volume (Loudness)": "loudness", "Tempo (BPM)": "tempo"
                    }
                    variavel_display = st.selectbox("Variável", list(mapa_variaveis_num.keys()))
                    variavel_interna = mapa_variaveis_num[variavel_display]

                    if st.button("Calcular Teste t"):
                        d1 = df_d1[variavel_interna].dropna()
                        d2 = df_d2[variavel_interna].dropna()
                        stat, p_val = ttest_ind(d1, d2, equal_var=False)
                        
                        m1, m2 = d1.mean(), d2.mean()
                        col1, col2 = st.columns(2)
                        col1.metric(f"Média {decada_1}", f"{m1:.4f}")
                        col2.metric(f"Média {decada_2}", f"{m2:.4f}", delta=f"{m2-m1:.4f}")
                        
                        st.markdown("### Interpretação do Resultado")
                        if p_val < 0.05:
                            direcao = "aumentou" if m2 > m1 else "diminuiu"
                            st.success(f"✅ **Diferença Significativa!**")
                            st.write(f'Dado um p-valor de `{p_val:.10f}` (que é menor que 0.05), **rejeitamos a hipótese nula** de igualdade entre as médias.')
                            st.write(f'Isso indica estatisticamente que a **{variavel_display}** **{direcao}** quando comparamos o período **{decada_1}** com o período **{decada_2}**.')
                        else:
                            st.warning("❌ **Sem Diferença Significativa.**")
                            st.write(f'Dado um p-valor de `{p_val:.4f}` (que é maior que 0.05), **falhamos em rejeitar a hipótese nula**.')
                            st.write(f'Isso significa que não há evidência estatística suficiente para afirmar que a **{variavel_display}** mudou entre **{decada_1}** e **{decada_2}**.')

                else:
                    mapa_variaveis_cat = {"Gênero da Playlist": "playlist_genre", "Subgênero": "playlist_subgenre", "Tonalidade (Modo)": "mode_categoria"}
                    variavel_cat_display = st.selectbox("Categoria", list(mapa_variaveis_cat.keys()))
                    variavel_cat_interna = mapa_variaveis_cat[variavel_cat_display]
                    valores = sorted(df_unique[variavel_cat_interna].dropna().unique().astype(str))
                    alvo = st.selectbox(f"Valor específico a testar em '{variavel_cat_display}'", valores)

                    if st.button("Calcular Teste de Proporção"):
                        count1 = len(df_d1[df_d1[variavel_cat_interna].astype(str) == alvo])
                        total1 = len(df_d1)
                        count2 = len(df_d2[df_d2[variavel_cat_interna].astype(str) == alvo])
                        total2 = len(df_d2)

                        z_stat, p_val, p1, p2 = z_test_proportions(count1, total1, count2, total2)

                        col1, col2 = st.columns(2)
                        col1.metric(f"% em {decada_1}", f"{p1:.2%}", help=f"{count1}/{total1}")
                        col2.metric(f"% em {decada_2}", f"{p2:.2%}", delta=f"{(p2-p1)*100:.2f} p.p.")
                        
                        st.markdown("### Interpretação do Resultado")
                        if p_val < 0.05:
                            direcao = "aumentou" if p2 > p1 else "diminuiu"
                            st.success(f"✅ **Mudança Significativa na Proporção!**")
                            st.write(f'Dado um p-valor de `{p_val:.10f}`, **rejeitamos a hipótese nula** de que as proporções são iguais.')
                            st.write(f'Isso indica que a presença de **"{alvo}"** **{direcao}** significativamente quando comparamos **{decada_1}** com **{decada_2}**.')
                        else:
                            st.warning(f"❌ **Proporção Estável.**")
                            st.write(f'Dado um p-valor de `{p_val:.4f}`, **falhamos em rejeitar a hipótese nula**.')
                            st.write(f'Não há evidência estatística de que a proporção de **"{alvo}"** tenha mudado entre **{decada_1}** e **{decada_2}**.')
