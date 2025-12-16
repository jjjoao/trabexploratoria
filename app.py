import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, norm
import numpy as np

# --- Configuração da Página ---
st.set_page_config(
    page_title="Spotify Insights", 
    layout="wide",
    page_icon="🎧",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CSS PERSONALIZADA ---
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES ---

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

# --- SIDEBAR (NAVEGAÇÃO E FILTROS) ---
st.sidebar.title("🎧 Spotify Insights")
pagina = st.sidebar.radio("Navegação", ["🏠 Home / Apresentação", "📊 Dashboard Interativo"])

st.sidebar.markdown("---")

# FILTROS GLOBAIS (Só aparecem no Dashboard)
if df is not None and pagina == "📊 Dashboard Interativo":
    st.sidebar.header("🔍 Filtros Globais")
    st.sidebar.markdown("Os filtros abaixo afetam **todos** os gráficos.")
    
    # Filtro de Gênero
    todos_generos = sorted(df['playlist_genre'].unique())
    generos_sel = st.sidebar.multiselect("Filtrar por Gênero:", todos_generos, default=todos_generos)
    
    # Aplicar Filtro
    if not generos_sel:
        st.sidebar.warning("Selecione pelo menos um gênero.")
        df_dashboard = df[df['periodo'] == 'Inexistente'] # Retorna vazio
    else:
        df_dashboard = df[df['playlist_genre'].isin(generos_sel)]
        
    df_unique = df_dashboard.drop_duplicates(subset=['track_id'])
    
    st.sidebar.info(f"Músicas filtradas: **{len(df_unique):,}**")

else:
    # Se estiver na Home, usa o dataset completo apenas para info básica
    if df is not None:
        df_unique = df.drop_duplicates(subset=['track_id'])

# --- CONTEÚDO PRINCIPAL ---

if df is not None:

    # === PÁGINA 1: HOME ===
    if pagina == "🏠 Home / Apresentação":
        st.title("🎵 A Evolução da Música Popular (1991-2020)")
        
        col_hero1, col_hero2 = st.columns([2, 1])
        with col_hero1:
            st.markdown("""
            Bem-vindo ao **Spotify Insights**. Este dashboard explora como a música mudou nas últimas três décadas.
            
            A análise foca em três pilares principais:
            1.  **A "Compactação" da Música:** A redução drástica na duração das faixas.
            2.  **A Mudança de Humor:** A transição de músicas felizes para tons mais melancólicos.
            3.  **A Guerra dos Gêneros:** A ascensão do EDM e a queda do Rock nas paradas.
            """)
            st.info("👈 **Use a barra lateral** para acessar o Dashboard e aplicar filtros dinâmicos!")
        
        with col_hero2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1024px-Spotify_logo_without_text.svg.png", width=150)

        st.divider()
        st.subheader("📚 Dicionário de Variáveis")
        with st.expander("Clique para ver o significado das métricas de áudio"):
            st.markdown("""
            * **Danceability:** Quão adequada a música é para dançar.
            * **Energy:** Intensidade e atividade da música.
            * **Valence:** Positividade (Alto = Feliz, Baixo = Triste).
            * **Acousticness:** Presença de instrumentos acústicos.
            * **Instrumentalness:** Ausência de vocais.
            * **Loudness:** Volume médio (dB).
            * **Popularity:** Índice atual de reproduções (0-100).
            """)

    # === PÁGINA 2: DASHBOARD ===
    elif pagina == "📊 Dashboard Interativo":
        
        # --- LINHA DE KPIs (INDICADORES) ---
        st.markdown("### ⚡ Visão Geral")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        top_artist = df_dashboard['track_artist'].mode()[0] if not df_dashboard.empty else "N/A"
        avg_bpm = df_dashboard['tempo'].mean()
        
        kpi1.metric("Total de Músicas", f"{len(df_unique):,}")
        kpi2.metric("Artista Top (Frequência)", top_artist)
        kpi3.metric("BPM Médio", f"{avg_bpm:.0f}")
        kpi4.metric("Duração Média", f"{df_unique['duration_ms'].mean()/60000:.2f} min")
        
        st.divider()

        # --- ABAS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⏱️ Duração", 
            "🎸 Gêneros", 
            "🎛️ Análise de Áudio (Radar)", 
            "⭐ Popularidade & Mood",
            "🧪 Teste Estatístico"
        ])

        # ABA 1: DURAÇÃO
        with tab1:
            st.subheader("A Queda na Duração")
            
            resumo_duracao = df_unique.groupby('periodo')['duration_ms'].mean().reset_index()
            resumo_duracao['minutos'] = resumo_duracao['duration_ms'] / 60000
            
            # Gráfico colorido e limpo
            fig_dur = px.bar(
                resumo_duracao, x='periodo', y='minutos', 
                color='periodo', 
                text_auto='.2f',
                title="Duração Média (Minutos) por Década",
                color_discrete_sequence=px.colors.qualitative.Prism # Cores vibrantes
            )
            fig_dur.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(fig_dur, use_container_width=True)

        # ABA 2: GÊNEROS
        with tab2:
            st.subheader("Paisagem de Gêneros")
            
            # Cálculo de proporção
            genre_data = df.groupby(['periodo', 'playlist_genre']).size().reset_index(name='count')
            genre_data['percentage'] = genre_data.groupby('periodo')['count'].transform(lambda x: x / x.sum())
            
            fig_genre = px.bar(
                genre_data, x="periodo", y="percentage", 
                color="playlist_genre", 
                title="Distribuição de Gêneros (% nas Playlists)",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Vivid # Cores bem distintas
            )
            fig_genre.layout.yaxis.tickformat = ',.0%'
            st.plotly_chart(fig_genre, use_container_width=True)

        # ABA 3: RADAR CHART (NOVO!)
        with tab3:
            col_radar, col_line = st.columns([1, 1])
            
            with col_radar:
                st.subheader("📸 Perfil Sonoro (Radar)")
                st.markdown("Compare a 'forma' das décadas.")
                
                # Preparar dados para Radar
                features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness']
                radar_df = df_unique.groupby('periodo')[features].mean().reset_index()
                
                # Criar Radar Chart com Graph Objects
                fig_radar = go.Figure()
                
                colors = ['#FF6692', '#636EFA', '#00CC96'] # Cores manuais bonitas
                
                for i, row in radar_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row[features].values,
                        theta=features,
                        fill='toself',
                        name=row['periodo'],
                        line_color=colors[i % len(colors)],
                        opacity=0.6
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=450,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col_line:
                st.subheader("📈 Evolução Temporal")
                metrics_sel = st.multiselect("Escolha as métricas:", features, default=['energy', 'valence'])
                
                yearly = df_unique.groupby('year')[metrics_sel].mean().reset_index()
                yearly_melted = yearly.melt(id_vars='year', var_name='Métrica', value_name='Valor')
                
                fig_line = px.line(
                    yearly_melted, x='year', y='Valor', color='Métrica',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_line.add_vline(x=2000.5, line_dash="dash", line_color="gray")
                fig_line.add_vline(x=2010.5, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_line, use_container_width=True)

        # ABA 4: POPULARIDADE E MOOD
        with tab4:
            st.subheader("⭐ Popularidade & Mood")
            
            col_pop1, col_pop2 = st.columns(2)
            
            with col_pop1:
                st.markdown("**Trajetória da Popularidade**")
                pop_ano = df_unique.groupby('year')['track_popularity'].mean().reset_index()
                fig_pop = px.area(
                    pop_ano, x='year', y='track_popularity',
                    color_discrete_sequence=['gold']
                )
                fig_pop.update_layout(yaxis_title="Score (0-100)")
                st.plotly_chart(fig_pop, use_container_width=True)
                
            with col_pop2:
                st.markdown("**Mapa de Humor (Energy vs Valence)**")
                # Amostragem para não pesar o gráfico (máx 500 pontos por década)
                sample_df = df_unique.groupby('periodo').apply(lambda x: x.sample(min(len(x), 300))).reset_index(drop=True)
                
                fig_scatter = px.scatter(
                    sample_df, x='valence', y='energy', color='periodo',
                    hover_data=['track_name', 'track_artist'],
                    title="Energia (Agitação) vs Valence (Felicidade)",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    opacity=0.7
                )
                fig_scatter.add_hline(y=0.5, line_dash="dot", line_color="gray")
                fig_scatter.add_vline(x=0.5, line_dash="dot", line_color="gray")
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("Quadrante Superior Direito: Músicas Felizes e Agitadas. Inferior Esquerdo: Tristes e Calmas.")

        # ABA 5: FERRAMENTA DE TESTES
        with tab5:
            st.header("🧪 Lab Estatístico")
            
            with st.container():
                st.markdown("Compare duas décadas para validar suas hipóteses.")
                
                c1, c2, c3 = st.columns([1, 1, 1])
                tipo = c1.radio("Tipo de Análise", ["Numérica (Médias)", "Categórica (Proporções)"])
                d1 = c2.selectbox("Década A", sorted(df_unique['periodo'].unique()), index=0)
                d2 = c3.selectbox("Década B", sorted(df_unique['periodo'].unique()), index=1)

                st.markdown("---")

                if d1 == d2:
                    st.warning("Selecione períodos diferentes.")
                else:
                    g1 = df_unique[df_unique['periodo'] == d1]
                    g2 = df_unique[df_unique['periodo'] == d2]

                    if "Numérica" in tipo:
                        col_var, col_btn = st.columns([2, 1])
                        mapa_num = {
                            "Dançabilidade": "danceability", "Energia": "energy", 
                            "Positividade": "valence", "Popularidade": "track_popularity", 
                            "Duração": "duration_ms", "BPM": "tempo"
                        }
                        var = col_var.selectbox("Variável", list(mapa_num.keys()))
                        var_code = mapa_num[var]
                        
                        if col_btn.button("Analisar Diferença"):
                            v1, v2 = g1[var_code].dropna(), g2[var_code].dropna()
                            stat, p = ttest_ind(v1, v2, equal_var=False)
                            
                            m1, m2 = v1.mean(), v2.mean()
                            delta = m2 - m1
                            
                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric(f"Média {d1}", f"{m1:.3f}")
                            res_col2.metric(f"Média {d2}", f"{m2:.3f}", delta=f"{delta:.3f}")
                            
                            p_formatted = "< 0.001" if p < 0.001 else f"{p:.4f}"
                            res_col3.metric("Valor-p", p_formatted)
                            
                            if p < 0.05:
                                st.success(f"**Resultado Significativo:** A variável '{var}' mudou estatisticamente.")
                            else:
                                st.info("Resultado não significativo. A diferença pode ser acaso.")

                    else: # Categórica
                        col_cat, col_val, col_btn = st.columns([1, 1, 1])
                        mapa_cat = {"Gênero": "playlist_genre", "Tonalidade": "mode_categoria"}
                        cat = col_cat.selectbox("Categoria", list(mapa_cat.keys()))
                        cat_code = mapa_cat[cat]
                        
                        val = col_val.selectbox("Valor Alvo", sorted(df_unique[cat_code].dropna().unique().astype(str)))
                        
                        if col_btn.button("Analisar Proporção"):
                            c1_val = len(g1[g1[cat_code].astype(str) == val])
                            c2_val = len(g2[g2[cat_code].astype(str) == val])
                            
                            z, p, p1, p2 = z_test_proportions(c1_val, len(g1), c2_val, len(g2))
                            
                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric(f"% em {d1}", f"{p1:.1%}")
                            res_col2.metric(f"% em {d2}", f"{p2:.1%}", delta=f"{(p2-p1)*100:.1f} p.p.")
                            res_col3.metric("Valor-p", f"{p:.4f}")
                            
                            if p < 0.05:
                                st.success(f"**Significativo:** A proporção de '{val}' mudou.")
                            else:
                                st.info("Mudança de proporção não significativa.")

