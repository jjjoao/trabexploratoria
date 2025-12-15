import streamlit as st
import pandas as pd
import plotly.express as px

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
    
    return df_filtered

df = load_data()

# --- NAVEGAÇÃO LATERAL ---
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["🏠 Apresentação", "📊 Dashboard de Análise"])
st.sidebar.markdown("---")
st.sidebar.info("Dados extraídos via Spotifyr Package / TidyTuesday.")

if df is not None:
    # Dataset de músicas únicas para estatísticas de áudio
    df_unique = df.drop_duplicates(subset=['track_id'])

    # --- PÁGINA 1: APRESENTAÇÃO ---
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

        col1, col2 = st.columns(2)
        
        with col1:
            st.info("### 📂 Sobre o Banco de Dados")
            st.markdown(f"""
            * **Total de Registros Analisados:** {len(df_unique):,} músicas únicas.
            * **Período:** De 1991 a 2020.
            * **Fonte:** Dados extraídos da API do Spotify.
            """)
            
        with col2:
            st.warning("### 📖 Dicionário de Variáveis (Principais)")
            st.markdown("""
            * **Danceability:** O quão adequada a música é para dançar.
            * **Energy:** Medida de intensidade e atividade.
            * **Valence:** Positividade musical (Alto = Feliz, Baixo = Triste/Sério).
            * **Acousticness:** Confiança de que a música é acústica.
            * **Popularity:** Índice de 0 a 100 baseado na reprodução atual.
            """)

    # --- PÁGINA 2: DASHBOARD ---
    elif pagina == "📊 Dashboard de Análise":
        st.title("📊 Dashboard Analítico")

        # Abas do Dashboard
        tab1, tab2, tab3, tab4 = st.tabs([
            "📉 Estatísticas Gerais", 
            "🎸 Gêneros", 
            "🎛️ Características de Áudio", 
            "⭐ Popularidade"
        ])

        # --- ABA 1: ESTATÍSTICAS GERAIS ---
        with tab1:
            st.header("Resumo por Década")
            
            # Tabela de Resumo
            resumo = df_unique.groupby('periodo').agg({
                'duration_ms': lambda x: (x.mean() / 60000),
                'energy': 'mean',
                'valence': 'mean',
                'danceability': 'mean',
                'track_id': 'count'
            }).reset_index()
            
            resumo.columns = ['Período', 'Duração (min)', 'Energia', 'Positividade', 'Dançabilidade', 'Nº Músicas']
            
            st.dataframe(resumo.style.format({
                'Duração (min)': '{:.2f}', 
                'Energia': '{:.3f}', 
                'Positividade': '{:.3f}', 
                'Dançabilidade': '{:.3f}'
            }), use_container_width=True)

            st.subheader("A Queda na Duração das Músicas")
            st.markdown("Nota-se uma redução consistente no tempo médio das músicas, possivelmente devido à era do streaming.")
            
            fig_duracao = px.bar(
                resumo, x='Período', y='Duração (min)', 
                color='Período', text_auto='.2f',
                title="Duração Média (Minutos) por Década"
            )
            fig_duracao.update_traces(textposition='outside')
            st.plotly_chart(fig_duracao, use_container_width=True)

        # --- ABA 2: EVOLUÇÃO DOS GÊNEROS ---
        with tab2:
            st.header("Dominância de Gêneros")
            
            # Agrupamento para Gêneros
            genre_counts = df.groupby(['periodo', 'playlist_genre']).size().reset_index(name='n')
            genre_counts['total'] = genre_counts.groupby('periodo')['n'].transform('sum')
            genre_counts['proporcao'] = genre_counts['n'] / genre_counts['total']
            
            fig_genre = px.bar(
                genre_counts, x="periodo", y="proporcao", color="playlist_genre",
                title="Distribuição de Gêneros (% nas Playlists)",
                labels={"proporcao": "Proporção", "periodo": "Década", "playlist_genre": "Gênero"},
                barmode="group"
            )
            fig_genre.layout.yaxis.tickformat = ',.0%'
            st.plotly_chart(fig_genre, use_container_width=True)

        # --- ABA 3: CARACTERÍSTICAS TÉCNICAS ---
        with tab3:
            st.header("Tendências de Áudio (1991-2020)")
            
            yearly_stats = df_unique.groupby('year')[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'loudness']].mean().reset_index()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Humor e Ritmo")
                melted_main = yearly_stats.melt(id_vars='year', value_vars=['danceability', 'energy', 'valence'], var_name='Métrica', value_name='Valor')
                
                fig_lines1 = px.line(
                    melted_main, x='year', y='Valor', color='Métrica',
                    color_discrete_map={"danceability": "blue", "energy": "red", "valence": "green"},
                    title="Energia, Dançabilidade e Positividade"
                )
                fig_lines1.add_vline(x=2000.5, line_dash="dash", line_color="gray")
                fig_lines1.add_vline(x=2010.5, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_lines1, use_container_width=True)
                
            with col_b:
                st.subheader("Elementos Sonoros")
                melted_sec = yearly_stats.melt(id_vars='year', value_vars=['acousticness', 'instrumentalness', 'speechiness'], var_name='Métrica', value_name='Valor')
                
                fig_lines2 = px.line(
                    melted_sec, x='year', y='Valor', color='Métrica',
                    title="Acústico, Instrumental e Fala"
                )
                fig_lines2.add_vline(x=2000.5, line_dash="dash", line_color="gray")
                fig_lines2.add_vline(x=2010.5, line_dash="dash", line_color="gray")
                st.plotly_
