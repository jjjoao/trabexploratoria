import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuração da Página ---
st.set_page_config(page_title="Análise Spotify (1991-2020)", layout="wide")

st.title("🎵 Análise da Evolução Musical no Spotify (1991 - 2020)")
st.markdown("""
Esta apresentação analisa como as características das músicas, gêneros e popularidade 
mudaram ao longo das últimas três décadas, baseada no dataset `spotify_songs.csv`.
""")

# --- 1. Carregamento e Processamento de Dados ---
@st.cache_data
def load_data():
    # Tente carregar o arquivo. O usuário deve ter o arquivo na mesma pasta.
    try:
        df = pd.read_csv("spotify_songs.csv")
    except FileNotFoundError:
        st.error("Arquivo 'spotify_songs.csv' não encontrado. Por favor, coloque-o na mesma pasta do script.")
        return None

    # Processamento de Data (Lógica do substr do R)
    # Pega os primeiros 4 caracteres e converte para numérico
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

df = load_data()

if df is not None:
    # Criação de um dataset de músicas únicas (sem duplicatas de playlist) para estatísticas de áudio
    df_unique = df.drop_duplicates(subset=['track_id'])

    # --- Abas para Organizar a Apresentação ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumo Geral", 
        "🎸 Gêneros", 
        "Técnico (Áudio)", 
        "🎹 Tonalidade (Mode)", 
        "⭐ Popularidade"
    ])

    # --- ABA 1: RESUMO GERAL ---
    with tab1:
        st.header("Visão Geral por Década")
        
        # Tabela de Resumo (Equivalente ao summarise do R)
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

        st.subheader("Queda na Duração das Músicas")
        # Gráfico de Duração
        fig_duracao = px.bar(
            resumo, x='Período', y='Duração (min)', 
            color='Período', text_auto='.2f',
            title="Duração Média (Minutos) por Década"
        )
        fig_duracao.update_traces(textposition='outside')
        st.plotly_chart(fig_duracao, use_container_width=True)

    # --- ABA 2: EVOLUÇÃO DOS GÊNEROS ---
    with tab2:
        st.header("Mudança nos Gêneros Musicais")
        
        # Agrupamento para Gêneros (usa o df completo, não o unique, pois a playlist importa)
        genre_counts = df.groupby(['periodo', 'playlist_genre']).size().reset_index(name='n')
        # Calcular proporção
        genre_counts['total'] = genre_counts.groupby('periodo')['n'].transform('sum')
        genre_counts['proporcao'] = genre_counts['n'] / genre_counts['total']
        
        fig_genre = px.bar(
            genre_counts, x="periodo", y="proporcao", color="playlist_genre",
            title="Distribuição de Gêneros (Proporcional)",
            labels={"proporcao": "Proporção", "periodo": "Década", "playlist_genre": "Gênero"},
            barmode="group" # ou "stack" se preferir empilhado
        )
        fig_genre.layout.yaxis.tickformat = ',.0%'
        st.plotly_chart(fig_genre, use_container_width=True)

    # --- ABA 3: CARACTERÍSTICAS TÉCNICAS (ÁUDIO) ---
    with tab3:
        st.header("Evolução das Características de Áudio")
        
        # Dados anuais para linhas
        yearly_stats = df_unique.groupby('year')[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'loudness']].mean().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Energia, Dança e Positividade")
            # Melt para formato longo (pivot_longer do R)
            melted_main = yearly_stats.melt(id_vars='year', value_vars=['danceability', 'energy', 'valence'], var_name='Métrica', value_name='Valor')
            
            fig_lines1 = px.line(
                melted_main, x='year', y='Valor', color='Métrica',
                color_discrete_map={"danceability": "blue", "energy": "red", "valence": "green"},
                title="Tendências (1991-2020)"
            )
            # Adicionar linhas verticais (equivalente ao geom_vline)
            fig_lines1.add_vline(x=2000.5, line_dash="dash", line_color="gray")
            fig_lines1.add_vline(x=2010.5, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_lines1, use_container_width=True)
            
        with col2:
            st.subheader("Acústico, Instrumental e Fala")
            melted_sec = yearly_stats.melt(id_vars='year', value_vars=['acousticness', 'instrumentalness', 'speechiness'], var_name='Métrica', value_name='Valor')
            
            fig_lines2 = px.line(
                melted_sec, x='year', y='Valor', color='Métrica',
                title="Evolução de Elementos Específicos"
            )
            fig_lines2.add_vline(x=2000.5, line_dash="dash", line_color="gray")
            fig_lines2.add_vline(x=2010.5, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_lines2, use_container_width=True)

        st.subheader("Evolução do Volume (Loudness)")
        fig_loud = px.line(yearly_stats, x='year', y='loudness', title="Volume Médio (dB)", markers=True)
        fig_loud.add_vline(x=2000.5, line_dash="dash", line_color="gray")
        fig_loud.add_vline(x=2010.5, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_loud, use_container_width=True)

    # --- ABA 4: TONALIDADE (MODE) ---
    with tab4:
        st.header("Maior vs. Menor")
        
        # Contagem e Proporção
        mode_counts = df_unique.groupby(['periodo', 'mode_categoria']).size().reset_index(name='n')
        mode_counts['total'] = mode_counts.groupby('periodo')['n'].transform('sum')
        mode_counts['proporcao'] = mode_counts['n'] / mode_counts['total']
        
        fig_mode = px.bar(
            mode_counts, x="periodo", y="proporcao", color="mode_categoria",
            title="Proporção de Tonalidade (Maior vs Menor)",
            color_discrete_map={"Menor": "#E74C3C", "Maior": "#2ECC71"},
            text_auto='.1%'
        )
        fig_mode.layout.yaxis.tickformat = ',.0%'
        st.plotly_chart(fig_mode, use_container_width=True)
        st.info("Nota-se um aumento da tonalidade Menor (geralmente associada a músicas mais tristes ou sérias) na última década.")

    # --- ABA 5: POPULARIDADE ---
    with tab5:
        st.header("Popularidade Atual das Músicas")
        
        pop_stats = df_unique.groupby('periodo')['track_popularity'].mean().reset_index()
        
        fig_pop = px.bar(
            pop_stats, x='periodo', y='track_popularity',
            color='periodo',
            color_discrete_sequence=px.colors.sequential.YlOrBr,
            text_auto='.1f',
            title="Popularidade Média (Score 0-100)"
        )
        fig_pop.update_layout(showlegend=False)
        st.plotly_chart(fig_pop, use_container_width=True)
        st.markdown("""
        **Interpretação:**
        * **1991-2000:** Alta popularidade devido ao status de "Clássicos".
        * **2001-2010:** Menor média ("Vale do esquecimento").
        * **2011-2020:** Maior média devido ao fator "Recência" (hits atuais).
        """)

else:
    st.write("Aguardando arquivo de dados...")