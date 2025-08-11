import streamlit as st
import pandas as pd
from PIL import Image
from collections import deque
import os
import re

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="Asistente Yape 2.1",
    page_icon="🤖",
    layout="wide"
)

# Estilo personalizado con el color de la marca
YAPE_PURPLE = "#5E4DB2"
st.markdown(f"""
<style>
    [data-theme="light"] {{ --primary: {YAPE_PURPLE}; }}
    .stButton>button[kind="primary"] {{ background-color: {YAPE_PURPLE}; color: white; }}
    .stButton>button {{ border-color: {YAPE_PURPLE}; }}
</style>
""", unsafe_allow_html=True)


# --- 2. GESTIÓN DE DATOS (CARGA Y GUARDADO) ---

@st.cache_data(ttl=3600)
def load_main_data(file_path):
    """Carga, limpia y pre-procesa el CSV principal de speeches, ahora con Tags."""
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Texto_del_Speech'], inplace=True)
        df['Texto_del_Speech'] = df['Texto_del_Speech'].str.replace('<br>', '\n', regex=False)
        for col in ['ID_Bloque', 'ID_Siguiente_Paso']:
            df[col] = df[col].astype(str).str.strip().replace('nan', '')
        
        if 'Tags' not in df.columns:
            df['Tags'] = ''
        df['Tags'] = df['Tags'].fillna('').astype(str).str.lower()
            
        df['search_text'] = df['Titulo_del_Bloque'].str.lower() + " " + \
                            df['Texto_del_Speech'].str.lower() + " " + \
                            df['Tags']
        return df
    except FileNotFoundError:
        return None

def load_csv(file_path, columns=None):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
    return pd.read_csv(file_path)

def save_main_data(df, file_path):
    df.to_csv(file_path, index=False)
    st.cache_data.clear()

def log_action(file_path, data_dict):
    df_log = load_csv(file_path, columns=data_dict.keys())
    is_usage_log = 'Usos' in data_dict
    if is_usage_log and not df_log.empty and not df_log[df_log['ID_Bloque'] == data_dict['ID_Bloque']].empty:
        idx = df_log.index[df_log['ID_Bloque'] == data_dict['ID_Bloque']][0]
        df_log.loc[idx, 'Usos'] += 1
    else:
        new_entry = pd.DataFrame([data_dict])
        df_log = pd.concat([df_log, new_entry], ignore_index=True)
    df_log.to_csv(file_path, index=False)

# --- 3. LÓGICA DE BÚSQUEDA Y ANÁLISIS "JARVIS NIVEL 2.1" ---

# --- 3A. ANÁLISIS DE SENTIMIENTO (MEJORADO) ---
SENTIMENT_KEYWORDS = {
    'negativo_alto': ['mierda', 'odio', 'joder', 'puta', 'estafa', 'robo', 'ladrones', 'denuncia', 'indecopi'],
    'negativo_medio': ['pésimo', 'nunca', 'basura', 'inútil', 'terrible', 'horrible', 'decepcionado', 'frustrado', 'molesto', 'enojado', 'rabia', 'exijo', 'demando', 'problema', 'queja', 'reclamo', 'deficiente', 'malo'],
    'positivo': ['gracias', 'excelente', 'genial', 'perfecto', 'maravilloso', 'increíble', 'amo', 'encanta', 'solucionado', 'ayuda', 'rápido', 'eficiente', 'amable', 'gracias']
}

def analyze_sentiment(query):
    """Analiza el texto y devuelve un sentimiento y un emoji."""
    query_lower = query.lower()
    
    # Usar búsqueda de substring en lugar de palabras exactas para más flexibilidad
    if any(word in query_lower for word in SENTIMENT_KEYWORDS['negativo_alto']):
        return 'Crítico / Insulto', '🚨'
    if any(word in query_lower for word in SENTIMENT_KEYWORDS['negativo_medio']):
        return 'Enojado / Queja', '😡'
    if any(word in query_lower for word in SENTIMENT_KEYWORDS['positivo']):
        return 'Amable / Positivo', '😊'
    return 'Neutral', '😐'

# --- 3B. BÚSQUEDA POR CONCEPTOS Y TAGS ---
SPANISH_STOP_WORDS = set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'ha', 'me', 'si', 'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'también', 'mi', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'esto', 'mí', 'antes', 'algunos', 'qué', 'entre', 'ser', 'era', 'está', 'puedo', 'ayuda'])
CONCEPT_KEYWORDS = {
    'bloqueo': ['bloqueo', 'bloqueado', 'bloquear', 'desbloqueo', 'acceso', 'entrar', 'ingresar', 'restringido', 'inactiva'],
    'transaccion': ['transacción', 'transferencia', 'enviado', 'enviar', 'recibido', 'recibir', 'movimiento', 'dinero', 'plata', 'yapeo', 'pago', 'cobro'],
    'clave': ['clave', 'contraseña', 'pin', 'olvidé', 'cambiar', 'restablecer', 'secreto'],
    'registro': ['registro', 'registrarme', 'crear', 'nueva', 'afiliarme', 'afiliación'],
    'error': ['error', 'problema', 'falla', 'fallando', 'funciona', 'inconveniente'],
    'datos': ['datos', 'actualizar', 'nombre', 'correo', 'celular', 'número', 'informacion']
}

def find_best_match_pro(query, df):
    if not query or not query.strip() or df is None: return None
    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    cleaned_query_words = query_words - SPANISH_STOP_WORDS
    if not cleaned_query_words: return None

    scores = pd.Series(0, index=df.index, dtype=float)
    for word in cleaned_query_words:
        for concept, keywords in CONCEPT_KEYWORDS.items():
            if word in keywords:
                for keyword_variant in keywords:
                    scores[df['search_text'].str.contains(keyword_variant, na=False)] += 2.0
                    scores[df['Titulo_del_Bloque'].str.lower().str.contains(keyword_variant, na=False)] += 5.0
        scores[df['search_text'].str.contains(word, na=False)] += 1.0
        scores[df['Titulo_del_Bloque'].str.lower().str.contains(word, na=False)] += 2.0
    return scores.idxmax() if scores.max() > 0 else None

def find_placeholders(text):
    return re.findall(r'\[([\w\s_]+)\]', text)

# --- 4. INICIALIZACIÓN DE DATOS Y ESTADO DE SESIÓN ---
if 'selected_index' not in st.session_state: st.session_state.selected_index = None
if 'favorites' not in st.session_state: st.session_state.favorites = []
if 'history' not in st.session_state: st.session_state.history = deque(maxlen=7)
if 'editing_mode' not in st.session_state: st.session_state.editing_mode = False
if 'placeholders' not in st.session_state: st.session_state.placeholders = {}
if 'feedback_submitted' not in st.session_state: st.session_state.feedback_submitted = {}
if 'query_sentiment' not in st.session_state: st.session_state.query_sentiment = None

df_main = load_main_data('Speech.csv')

# --- 5. INTERFAZ DE USUARIO (UI) ---

# --- BARRA LATERAL (SIDEBAR) ---
# (Sin cambios)
try: st.sidebar.image(Image.open('yap1.png'))
except FileNotFoundError: st.sidebar.title("Asistente Yape")
st.sidebar.title(f"Hola, Enrique 🤖")
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 Búsqueda Manual", expanded=True):
    if df_main is not None:
        main_categories = ['Todos los Casos'] + sorted(df_main['Categoria_Principal'].unique().tolist())
        selected_category = st.selectbox("Filtrar por Categoría:", main_categories)
        search_query = st.text_input("Buscar por palabra clave:", placeholder="Ej: bloqueo, transacción")
    else: st.sidebar.warning("No se encuentra `Speech.csv`")
with st.sidebar.expander("⭐ Favoritos y ⏳ Historial"):
    st.subheader("Favoritos")
    if not st.session_state.favorites: st.info("Marca tus speeches más usados para verlos aquí.")
    elif df_main is not None:
        fav_df = df_main[df_main.index.isin(st.session_state.favorites)]
        for index, row in fav_df.iterrows():
            if st.button(f"▫️ {row['Titulo_del_Bloque']}", key=f"fav_view_{index}"): st.session_state.selected_index = index; st.rerun()
    st.subheader("Historial Reciente")
    if not st.session_state.history: st.info("Tus últimos speeches visitados aparecerán aquí.")
    elif df_main is not None:
        hist_df = df_main[df_main.index.isin(list(st.session_state.history))]
        for index in reversed(st.session_state.history):
            if index in hist_df.index:
                if st.button(f"▫️ {hist_df.loc[index]['Titulo_del_Bloque']}", key=f"hist_view_{index}"): st.session_state.selected_index = index; st.rerun()
with st.sidebar.expander("📊 Analíticas Rápidas"):
    analytics_df = load_csv('analytics.csv', columns=['ID_Bloque', 'Titulo', 'Usos'])
    if not analytics_df.empty:
        st.markdown("**Top 5 Speeches más usados:**")
        top_5 = analytics_df.groupby('Titulo')['Usos'].sum().nlargest(5)
        st.dataframe(top_5)
    else: st.info("Aún no hay datos de uso.")
with st.sidebar.expander("✍️ Mis Snippets Personales"):
    snippets_path = 'enrique_snippets.csv'
    snippets_df = load_csv(snippets_path, columns=['Snippet'])
    new_snippet = st.text_area("Añadir nuevo snippet:", height=100, key="snippet_input")
    if st.button("Guardar Snippet"):
        if new_snippet.strip(): log_action(snippets_path, {'Snippet': new_snippet}); st.rerun()
    if snippets_df is not None and not snippets_df.empty:
        st.markdown("**Mis snippets guardados:**")
        for i, row in snippets_df.iterrows(): st.code(row['Snippet'])

# --- PESTAÑAS PRINCIPALES ---
tab_asistente, tab_rendimiento = st.tabs(["🚀 Asistente Principal", "📈 Mi Rendimiento"])

with tab_asistente:
    st.title("Biblioteca de Respuestas Dinámica")
    st.header("🤖 Triaje Rápido: Analizador de Consultas")
    with st.container(border=True):
        customer_query = st.text_area("Pega aquí la consulta del cliente:", height=100, key="triage_input")
        if st.button("Analizar Consulta", type="primary"):
            if df_main is not None and customer_query.strip():
                sentiment, emoji = analyze_sentiment(customer_query)
                st.session_state.query_sentiment = f"**Tono del cliente detectado:** {sentiment} {emoji}"
                match_index = find_best_match_pro(customer_query, df_main)
                if match_index is not None: st.session_state.selected_index = match_index; st.toast("¡Respuesta sugerida!", icon="✅")
                else: st.warning("No se encontró una coincidencia clara. Usa la búsqueda manual."); st.session_state.query_sentiment = None
            else: st.warning("Por favor, ingresa una consulta para analizar.")
    
    if st.session_state.query_sentiment:
        st.info(st.session_state.query_sentiment)

    st.markdown("---")
    
    if df_main is not None:
        filtered_df = df_main.copy()
        if 'selected_category' in locals() and selected_category != 'Todos los Casos': filtered_df = filtered_df[filtered_df['Categoria_Principal'] == selected_category]
        if 'search_query' in locals() and search_query: mask = filtered_df.apply(lambda row: search_query.lower() in str(row['search_text']).lower(), axis=1); filtered_df = filtered_df[mask]
    else: filtered_df = pd.DataFrame(); st.error("Error crítico: No se puede cargar `Speech.csv`.")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.header("Lista de Respuestas")
        if not filtered_df.empty:
            for subcategory, group in filtered_df.groupby('Subcategoria_Topico'):
                st.markdown(f"**{subcategory}**")
                for index, row in group.iterrows():
                    is_favorite = index in st.session_state.favorites
                    if st.button(f"{'⭐' if is_favorite else '✩'} {row['Titulo_del_Bloque']}", key=f"select_{index}"):
                        st.session_state.selected_index = index; st.session_state.editing_mode = False; st.session_state.placeholders = {}; st.session_state.query_sentiment = None;
                        if index not in st.session_state.history: st.session_state.history.append(index)
                        log_action('analytics.csv', {'ID_Bloque': row['ID_Bloque'], 'Titulo': row['Titulo_del_Bloque'], 'Usos': 1}); st.rerun()
                st.markdown("---")
        else: st.warning("No se encontraron resultados para tu búsqueda.")

    with col2:
        if st.session_state.selected_index is not None:
            if st.button("🧹 Limpiar Selección", use_container_width=True):
                st.session_state.selected_index = None; st.session_state.query_sentiment = None; st.rerun()
            st.markdown("---")
            
        st.header("Detalles y Acciones")
        if st.session_state.selected_index is not None and df_main is not None and st.session_state.selected_index in df_main.index:
            idx = st.session_state.selected_index
            selected_row = df_main.loc[idx]

            if st.session_state.editing_mode:
                st.subheader(f"✏️ Editando: {selected_row['Titulo_del_Bloque']}")
                with st.form(key="edit_form"):
                    new_text = st.text_area("Texto del Speech:", value=selected_row['Texto_del_Speech'], height=250)
                    new_reco = st.text_area("Recomendación Interna:", value=selected_row['Recomendacion_Interna'], height=100)
                    if st.form_submit_button("💾 Guardar Cambios", type="primary"):
                        df_main.loc[idx, 'Texto_del_Speech'] = new_text; df_main.loc[idx, 'Recomendacion_Interna'] = new_reco
                        save_main_data(df_main, 'Speech.csv'); st.session_state.editing_mode = False; st.success("¡Speech actualizado!"); st.rerun()
                if st.button("Cancelar"): st.session_state.editing_mode = False; st.rerun()
            else:
                st.subheader(selected_row['Titulo_del_Bloque'])
                action_cols = st.columns(3)
                with action_cols[0]:
                    if st.button("⭐ Añadir a Favoritos" if idx not in st.session_state.favorites else "🌟 Quitar de Favoritos", use_container_width=True):
                        if idx in st.session_state.favorites: st.session_state.favorites.remove(idx)
                        else: st.session_state.favorites.append(idx)
                        st.rerun()
                with action_cols[1]:
                    if st.button("✏️ Editar Speech", use_container_width=True): st.session_state.editing_mode = True; st.rerun()
                with action_cols[2]:
                    if st.button("🚩 Marcar p/ Revisión", use_container_width=True): log_action('review_log.csv', {'ID_Bloque': selected_row['ID_Bloque'], 'Titulo': selected_row['Titulo_del_Bloque']}); st.toast("Marcado para revisión. ¡Gracias!", icon="🚩")
                
                final_text = selected_row['Texto_del_Speech']
                placeholders = find_placeholders(final_text)
                if placeholders:
                    with st.container(border=True):
                        st.markdown("##### ✏️ Completa las variables del mensaje:")
                        for ph in placeholders: user_input = st.text_input(f"Valor para `{ph}`:", key=f"{idx}_{ph}"); final_text = final_text.replace(f"[{ph}]", user_input, 1)

                if pd.notna(selected_row['Recomendacion_Interna']) and selected_row['Recomendacion_Interna'].strip(): st.info(f"💡 **Nota para ti:** {selected_row['Recomendacion_Interna']}")
                with st.container(border=True):
                    st.markdown("##### 💬 Texto Final para el Cliente:"); st.code(final_text, language=None)
                    st.caption(f"Caracteres: {len(final_text)}")
                
                next_step_id = selected_row['ID_Siguiente_Paso']
                if next_step_id:
                    next_step_row = df_main[df_main['ID_Bloque'] == next_step_id]
                    if not next_step_row.empty:
                        next_step_title = next_step_row.iloc[0]['Titulo_del_Bloque']; next_step_index = next_step_row.index[0]
                        st.markdown("---");
                        if st.button(f"Siguiente Paso Sugerido ➡️: **{next_step_title}**", type="primary"): st.session_state.selected_index = next_step_index; st.rerun()
                
                st.markdown("---")
                st.markdown("##### ¿Te fue útil esta respuesta?")
                feedback_key = selected_row['ID_Bloque']
                if st.session_state.feedback_submitted.get(feedback_key):
                     st.info("Gracias por tu feedback sobre este ítem.")
                else:
                    feedback_cols = st.columns(2)
                    if feedback_cols[0].button("👍 Sí, me sirvió", use_container_width=True, key=f"fb_pos_{idx}"):
                        log_action('feedback.csv', {'ID_Bloque': feedback_key, 'Feedback': '👍 Positivo', 'Comentario': ''})
                        st.session_state.feedback_submitted[feedback_key] = True; st.rerun()
                    if feedback_cols[1].button("👎 No, se puede mejorar", use_container_width=True, key=f"fb_neg_{idx}"):
                         st.session_state.feedback_submitted[feedback_key] = 'pending'; st.rerun()
                    if st.session_state.feedback_submitted.get(feedback_key) == 'pending':
                        with st.form(key='feedback_form'):
                            comment = st.text_input("¿Qué podríamos mejorar? (Opcional)")
                            if st.form_submit_button("Enviar Feedback"):
                                log_action('feedback.csv', {'ID_Bloque': feedback_key, 'Feedback': '👎 Negativo', 'Comentario': comment})
                                st.session_state.feedback_submitted[feedback_key] = True; st.rerun()
        else:
            st.info("Selecciona una respuesta de la lista o usa el analizador de consultas para empezar.")

with tab_rendimiento:
    st.header(f"📈 Panel de Rendimiento de Enrique")
    analytics_df = load_csv('analytics.csv', columns=['ID_Bloque', 'Titulo', 'Usos'])
    feedback_df = load_csv('feedback.csv', columns=['ID_Bloque', 'Feedback', 'Comentario'])
    if analytics_df.empty and feedback_df.empty:
        st.warning("Aún no se han generado datos de rendimiento.")
    else:
        total_usos = int(analytics_df['Usos'].sum()) if not analytics_df.empty else 0
        positivos, negativos = (0, 0)
        if not feedback_df.empty and 'Feedback' in feedback_df.columns:
            feedback_counts = feedback_df['Feedback'].value_counts()
            positivos = feedback_counts.get('👍 Positivo', 0)
            negativos = feedback_counts.get('👎 Negativo', 0)
        
        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Total de Usos", total_usos); kpi_cols[1].metric("Feedbacks Positivos 👍", positivos); kpi_cols[2].metric("Feedbacks Negativos 👎", negativos)
        st.markdown("---")
        viz_cols = st.columns(2)
        with viz_cols[0]:
            st.subheader("Speeches Más Utilizados")
            if not analytics_df.empty:
                top_speeches = analytics_df.groupby('Titulo')['Usos'].sum().nlargest(10)
                st.bar_chart(top_speeches)
            else: st.info("No hay datos de uso.")
        with viz_cols[1]:
            st.subheader("Últimos Feedbacks Recibidos")
            if not feedback_df.empty: st.dataframe(feedback_df.tail(10), hide_index=True, use_container_width=True)
            else: st.info("No hay feedbacks registrados.")

    st.markdown("---")
    st.header("🚩 Panel de Revisión (Supervisión)")
    review_df = load_csv('review_log.csv', columns=['ID_Bloque', 'Titulo'])
    if review_df.empty:
        st.info("¡Excelente! No hay speeches marcados para revisión.")
    else:
        st.warning("Los siguientes speeches han sido marcados para revisión:")
        st.dataframe(review_df, hide_index=True, use_container_width=True)