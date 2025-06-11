import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

# --- 1) CARGA DE DATOS ---
df_str      = pd.read_csv('fdfStr_name.csv')
df_rest     = pd.read_csv('df_rest_info.csv')
df_personas = pd.read_csv('df_personas.csv')

# --- 2) NORMALIZAR COLUMNAS (sin espacios ni mayúsculas extrañas) ---
for df in (df_str, df_rest, df_personas):
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]

# --- 3) RENOMBRAR LAS 9 CATEGORÍAS exactamente como pediste ---
rename_mapping = {
    'Asiática':       'category_Asiática',
    'Bebidas':        'category_Bebidas',
    'Categoría_2':    'category_Categoría_2',
    'Categoría_8':    'category_Categoría_8',
    'Comida_Rápida':  'category_Comida_Rápida',
    'Desayunos':      'category_Desayunos',
    'Italiana':       'category_Italiana',
    'Postres':        'category_Postres',
    'Saludable':      'category_Saludable',
}

# Aplica el rename tanto al DataFrame de platos como al de personas
df_str.rename(columns=rename_mapping,      inplace=True)
df_personas.rename(columns=rename_mapping, inplace=True)

# --- 4) COLUMNAS DE FEATURES PARA SVD ---
feature_cols = [c for c in df_str.columns if c.startswith('category_')]

# --- 5) OPCIONAL: Label “bonitos” para los sliders ---
display_names = {
    'category_Asiática':       'Comida Asiática',
    'category_Bebidas':        'Cafeteria',
    'category_Categoría_2':    'Bebidas ',
    'category_Categoría_8':    'Mariscos ',
    'category_Comida_Rápida':  'Comida Rápida',
    'category_Desayunos':      'Desayunos',
    'category_Italiana':       'Italiana',
    'category_Postres':        'Postres',
    'category_Saludable':      'Saludable'
}

# --- 6) PREPARAR MATRICES Y ENTRENAR SVD ---
X_dishes   = df_str[feature_cols].values
X_users    = df_personas[feature_cols].values
svd        = TruncatedSVD(n_components=3, random_state=42)
X_dishes_svd = svd.fit_transform(X_dishes)

def recommend_from_vector(user_vec, top_n=5):
    u_svd = svd.transform(user_vec.reshape(1, -1))
    dists = cosine_distances(u_svd, X_dishes_svd)[0]
    idxs  = dists.argsort()[:top_n]
    recs  = df_str.iloc[idxs].copy()
    recs['similarity'] = 1 - dists[idxs]
    return recs.merge(df_rest, on='EstablishmentId', how='left')[
        ['Name', 'RestaurantName', 'similarity']
    ]

# --- 7) INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Recomendador de Platillos", layout="centered")
st.title("🍽️ Recomendador de Platillos por Gustos del Cliente")

# — Nuevo cliente —
st.header("✍️ Crear nuevo cliente")
with st.form("form_nuevo_cliente"):
    new_id = st.text_input("ID del cliente:", "nuevo_cliente")
    vector = []
    for col in feature_cols:
        label = display_names.get(col, col.replace('category_', ''))
        vector.append(st.slider(label, 0.0, 1.0, 0.5, 0.05))
    if st.form_submit_button("Recomendar platillos"):
        vec = pd.Series(vector, index=feature_cols)
        recs = recommend_from_vector(vec.values)
        st.success(f"Recomendaciones para «{new_id}»:")
        st.dataframe(recs)

st.markdown("---")

# — Clientes existentes —
st.header("👥 Recomendaciones de clientes existentes")
cliente = st.selectbox("Selecciona un cliente:", df_personas['NumeroSocioConsumidor'].unique())
if st.button("Mostrar recomendaciones"):
    row = df_personas[df_personas['NumeroSocioConsumidor'] == cliente].iloc[0]
    recs = recommend_from_vector(row[feature_cols].values)
    st.success(f"Recomendaciones para «{cliente}»:")
    st.dataframe(recs)


