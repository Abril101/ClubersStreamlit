import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

# --- Cargar datos ---
df_str = pd.read_csv('fdfStr_name.csv')
df_rest_info = pd.read_csv('df_rest_info.csv')
df_personas = pd.read_csv('df_personas.csv')

# --- Normalizar columnas ---
df_str.columns = [c.strip().replace(' ', '_') for c in df_str.columns]
df_rest_info.columns = [c.strip().replace(' ', '_') for c in df_rest_info.columns]
df_personas.columns = [c.strip().replace(' ', '_') for c in df_personas.columns]

# --- Renombrar columnas de categorías en df_personas ---
categoria_cols = ['Asiática', 'Cafeterias', 'Bebidas', 'Maritimo',
                  'Comida_Rápida', 'Desayunos', 'Italiana', 'Postres', 'Saludable']
rename_mapping = {col: f'category_{col}' for col in categoria_cols}
df_personas.rename(columns=rename_mapping, inplace=True)

# --- Columnas de características ---
feature_cols = [col for col in df_str.columns if col.startswith('category_')]

# --- Matrices para SVD ---
X_dishes = df_str[feature_cols].values
X_users = df_personas[feature_cols].values

# --- SVD ---
svd = TruncatedSVD(n_components=3, random_state=42)
X_dishes_svd = svd.fit_transform(X_dishes)
X_users_svd = svd.transform(X_users)

# --- Función de recomendación (genérica) ---
def recommend_from_vector(user_vector, top_n=5):
    user_svd = svd.transform(user_vector.reshape(1, -1))
    dists = cosine_distances(user_svd, X_dishes_svd)[0]
    nearest = dists.argsort()[:top_n]
    recs = df_str.iloc[nearest].copy()
    recs['similarity'] = 1 - dists[nearest]
    recs = recs.merge(df_rest_info, on='EstablishmentId', how='left')
    return recs[['Name', 'RestaurantName', 'similarity']]

# --- Interfaz ---
st.set_page_config(page_title="Recomendador de Platillos", layout="centered")
st.title("🍽️ Recomendador de Platillos por Gustos del Cliente")

st.markdown("## ✍️ Crear nuevo cliente")
with st.form("nuevo_cliente_form"):
    new_id = st.text_input("ID del nuevo cliente (solo para referencia):", "nuevo_cliente")
    user_vector = []
    for col in feature_cols:
        val = st.slider(f"{col.replace('category_', '')}:", 0.0, 1.0, 0.2, 0.05)
        user_vector.append(val)
    submitted = st.form_submit_button("Recomendar platillos")
    if submitted:
        user_vector = pd.Series(user_vector, index=feature_cols)
        recomendaciones = recommend_from_vector(user_vector.values, top_n=5)
        st.success(f"Recomendaciones para el cliente: {new_id}")
        st.dataframe(recomendaciones)

st.markdown("---")
st.markdown("## 👥 Ver recomendaciones de clientes existentes")
selected_id = st.selectbox("Selecciona un cliente:", df_personas['NumeroSocioConsumidor'].unique())
if st.button("Mostrar recomendaciones"):
    idx = df_personas[df_personas['NumeroSocioConsumidor'] == selected_id].index[0]
    user_vector = df_personas.loc[idx, feature_cols]
    recomendaciones = recommend_from_vector(user_vector.values, top_n=5)
    st.success(f"Recomendaciones para el cliente: {selected_id}")
    st.dataframe(recomendaciones)









