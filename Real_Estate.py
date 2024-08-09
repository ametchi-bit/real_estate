import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from util import (
    scenario_types,
    data_visualzation,
    symphonia_data,
    load_combined_data,
    combined_file_path,
    visualize_sunburst,
    analysis,
)
import os

from streamlit_modal import Modal


# CSS to style the container
st.markdown(
    """
    <style>
    .custom-container {
        background-color: #F0F2F6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# page configuration
# st.set_page_config(layout="wide")

# Define the paths to the logos
SIDEBAR_LOGO = "images/LogoDATARIUM.png"  # or any other path you prefer
MAIN_BODY_LOGO = "images/LogoDATARIUM_pill.png"  # or any other path you prefer

# Use st.logo() for the main body (branding)
#st.logo(SIDEBAR_LOGO, icon_image=MAIN_BODY_LOGO)

# title
st.markdown(
    '<h1 style="text-align: left; color: #FF4B4B;">Analyse des caméras de surveillance</h1>',
    unsafe_allow_html=True,
)

st.divider()

# Ensure the Horodatage column is in datetime format, assuming dayfirst format
symphonia_data["Horodatage"] = pd.to_datetime(
    symphonia_data["Horodatage"], dayfirst=True
)

# Calculate the date range
start_date = symphonia_data["Horodatage"].min()
end_date = symphonia_data["Horodatage"].max()
grouped_data = (
    symphonia_data.groupby(["Caméra", "Scénario", "Catégorie", "Type d'objet"])
    .agg({"Nombre": "sum"})
    .reset_index()
)
total_events = grouped_data["Nombre"].sum()

# visualization_sunburst_interpretation(symphonia_data)
st.markdown(
    '<h3 style="text-align: left; color: #000000; margin-bottom: 40px">Vue globale de données des caméras de surveillance</h3>',
    unsafe_allow_html=True,
)
st.markdown(f"**Nombre total d'événements enregistrés**: {total_events}")
st.markdown(f"**Plage de dates**: du {start_date.date()} au {end_date.date()}")
visualize_sunburst(symphonia_data)
st.markdown(
    """
        ### Analyse de la Structure Hiérarchique
        Le graphique en rayons représente la distribution hiérarchique des données des caméras de surveillance. Voici quelques points clés à considérer:

        - **Caméra**: Le niveau le plus externe représente les différentes caméras. Chaque segment montre la contribution d'une caméra au total des événements.
        - **Scénario**: Chaque segment de caméra est divisé en sous-segments représentant les différents scénarios observés par cette caméra.
        - **Catégorie**: Les segments de scénario sont ensuite divisés en catégories, illustrant les types d'événements spécifiques observés dans chaque scénario.
        - **Type d'objet**: Enfin, chaque catégorie est décomposée en types d'objets, indiquant la nature précise de chaque événement.

        Ce type de visualisation permet de comprendre rapidement la répartition et la hiérarchie des données, en identifiant les caméras et scénarios les plus actifs, ainsi que les types d'événements les plus fréquents.
        """
)


# Load datasets
symphonia_data = load_combined_data(combined_file_path)

# Load datasets
folder_path = "dataframe"
files = [
    f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
]
dataframes = {file: pd.read_csv(os.path.join(folder_path, file)) for file in files}

st.markdown(
    '<h3 style="text-align: left; color: #000000; margin-top: 40px">Données par caméras de surveillance</h3>',
    unsafe_allow_html=True,
)

# Create a list for selection
cam_list = list(dataframes.keys())

# Selectbox for choosing dataset
option = st.selectbox(
    "Sélectionnez la source des données",
    cam_list,
    index=0,
    placeholder="Sélectionnez les données à analyser...",
)


# Display selected dataset
st.write("Vous avez sélectionné:", option)
selected_df = dataframes[option]

# Initialize the session state variable if it does not exist
if "show_data" not in st.session_state:
    st.session_state.show_data = False

# Button to toggle the visibility
if st.session_state.show_data:
    if st.button("Masquer les données"):
        st.session_state.show_data = False
else:
    if st.button("Voir les données"):
        st.session_state.show_data = True

# Show or hide the dataframe based on the state
if st.session_state.show_data:
    st.write(selected_df)


# Sidebar for selecting task
def main():
    st.sidebar.title("Barre d'outils")
    task = st.sidebar.radio("Tâche", ["Analyse descriptive", "Visualisation"])

    if task == "Analyse descriptive":
        tab1, tab2 = st.tabs(["Analyse Descriptive", "Linear Regression"])
        with tab1:
            analysis(selected_df)

        with tab2:
            st.write("Aucune donnee")

    elif task == "Visualisation":
        with st.container():
            st.markdown(
                '<div class="custom-container">'
                '<h2 style="text-align: center; color: #FF4B4B;">Visualisation de données détaillée des différentes caméras de surveillance</h2>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            with st.markdown('<div class="custom-container">', unsafe_allow_html=True):
                data_visualzation(selected_df)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
