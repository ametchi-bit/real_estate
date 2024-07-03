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
)
from generate_report import generate_report, export_to_xlsx
import os

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
st.logo(SIDEBAR_LOGO, icon_image=MAIN_BODY_LOGO)

# title
st.markdown(
    '<h1 style="text-align: left; color: #FF4B4B;">Analyse des caméras de surveillance</h1>',
    unsafe_allow_html=True,
)

st.divider()

visualize_sunburst(symphonia_data)
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

st.write(selected_df)


# Sidebar for selecting task
def main():
    st.sidebar.title("Barre d'outils")
    task = st.sidebar.radio(
        "Tâche", ["Analyse descriptive", "Visualisation", "Générer le rapport"]
    )

    if task == "Analyse descriptive":
        st.subheader("Analyse Descriptive")
        vehicule_comparison = (
            selected_df.groupby(["Type d'objet", "Scénario"])["Nombre"]
            .sum()
            .reset_index()
        )
        st.write("Comparaison par type d'objet")
        st.write(vehicule_comparison)

    elif task == "Visualisation":
        with st.container():
            st.markdown(
                '<div class="custom-container">'
                '<h2 style="text-align: center; color: #FF4B4B;">Visualisation de données détaillée des différentes caméras de surveillance</h2>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            data_visualzation(selected_df)

    elif task == "Générer le rapport":
        st.subheader("Générer le rapport")
        report_format = st.selectbox(
            "Sélectionnez le format de rapport", ["Excel", "PDF"]
        )
        report_title = st.text_input(
            "Entrez le titre du rapport", "Rapport des Données de Surveillance"
        )
        report_subtitle = (
            option  # Use the name of the selected dataframe as the subtitle
        )
        if st.button("Générer"):
            if report_format == "Excel":
                xlsx_data = export_to_xlsx(selected_df)
                st.download_button(
                    label="Télécharger le fichier Excel",
                    data=xlsx_data,
                    file_name="rapport.xlsx",
                    mime="application/vnd.ms-excel",
                )
            elif report_format == "PDF":
                pdf_data = generate_report(report_title, report_subtitle, selected_df)
                if pdf_data:  # Ensure the data is not None
                    st.download_button(
                        label="Télécharger le fichier PDF",
                        data=pdf_data,
                        file_name="rapport.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.error("Erreur lors de la génération du rapport PDF.")
            st.success(f"Rapport {report_format} généré avec succès!")


if __name__ == "__main__":
    main()
