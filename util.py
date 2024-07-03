import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd


def scenario_types(selected_df):
    # Get unique scenarios
    unique_scenario = selected_df["Scénario"].unique()
    return unique_scenario


def combine_dataframes(folder_path):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    dataframes = [pd.read_csv(file) for file in files]

    # Combine all dataframes
    symphonia_data = pd.concat(dataframes).reset_index(drop=True)

    combined_file_path = os.path.join(folder_path, "./combineDf/symphonia_data.csv")
    symphonia_data.to_csv(combined_file_path, index=False)

    return symphonia_data, combined_file_path


def load_combined_data(combined_file_path):
    # Load combined dataframe from CSV file
    symphonia_data = pd.read_csv(combined_file_path)
    return symphonia_data


# Define folder path and combine dataframes
folder_path = "dataframe"
symphonia_data, combined_file_path = combine_dataframes(folder_path)


def visualize_sunburst(symphonia_data):
    st.markdown(
        '<h3 style="text-align: left; color: #000000; margin-bottom: 40px">Vue globale de données des caméras de surveillance</h3>',
        unsafe_allow_html=True,
    )
    # st.write(symphonia_data)

    # Group and aggregate the data
    grouped_data = (
        symphonia_data.groupby(["Caméra", "Scénario", "Catégorie", "Type d'objet"])
        .agg({"Nombre": "sum"})
        .reset_index()
    )

    # Create ids, labels, and parents for the Sunburst chart
    ids = []
    labels = []
    parents = []

    for i, row in grouped_data.iterrows():
        camera = row["Caméra"]
        scenario = row["Scénario"]
        categorie = row["Catégorie"]
        type_objet = row["Type d'objet"]
        nombre = row["Nombre"]

        camera_id = f"{camera}"
        scenario_id = f"{camera} - {scenario}"
        categorie_id = f"{camera} - {scenario} - {categorie}"
        type_objet_id = f"{camera} - {scenario} - {categorie} - {type_objet}"

        if camera_id not in ids:
            ids.append(camera_id)
            labels.append(camera)
            parents.append("")

        if scenario_id not in ids:
            ids.append(scenario_id)
            labels.append(scenario)
            parents.append(camera_id)

        if categorie_id not in ids:
            ids.append(categorie_id)
            labels.append(categorie)
            parents.append(scenario_id)

        ids.append(type_objet_id)
        labels.append(f"{type_objet} ({nombre})")
        parents.append(categorie_id)

    # Create the Sunburst chart
    fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents))
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    st.plotly_chart(fig)


def data_visualzation(selected_df):

    def visualize_histogram(selected_df):
        st.markdown(
            '<h3 style="text-align: center;">Histogramme des volumes par mois et type d\'objet</h3>',
            unsafe_allow_html=True,
        )

        unique_scenario = scenario_types(selected_df)

        # Convert Horodatage to datetime and extract the month
        selected_df["Horodatage"] = pd.to_datetime(
            selected_df["Horodatage"], format="%d/%m/%Y %H:%M:%S"
        )
        selected_df["Month"] = selected_df["Horodatage"].dt.strftime("%Y-%m")

        # Group by Month, Type d'objet, and Scénario to sum Nombre
        grouped_df = (
            selected_df.groupby(["Month", "Type d'objet", "Scénario"])["Nombre"]
            .sum()
            .reset_index()
        )

        # Create columns for the histograms
        cols = st.columns(2)

        # Add filters
        selected_months = st.sidebar.multiselect(
            "Sélectionnez le mois", sorted(selected_df["Month"].unique())
        )
        selected_types = st.sidebar.multiselect(
            "Sélectionnez le type d'objet", sorted(selected_df["Type d'objet"].unique())
        )
        selected_scenarios = st.sidebar.multiselect(
            "Sélectionnez le scénario", unique_scenario.tolist()
        )

        # Filter data based on selected filters
        if selected_months:
            grouped_df = grouped_df[grouped_df["Month"].isin(selected_months)]

        if selected_types:
            grouped_df = grouped_df[grouped_df["Type d'objet"].isin(selected_types)]

        if selected_scenarios:
            grouped_df = grouped_df[grouped_df["Scénario"].isin(selected_scenarios)]

        # Loop through unique scenarios and plot them side by side
        for i, scenario in enumerate(unique_scenario):
            scenario_df = grouped_df[grouped_df["Scénario"] == scenario]

            fig = px.histogram(
                scenario_df,
                x="Month",
                y="Nombre",
                color="Type d'objet",
                barmode="group",
                title=f"Type de Scénario:{scenario}",
                labels={
                    "value": "Nombre",
                    "Month": "Mois",
                    "Type d'objet": "Type de véhicule",
                },
            )

            # Display the plot in the correct column
            col = cols[i % 2]
            with col:
                st.plotly_chart(fig)

    def visualize_doughnut(selected_df):
        unique_scenario = scenario_types(selected_df)

        # Convert Horodatage to datetime and extract the month
        selected_df["Horodatage"] = pd.to_datetime(
            selected_df["Horodatage"], format="%d/%m/%Y %H:%M:%S"
        )
        selected_df["Month"] = selected_df["Horodatage"].dt.strftime("%Y-%m")

        grouped_df = (
            selected_df.groupby(["Type d'objet", "Scénario"])["Nombre"]
            .sum()
            .reset_index()
        )

        # Add filters
        selected_months = st.sidebar.multiselect(
            "Sélectionnez le mois", sorted(selected_df["Month"].unique())
        )
        selected_types = st.sidebar.multiselect(
            "Sélectionnez le type d'objet", sorted(selected_df["Type d'objet"].unique())
        )
        selected_scenarios = st.sidebar.multiselect(
            "Sélectionnez le scénario", unique_scenario.tolist()
        )

        # Filter data based on selected filters
        if selected_months:
            grouped_df = grouped_df[grouped_df["Month"].isin(selected_months)]

        if selected_types:
            grouped_df = grouped_df[grouped_df["Type d'objet"].isin(selected_types)]

        if selected_scenarios:
            grouped_df = grouped_df[grouped_df["Scénario"].isin(selected_scenarios)]

        # Create doughnut pie chart for each scenario
        fig = make_subplots(
            rows=1,
            cols=len(unique_scenario),
            specs=[[{"type": "domain"}] * len(unique_scenario)],
            subplot_titles=[f"Scénario {scenario}" for scenario in unique_scenario],
        )

        for i, scenario in enumerate(unique_scenario):
            scenario_df = grouped_df[grouped_df["Scénario"] == scenario]

            fig.add_trace(
                go.Pie(
                    labels=scenario_df["Type d'objet"],
                    values=scenario_df["Nombre"],
                    name=f"Scénario {scenario}",
                    hole=0.5,
                    hoverinfo="label+percent+name",
                ),
                1,
                i + 1,
            )

        fig.update_layout(
            title_text="Visualisation par type de véhicule et scénario",
            # Add annotations in the center of the donut pies.
            annotations=[
                dict(
                    text=f"{scenario}",
                    x=(i + 0.5) / (len(unique_scenario) + 0.30),
                    y=0.5,
                    font_size=12,
                    showarrow=False,
                )
                for i, scenario in enumerate(unique_scenario)
            ],
            height=600,
            width=1400,
        )

        st.plotly_chart(fig)

    def my_radio(selected_df):
        task = st.sidebar.radio("Type de Visualisation", ["Histogram", "Donnut"], 0)
        if task == "Histogram":
            visualize_histogram(selected_df)
        elif task == "Donnut":
            visualize_doughnut(selected_df)

    my_radio(selected_df)
