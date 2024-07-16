import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_modal import Modal
import plotly.io as pio
from io import BytesIO


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


def handle_missing_val_real_estate(data):
    """
    Filter out rows where 'Nombre' is zero.

    Args:
    - data (pd.DataFrame): Input DataFrame to filter.

    Returns:
    - filtered_data (pd.DataFrame): Data with rows where 'Nombre' is zero removed.
    """
    filtered_data = data[data["Nombre"] != 0]  # Filter rows where 'Nombre' is not zero
    return filtered_data


# Define folder path and combine dataframes
folder_path = "dataframe"
symphonia_data, combined_file_path = combine_dataframes(folder_path)

#  function for whole dataset need to implement a report

# Define column descriptions
column_descriptions = {
    "Horodatage": "Horodatage de l'enregistrement de données",
    "Caméra": "Identifiant de la caméra",
    "Scénario": "Description du scénario",
    "Catégorie": "Catégorie de l'objet détecté",
    "Type d'objet": "Type d'objet détecté",
    "Nombre": "Nombre d'objets détectés",
}


# Function to get column names and descriptions
def get_column_descriptions(symphonia_data):
    columns = symphonia_data.columns
    descriptions = {}
    for col in columns:
        description = column_descriptions.get(col, "No description available")
        descriptions[col] = description
    return descriptions


# Display the column names and descriptions using Streamlit
def display_column_descriptions(symphonia_data):
    st.subheader("Descriptions des colonnes")
    descriptions = get_column_descriptions(symphonia_data)
    for col, desc in descriptions.items():
        st.write(f"**{col}**: {desc}")


# summary description for streamlit display
def summary_desc_symphonia(symphonia_data):
    return symphonia_data.describe().T


def explain_summary_desc(df_desc):
    explanation = {}
    for column in df_desc.index:
        explanation[column] = {
            "count": f"Number of non-null entries in the {column} column.",
            "mean": f"Average value of the {column} column.",
            "std": f"Standard deviation (spread) of the values in the {column} column.",
            "min": f"Minimum value in the {column} column.",
            "25%": f"25th percentile value (first quartile) in the {column} column.",
            "50%": f"50th percentile value (median) in the {column} column.",
            "75%": f"75th percentile value (third quartile) in the {column} column.",
            "max": f"Maximum value in the {column} column.",
        }
    return explanation


def distribution_real_estate(symphonia_data, group_by_column, value_column):
    # Group by the specified column and sum the values
    data_distribution = symphonia_data.groupby(group_by_column)[value_column].sum().T
    return data_distribution


def plot_real_estate_histogram(symphonia_data, group_by_column, value_column):
    data_distribution = get_data_distribution(
        symphonia_data, group_by_column, value_column
    )
    fig = px.histogram(
        data_distribution,
        x=data_distribution.index,
        y=data_distribution.values,
        labels={"x": group_by_column, "y": "Count"},
    )
    fig.update_layout(
        title=f"Distribution of {value_column} by {group_by_column}",
        xaxis_title=group_by_column,
        yaxis_title="Count",
    )
    st.plotly_chart(fig)


def comparaison_type_objet_real_estate(symphonia_data):
    symphonia_data_filtered = handle_missing_val_real_estate(symphonia_data)
    vehicule_comparison = (
        symphonia_data_filtered.groupby(["Type d'objet", "Scénario"])["Nombre"]
        .sum()
        .reset_index()
    )
    return vehicule_comparison


def plot_comparaison_type_objet_real_estate(symphonia_data):
    symphonia_data_filtered = handle_missing_val_real_estate(symphonia_data)
    vehicule_comparison = (
        symphonia_data_filtered.groupby(["Type d'objet", "Scénario"])["Nombre"]
        .sum()
        .reset_index()
    )

    comparaison_fig = px.bar(
        vehicule_comparison,
        x="Scénario",
        y="Nombre",
        color="Type d'objet",
        barmode="group",
        title="Comparaison par type d'objet et scénario",
        labels={
            "Nombre": "Nombre",
            "Scénario": "Scénario",
            "Type d'objet": "Type d'objet",
        },
    )
    image_stream_comparaison = BytesIO()
    pio.write_image(comparaison_fig, image_stream_comparaison, format="png", width=955, height=525, scale=2)
    image_stream_comparaison.seek(0)
    return comparaison_fig, image_stream_comparaison


def repartition_viz(symphonia_data):
    # Group and aggregate the data
    grouped_data = (
        symphonia_data.groupby(["Caméra", "Scénario", "Catégorie", "Type d'objet"])
        .agg({"Nombre": "sum"})
        .reset_index()
    )

    # Extract relevant statistics for interpretation

    camera_counts = grouped_data.groupby("Caméra")["Nombre"].sum()
    scenario_counts = grouped_data.groupby("Scénario")["Nombre"].sum()
    category_counts = grouped_data.groupby("Catégorie")["Nombre"].sum()

    camera_data_list = []
    for camera, count in camera_counts.items():
        # st.markdown(f"- **{camera}**: {count} événements")
        camera_data_list.append({"Caméra": camera, "événements": count})
    rep_cam = pd.DataFrame(camera_data_list)

    # Creating the pie chart
    rep_cam_fig = px.pie(
        rep_cam,
        values="événements",
        names="Caméra",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    image_stream_cam = BytesIO()
    pio.write_image(rep_cam_fig, image_stream_cam, format="png", width=955, height=525, scale=2)
    image_stream_cam.seek(0)
    

    scenario_data_list = []
    for scenario, count in scenario_counts.items():
        # st.markdown(f"- **{scenario}**: {count} événements")
        scenario_data_list.append({"Scénario": scenario, "événements": count})
    rep_scen = pd.DataFrame(scenario_data_list)

    # Creating the pie chart
    rep_scen_fig = px.pie(
        rep_scen,
        values="événements",
        names="Scénario",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    image_stream_scen = BytesIO()
    pio.write_image(rep_scen_fig, image_stream_scen, format="png", width=955, height=525, scale=2)
    image_stream_scen.seek(0)

    categorie_data_list = []
    for category, count in category_counts.items():
        # st.markdown(f"- **{category}**: {count} événements")
        categorie_data_list.append({"Catégorie": category, "événements": count})
    rep_cat = pd.DataFrame(categorie_data_list)

    # Creating the pie chart
    fig_categorie = px.pie(
        rep_cat,
        values="événements",
        names="Catégorie",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    image_stream_cat = BytesIO()
    pio.write_image(fig_categorie, image_stream_cat, format="png", width=955, height=525, scale=2)
    image_stream_cat.seek(0)
    return rep_cam_fig, rep_scen_fig, fig_categorie, image_stream_cam, image_stream_cat, image_stream_scen

def calculate_monthly_weekly_daily_patterns_real_estate(symphonia_data, date_column, value_column):
    # Convert Horodatage to datetime and extract the month
    symphonia_data[date_column] = pd.to_datetime(symphonia_data[date_column], format="%d/%m/%Y %H:%M:%S")
    symphonia_data["Month"] = symphonia_data[date_column].dt.strftime("%Y-%m")

    monthly_pattern = symphonia_data.resample("M", on=date_column)[value_column].sum()
    weekly_pattern = symphonia_data.resample("W", on=date_column)[value_column].sum()
    daily_pattern = symphonia_data.resample("D", on=date_column)[value_column].sum()

    return monthly_pattern, weekly_pattern, daily_pattern


def plot_weekly_daily_patterns_real_estate(symphonia_data, date_column, value_column):
    
    monthly_pattern, weekly_pattern, daily_pattern = calculate_monthly_weekly_daily_patterns_real_estate(
        symphonia_data, date_column, value_column
    )

    mwd_fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Modèle Mensuel", "Modèle Hebdomadaire", "Modèle quotidien"),
    )

    mwd_fig.add_trace(
        go.Scatter(
            x=monthly_pattern.index, y=monthly_pattern, mode="lines", name="Mois"
        ),
        row=1,
        col=1,
    )

    mwd_fig.add_trace(
        go.Scatter(
            x=weekly_pattern.index, y=weekly_pattern, mode="lines", name="Semaine"
        ),
        row=2,
        col=1,
    )

    mwd_fig.add_trace(
        go.Scatter(x=daily_pattern.index, y=daily_pattern, mode="lines", name="Jour"),
        row=3,
        col=1,
    )

    mwd_fig.update_layout(
        title_text="Modèle Mensuel, Hebdomadaire, et quotidien",
        height=800,
        showlegend=True,
    )
    image_stream_mwd = BytesIO()
    pio.write_image(mwd_fig, image_stream_mwd, format="png", width=955, height=525, scale=2)
    image_stream_mwd.seek(0)
    
    return mwd_fig, image_stream_mwd
    
# end of it
def count_by_type_objet(selected_df):
    # Filter to ensure only relevant types of objects are considered
    relevant_objects = ["bus", "moto", "personne", "véhicule intermédiaire", "véhicule léger", "vélo"]
    filtered_data = selected_df[selected_df["Type d'objet"].isin(relevant_objects)]
    
     # Sum the 'Nombre' for each type of object
    total_counts = filtered_data.groupby("Type d'objet")["Nombre"].sum()

     # Assign total counts to variables
    counts = {obj: total_counts.get(obj, 0) for obj in relevant_objects}

    return counts

def generate_summary_statistics(selected_df):
    return selected_df.describe().T


def comparaison_type_objet(selected_df):
    vehicule_comparison = (
        selected_df.groupby(["Type d'objet", "Scénario"])["Nombre"].sum().reset_index()
    )
    return vehicule_comparison


def plot_comparaison_type_objet(selected_df):
    vehicule_comparison = comparaison_type_objet(selected_df)

    fig = px.bar(
        vehicule_comparison,
        x="Scénario",
        y="Nombre",
        color="Type d'objet",
        barmode="group",
        title="Comparaison par type d'objet et scénario",
        labels={
            "Nombre": "Nombre",
            "Scénario": "Scénario",
            "Type d'objet": "Type d'objet",
        },
    )

    st.plotly_chart(fig)


def get_data_distribution(selected_df, group_by_column, value_column):
    # Group by the specified column and sum the values
    data_distribution = selected_df.groupby(group_by_column)[value_column].sum()
    return data_distribution


def plot_histogram(selected_df, group_by_column, value_column):
    data_distribution = get_data_distribution(
        selected_df, group_by_column, value_column
    )
    fig = px.histogram(
        data_distribution,
        x=data_distribution.index,
        y=data_distribution.values,
        labels={"x": group_by_column, "y": "Count"},
    )
    fig.update_layout(
        title=f"Distribution of {value_column} by {group_by_column}",
        xaxis_title=group_by_column,
        yaxis_title="Count",
    )
    st.plotly_chart(fig)


def detect_anomalies(selected_df, column, threshold):
    anomalies = selected_df[selected_df[column] > threshold]
    return anomalies  # .to_string()


def calculate_monthly_weekly_daily_patterns(selected_df, date_column, value_column):
    selected_df[date_column] = pd.to_datetime(selected_df[date_column])
    monthly_pattern = selected_df.resample("M", on=date_column)[value_column].sum()
    weekly_pattern = selected_df.resample("W", on=date_column)[value_column].sum()
    daily_pattern = selected_df.resample("D", on=date_column)[value_column].sum()
    return monthly_pattern, weekly_pattern, daily_pattern


def plot_weekly_daily_patterns(selected_df, date_column, value_column):
    monthly_pattern, weekly_pattern, daily_pattern = (
        calculate_monthly_weekly_daily_patterns(selected_df, date_column, value_column)
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Modèle  Mensuel", "Modèle Hebdomadaire", "Modèle quotidien"),
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_pattern.index, y=monthly_pattern, mode="lines", name="Mois"
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=weekly_pattern.index, y=weekly_pattern, mode="lines", name="Semaine"
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=daily_pattern.index, y=daily_pattern, mode="lines", name="Jour"),
        row=3,
        col=1,
    )

    fig.update_layout(
        title_text="Modèle Mensuel, Hebdomadaire, et quotidien",
        height=800,
        showlegend=True,
    )

    st.plotly_chart(fig)


def analysis(selected_df):
    st.markdown(
        '<h2 style="text-align: center; color: #FF4B4B;">Analyse Descriptive</h2>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Statistiques descriptives")
    st.write(generate_summary_statistics(selected_df))

    adcol1, adcol2 = st.columns(2)
    with adcol1:
        st.subheader("Comparaison par type d'objet")
        st.write(comparaison_type_objet(selected_df))
    with adcol2:
        st.subheader("Comparaison par type d'objet")
        plot_comparaison_type_objet(selected_df)

    hiscol1, histcol2 = st.columns(2)
    with hiscol1:
        st.subheader("Distribution des données")
        st.write(get_data_distribution(selected_df, "Type d'objet", "Nombre"))
    with histcol2:
        plot_histogram(selected_df, "Type d'objet", "Nombre")

    st.subheader("Détection des anomalies")
    threshold = st.number_input("Définir le seuil d'anomalie", value=10)
    st.write(detect_anomalies(selected_df, "Nombre", threshold))

    st.subheader("Tendances hebdomadaires et quotidiennes")
    monthly, weekly, daily = calculate_monthly_weekly_daily_patterns(
        selected_df, "Horodatage", "Nombre"
    )
    tendcol1, tendcol2, tendcol3 = st.columns(3)
    with tendcol1:
        st.write("Tendance hebdomadaire:", weekly)
    with tendcol2:
        st.write("Tendance quotidienne:", daily)
    with tendcol3:
        st.write("Tendance mensuel: ", monthly)

    st.subheader("Tendances mensuel, hebdomadaires et quotidiennes")
    plot_weekly_daily_patterns(selected_df, "Horodatage", "Nombre")


# def model():
#   def linear_regression():
#      # assigning columns to x and y
#     model = LinearRegression()  # create model
#    model.fit()  # Fit the model


def handle_missing_val(data):
    """
    Filter out rows where 'Nombre' is zero.

    Args:
    - data (pd.DataFrame): Input DataFrame to filter.

    Returns:
    - filtered_data (pd.DataFrame): Data with rows where 'Nombre' is zero removed.
    """
    filtered_data = data[data["Nombre"] != 0]  # Filter rows where 'Nombre' is not zero
    return filtered_data


def visualize_sunburst(symphonia_data):
    # Handle missing values
    symphonia_df = handle_missing_val(symphonia_data)
    modal = Modal(key="demo-modal", title="Donnee de la cite Symphonia")

    open_modal = st.button("Voir les donnees")
    if open_modal:
        with modal.container():
            st.write("Donnee total des cameras de surveillance")
            st.write(symphonia_df)

    # Group and aggregate the data
    grouped_data = (
        symphonia_df.groupby(["Caméra", "Scénario", "Catégorie", "Type d'objet"])
        .agg({"Nombre": "sum"})
        .reset_index()
    )

    # Prepare the data for the sunburst plot
    grouped_data["path"] = grouped_data.apply(
        lambda row: [
            row["Caméra"],
            row["Scénario"],
            row["Catégorie"],
            row["Type d'objet"],
        ],
        axis=1,
    )

    # Calculate color_continuous_midpoint safely
    total_sum = grouped_data["Nombre"].sum()
    if total_sum == 0:
        color_midpoint = 0  # or set a default midpoint
    else:
        color_midpoint = np.average(
            grouped_data["Nombre"], weights=grouped_data["Nombre"]
        )

    # Create the Sunburst chart
    fig = px.sunburst(
        grouped_data,
        path=["Caméra", "Scénario", "Catégorie", "Type d'objet"],
        values="Nombre",
        color="Nombre",  # Using 'Nombre' for color mapping
        hover_data={"Nombre": True},
        color_continuous_scale="RdBu",
        color_continuous_midpoint=color_midpoint,
    )

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        width=800,
        height=800,
        autosize=False,
    )

    # Center the plot in Streamlit
    with st.container():
        st.plotly_chart(fig)


def visualization_pie_repartition(symphonia_data):

    # Group and aggregate the data
    grouped_data = (
        symphonia_data.groupby(["Caméra", "Scénario", "Catégorie", "Type d'objet"])
        .agg({"Nombre": "sum"})
        .reset_index()
    )

    # Extract relevant statistics for interpretation

    camera_counts = grouped_data.groupby("Caméra")["Nombre"].sum()
    scenario_counts = grouped_data.groupby("Scénario")["Nombre"].sum()
    category_counts = grouped_data.groupby("Catégorie")["Nombre"].sum()

    st.markdown(
        "### Répartition des donnees en fonction des cameras, Scenario et categorie "
    )

    st.markdown(
        '<h3 style="text-align: center;">Caméra</h3>',
        unsafe_allow_html=True,
    )
    camera_data_list = []
    for camera, count in camera_counts.items():
        # st.markdown(f"- **{camera}**: {count} événements")
        camera_data_list.append({"Caméra": camera, "événements": count})
    rep_cam = pd.DataFrame(camera_data_list)

    # Creating the pie chart
    fig = px.pie(
        rep_cam,
        values="événements",
        names="Caméra",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    st.plotly_chart(fig)

    st.markdown(
        '<h3 style="text-align: center;">Scénario</h3>',
        unsafe_allow_html=True,
    )
    scenario_data_list = []
    for scenario, count in scenario_counts.items():
        # st.markdown(f"- **{scenario}**: {count} événements")
        scenario_data_list.append({"Scénario": scenario, "événements": count})
    rep_scen = pd.DataFrame(scenario_data_list)

    # Creating the pie chart
    fig = px.pie(
        rep_scen,
        values="événements",
        names="Scénario",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    st.plotly_chart(fig)

    st.markdown(
        '<h3 style="text-align: center;">Catégorie</h3>',
        unsafe_allow_html=True,
    )
    categorie_data_list = []
    for category, count in category_counts.items():
        # st.markdown(f"- **{category}**: {count} événements")
        categorie_data_list.append({"Catégorie": category, "événements": count})
    rep_cat = pd.DataFrame(categorie_data_list)

    # Creating the pie chart
    fig = px.pie(
        rep_cat,
        values="événements",
        names="Catégorie",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
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
        st.sidebar.markdown(
            '<h3 style="text-align: left; margin-top:15px">Filtre</h3>',
            unsafe_allow_html=True,
        )
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
        st.sidebar.markdown(
            '<h3 style="text-align: left; margin-top:15px">Filtre</h3>',
            unsafe_allow_html=True,
        )
        selected_types = st.sidebar.multiselect(
            "Sélectionnez le type d'objet", sorted(selected_df["Type d'objet"].unique())
        )
        selected_scenarios = st.sidebar.multiselect(
            "Sélectionnez le scénario", unique_scenario.tolist()
        )

        # Filter data based on selected filters
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
