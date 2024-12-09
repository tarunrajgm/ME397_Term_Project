import pandas as pd
import requests
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.wkt import loads
import io
from folium.plugins import HeatMapWithTime
import folium
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import branca


def main():
    url = "https://data.transportation.gov/api/views/keg4-3bc2/rows.csv?accessType=DOWNLOAD"
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.text))

    # convert the 'Point' into geometry
    data['geometry'] = data['Point'].apply(loads)

    gdf = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")

    gdf.drop(columns=['Point'], inplace=True)


    gdf['Date'] = pd.to_datetime(gdf['Date'], format='%b %Y')
    gdf['month'] = gdf['Date'].dt.month
    gdf['year'] = gdf['Date'].dt.year

    gdf = gdf[gdf['State'] != 'Alaska']

    # apply filtering and create a copy for the filtered data
    filtered_data_copy = gdf.copy()

    filtered_data_copy = filtered_data_copy[
        (filtered_data_copy["Measure"].isin(["Pedestrians", "Personal Vehicles", "Trucks"])) &
        (filtered_data_copy['year'] > (max(filtered_data_copy['year']) - 8))
    ]

    # Extract the top ports by volume for each year-month
    top_ports = (
        filtered_data_copy.groupby(["year", "month", "Port Name"])["Value"]
        .sum()
        .reset_index()
        .sort_values(by=["year", "month", "Value"], ascending=[True, True, False])
        .groupby(["year", "month"])
        .head(30)
    )

    # join back with filtered data copy to restrict to top ports
    filtered_data_copy = filtered_data_copy.merge(
        top_ports[["year", "month", "Port Name"]],
        on=["year", "month", "Port Name"],
        how="inner"
    )

    # Prepare heatmap data for each measure 
    heatmap_data = {}
    for measure in ["Pedestrians", "Personal Vehicles", "Trucks"]:
        measure_data = filtered_data_copy[filtered_data_copy["Measure"] == measure]
        heatmap_data[measure] = [
            [
                [row["Latitude"], row["Longitude"], row["Value"]]
                for _, row in measure_data[(measure_data["year"] == year) & (measure_data["month"] == month)].iterrows()
            ]
            for year, month in sorted(
                measure_data[["year", "month"]].drop_duplicates().itertuples(index=False)
            )
        ]
        # labels for the slider include months
        heatmap_data[measure + "_labels"] = [
            f"{year}-{month:02d} ({measure})"
            for year, month in sorted(
                measure_data[["year", "month"]].drop_duplicates().itertuples(index=False)
            )
        ]

    # normalize 
    normalized_heatmap_data = {}
    scaler = MinMaxScaler()

    for measure, data in heatmap_data.items():
        if measure.endswith("_labels"):  
            continue
        normalized_data_slices = []
        for slice_data in data:
            if len(slice_data) > 0:  # Avoid empty slices
                # Extract intensity values, normalize them, and reattach to coordinates
                slice_array = np.array(slice_data)
                coords = slice_array[:, :2]  # Extract coordinates
                intensities = slice_array[:, 2].reshape(-1, 1)  # Extract intensities
                normalized_intensities = scaler.fit_transform(intensities).flatten()  # Normalize
                normalized_slice = np.hstack([coords, normalized_intensities[:, None]])
                normalized_data_slices.append(normalized_slice.tolist())
            else:
                normalized_data_slices.append([])

        normalized_heatmap_data[measure] = normalized_data_slices
        normalized_heatmap_data[measure + "_labels"] = heatmap_data[measure + "_labels"]

    # Calculate relative breakpoints for each measure based on monthly percentiles
    measure_breaks = {}
    for measure in ["Pedestrians", "Personal Vehicles", "Trucks"]:
        grouped_data = filtered_data_copy[filtered_data_copy["Measure"] == measure].groupby(["year", "month"])["Value"]
        monthly_breaks = []

        for (year, month), values in grouped_data:
            lower_bound = np.percentile(values, 20)
            median = np.median(values)
            upper_bound = np.percentile(values, 80)
            monthly_breaks.append([
                lower_bound,
                median,
                upper_bound,
            ])

        monthly_breaks_array = np.array(monthly_breaks)
        averaged_breaks = np.mean(monthly_breaks_array, axis=0)  
        
        # nearest 1000
        measure_breaks[measure] = [round(avg / 1000) * 1000 for avg in averaged_breaks]


    # gradient colors for each measure
    measure_gradients = {
        "Pedestrians": ["blue", "lightblue", "cyan"],
        "Personal Vehicles": ["green", "yellowgreen", "lime"],
        "Trucks": ["yellow", "orange", "red"],
    }

    m = folium.Map(location=[39.5, -98.35], zoom_start=5)

    # Add heatmap layers for each measure
    for idx, (measure, data) in enumerate(normalized_heatmap_data.items()):
        if measure.endswith("_labels"):
            continue
        breaks = measure_breaks[measure]
        gradient_colors = measure_gradients[measure]
        HeatMapWithTime(
            data,
            name=f"{measure}",
            radius=70,
            auto_play=False,
            max_opacity=0.95,
            min_opacity=0.4,
            index=normalized_heatmap_data[measure + "_labels"],
            gradient={ 
                0.2: gradient_colors[0],
                0.5: gradient_colors[1],
                0.8: gradient_colors[2],
            },
        ).add_to(m)
        add_measure_legend(m, measure, breaks, gradient_colors, idx)


    folium.LayerControl().add_to(m)

    m.save("interactive_map_monthly.html")


# Function to add legends with stacked positioning in the top-left corner
def add_measure_legend(map_object, measure, breaks, colors, offset_index):
    vertical_offset = 10 + offset_index * 120  
    legend_html = f"""
    <div style="
        position: fixed;
        top: {vertical_offset}px;
        left: 10px;
        width: 200px;
        height: auto;
        background-color: white;
        border: 2px solid grey;
        border-radius: 10px;
        padding: 10px;
        z-index:9999;
        font-size:14px;">
        <strong>{measure}</strong><br>
        <i style="background:{colors[0]}; width: 12px; height: 12px; display: inline-block;"></i>
        &lt; {breaks[0]:,}<br>
        <i style="background:{colors[1]}; width: 12px; height: 12px; display: inline-block;"></i>
        {breaks[0]:,} - {breaks[1]:,}<br>
        <i style="background:{colors[2]}; width: 12px; height: 12px; display: inline-block;"></i>
        &gt; {breaks[1]:,}<br>
    </div>
    """
    map_object.get_root().html.add_child(branca.element.Element(legend_html))


if __name__ == "__main__":
    main()