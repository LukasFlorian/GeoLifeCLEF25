import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import rasterio
import streamlit as st

warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="GeoLifeCLEF25 Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .map-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .satellite-container {
        border: 2px solid #2E86AB;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Satellite image processing functions
def quantile_normalize(band, low=2, high=98):
    """Normalize the band based on quantiles and rescale to [0, 255]."""
    if band.size == 0:
        return band

    band_clean = band[np.isfinite(band)]
    if band_clean.size == 0:
        return np.zeros_like(band, dtype=np.uint8)

    low_val = np.percentile(band_clean, low)
    high_val = np.percentile(band_clean, high)

    if high_val == low_val:
        return np.full_like(band, 128, dtype=np.uint8)

    normalized_band = np.clip(band, low_val, high_val)
    normalized_band = (normalized_band - low_val) / (high_val - low_val)

    return np.clip(normalized_band * 255, 0, 255).astype(np.uint8)


def get_image_path(
    survey_id, base_path="../../geolifeclef-2025/SatelitePatches/PA-train"
):
    """Generate image path based on surveyId with corrected folder structure."""
    survey_str = str(survey_id)
    padded_survey = survey_str.zfill(4)
    last_4 = padded_survey[-4:]
    cd = last_4[-2:]
    ab = last_4[-4:-2]
    return os.path.join(base_path, cd, ab, f"{survey_id}.tiff")


def calculate_comprehensive_spectral_indices(bands, use_processed=False):
    """Calculate ALL spectral indices including SR and SAVI with proper visualization ranges."""
    if bands is None or bands.shape[0] < 4:
        return None

    if use_processed:
        red = bands[0].astype(np.float32) / 255.0
        green = bands[1].astype(np.float32) / 255.0
        blue = bands[2].astype(np.float32) / 255.0
        nir = bands[3].astype(np.float32) / 255.0
    else:
        red = bands[0].astype(np.float32)
        green = bands[1].astype(np.float32)
        blue = bands[2].astype(np.float32)
        nir = bands[3].astype(np.float32)

    indices = {}

    # NDVI (Normalized Difference Vegetation Index)
    ndvi_denom = nir + red
    indices["NDVI"] = np.where(ndvi_denom != 0, (nir - red) / ndvi_denom, 0)

    # NDWI (Normalized Difference Water Index)
    ndwi_denom = green + nir
    indices["NDWI"] = np.where(ndwi_denom != 0, (green - nir) / ndwi_denom, 0)

    # EVI (Enhanced Vegetation Index)
    evi_denom = nir + 6 * red - 7.5 * blue + 1
    indices["EVI"] = np.where(evi_denom != 0, 2.5 * (nir - red) / evi_denom, 0)

    # SAVI (Soil Adjusted Vegetation Index)
    L = 0.5
    savi_denom = nir + red + L
    indices["SAVI"] = np.where(savi_denom != 0, (1 + L) * (nir - red) / savi_denom, 0)

    # Simple Ratio (SR)
    indices["SR"] = np.where(red != 0, nir / red, 0)

    # GNDVI (Green Normalized Difference Vegetation Index)
    gndvi_denom = nir + green
    indices["GNDVI"] = np.where(gndvi_denom != 0, (nir - green) / gndvi_denom, 0)

    # ARVI (Atmospherically Resistant Vegetation Index)
    arvi_denom = nir + red - blue
    indices["ARVI"] = np.where(arvi_denom != 0, (nir - red + blue) / arvi_denom, 0)

    # MSAVI (Modified Soil Adjusted Vegetation Index)
    msavi_term = (2 * nir + 1) - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))
    indices["MSAVI"] = msavi_term / 2

    return indices


@st.cache_data
def load_satellite_image(
    survey_id, base_path="../../geolifeclef-2025/SatelitePatches/PA-train"
):
    """Load satellite image and return both raw and processed versions."""
    image_path = get_image_path(survey_id, base_path)

    try:
        if not os.path.exists(image_path):
            return None, None, None

        with rasterio.open(image_path) as src:
            raw_bands = src.read()

            processed_bands = np.zeros_like(raw_bands, dtype=np.uint8)
            for i in range(raw_bands.shape[0]):
                processed_bands[i] = quantile_normalize(raw_bands[i])

            metadata = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "file_path": image_path,
            }

            return raw_bands, processed_bands, metadata

    except Exception as e:
        st.error(f"Error loading satellite image {survey_id}: {e}")
        return None, None, None


@st.cache_data
def find_available_survey_ids_for_species(po_df, pa_df, species_id, max_samples=10):
    """Find survey IDs that have both species observations and satellite images."""

    # Get survey IDs from PA data for the species
    species_surveys = pa_df.filter(pl.col("speciesId") == species_id)

    if species_surveys.shape[0] == 0:
        return []

    # Get survey IDs and check which ones have satellite images
    survey_ids = species_surveys["surveyId"].unique().to_list()

    available_surveys = []

    for survey_id in survey_ids[: max_samples * 3]:  # Check more than needed
        base_path = "../../geolifeclef-2025/SatelitePatches/PA-train"

        image_path = get_image_path(survey_id, base_path)

        if os.path.exists(image_path):
            available_surveys.append(survey_id)

        if len(available_surveys) >= max_samples:
            break

    return available_surveys


def create_satellite_rgb_composite(processed_bands):
    """Create RGB composite from processed bands."""
    if processed_bands is None or processed_bands.shape[0] < 3:
        return None

    rgb = np.dstack([processed_bands[0], processed_bands[1], processed_bands[2]])
    return rgb


def create_false_color_composite(processed_bands):
    """Create false color composite (NIR-Red-Green)."""
    if processed_bands is None or processed_bands.shape[0] < 4:
        return None

    false_color = np.dstack(
        [processed_bands[3], processed_bands[0], processed_bands[1]]
    )
    return false_color


def create_spectral_band_plots(processed_bands, survey_id):
    """Create individual spectral band visualizations using Plotly."""

    if processed_bands is None:
        return None

    band_names = ["Red", "Green", "Blue", "NIR"]
    band_colors = ["Reds", "Greens", "Blues", "Plasma"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=band_names,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    for i, (name, color) in enumerate(zip(band_names, band_colors)):
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Heatmap(
                z=processed_bands[i],
                colorscale=color,
                showscale=False,
                hovertemplate=f"{name} Band<br>Value: %{{z}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Spectral Bands - Survey ID: {survey_id}",
        height=(64 * 10 * 2),
        width=(64 * 10 * 2),
    )

    return fig


def create_spectral_indices_plots(indices, survey_id):
    """Create spectral indices visualizations using Plotly."""

    if not indices:
        return None

    # Select key indices for visualization
    key_indices = ["NDVI", "NDWI", "SAVI", "SR"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{idx} (Mean: {np.mean(indices[idx]):.3f})" for idx in key_indices
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    colormaps = {"NDVI": "RdYlGn", "NDWI": "Blues", "SAVI": "Viridis", "SR": "Plasma"}

    value_ranges = {
        "NDVI": (-1, 1),
        "NDWI": (-1, 1),
        "SAVI": (-1.5, 1.5),
        "SR": (0, 10),
    }

    for i, idx in enumerate(key_indices):
        row = i // 2 + 1
        col = i % 2 + 1

        vmin, vmax = value_ranges[idx]

        fig.add_trace(
            go.Heatmap(
                z=np.clip(indices[idx], vmin, vmax),
                colorscale=colormaps[idx],
                zmin=vmin,
                zmax=vmax,
                showscale=False,
                hovertemplate=f"{idx}<br>Value: %{{z:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Spectral Indices Analysis - Survey ID: {survey_id}",
        height=(64 * 10 * 2),
        width=(64 * 10 * 2),
    )

    return fig


def create_spectral_statistics_chart(indices):
    """Create statistics chart for all spectral indices."""

    stats_data = []

    for idx_name, idx_data in indices.items():
        stats_data.append(
            {
                "Index": idx_name,
                "Mean": np.mean(idx_data),
                "Std": np.std(idx_data),
                "Min": np.min(idx_data),
                "Max": np.max(idx_data),
            }
        )

    df_stats = pd.DataFrame(stats_data)

    fig = go.Figure(
        data=[
            go.Bar(name="Mean", x=df_stats["Index"], y=df_stats["Mean"]),
            go.Bar(name="Std", x=df_stats["Index"], y=df_stats["Std"]),
        ]
    )

    fig.update_layout(
        title="Spectral Indices Statistics Comparison",
        xaxis_title="Spectral Index",
        yaxis_title="Value",
        barmode="group",
        height=500,
    )

    return fig, df_stats


def analyze_vegetation_health(indices):
    """Analyze vegetation health based on spectral indices."""

    savi_mean = np.mean(indices["SAVI"])
    sr_mean = np.mean(indices["SR"])
    ndvi_mean = np.mean(indices["NDVI"])

    # SAVI assessment
    if savi_mean > 0.4:
        savi_health = "🟢 Dense/Healthy vegetation"
    elif savi_mean > 0.2:
        savi_health = "🟡 Moderate vegetation"
    elif savi_mean > 0.1:
        savi_health = "🟠 Sparse vegetation"
    else:
        savi_health = "🔴 Little/No vegetation"

    # SR assessment
    if sr_mean > 4:
        sr_health = "🟢 Healthy/Dense vegetation"
    elif sr_mean > 2:
        sr_health = "🟡 Moderate vegetation"
    elif sr_mean > 1:
        sr_health = "🟠 Stressed vegetation"
    else:
        sr_health = "🔴 Non-vegetated/Water/Bare soil"

    # NDVI assessment
    if ndvi_mean > 0.5:
        ndvi_health = "🟢 Dense vegetation"
    elif ndvi_mean > 0.3:
        ndvi_health = "🟡 Moderate vegetation"
    elif ndvi_mean > 0.1:
        ndvi_health = "🟠 Sparse vegetation"
    else:
        ndvi_health = "🔴 Little/No vegetation"

    return {
        "SAVI": savi_health,
        "SR": sr_health,
        "NDVI": ndvi_health,
        "values": {"SAVI": savi_mean, "SR": sr_mean, "NDVI": ndvi_mean},
    }


def load_data():
    """Load and cache the dataset"""
    try:
        with st.spinner("Loading GeoLifeCLEF25 datasets..."):
            possible_paths = [
                "GLC25_PO_metadata_train.csv",
                "../../geolifeclef-2025/GLC25_P0_metadata_train.csv",
                "./data/GLC25_PO_metadata_train.csv",
            ]

            po_df = None
            pa_df = None

            for path in possible_paths:
                try:
                    po_df = pl.read_csv(path)
                    st.success(f"✅ PO data loaded from: {path}")
                    break
                except:
                    continue

            pa_paths = [p.replace("P0", "PA") for p in possible_paths]
            for path in pa_paths:
                try:
                    pa_df = pl.read_csv(path)
                    st.success(f"✅ PA data loaded from: {path}")
                    break
                except:
                    continue

            if po_df is None or pa_df is None:
                st.error(
                    "❌ Could not load data files. Please ensure the CSV files are available."
                )
                return None, None

            return po_df, pa_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_data
def prepare_sample_data(po_df, pa_df, sample_size_po=10000, sample_size_pa=5000):
    """Prepare sample data for visualization"""
    sample_size_po = min(sample_size_po, po_df.shape[0])
    sample_size_pa = min(sample_size_pa, pa_df.shape[0])

    po_sample = po_df.sample(n=sample_size_po, seed=42)
    pa_sample = pa_df.sample(n=sample_size_pa, seed=42)

    return po_sample, pa_sample


def create_overview_metrics(po_df, pa_df):
    """Create overview metrics cards"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🔬 Presence-Only Records",
            value=f"{po_df.shape[0]:,}",
            help="Total number of presence-only observations",
        )

    with col2:
        st.metric(
            label="📊 Presence-Absence Surveys",
            value=f"{pa_df.shape[0]:,}",
            help="Total number of presence-absence surveys",
        )

    with col3:
        st.metric(
            label="🦋 Unique Species",
            value=f"{po_df['speciesId'].n_unique():,}",
            help="Number of unique species in PO data",
        )

    with col4:
        total_records = po_df.shape[0] + pa_df.shape[0]
        st.metric(
            label="📈 Total Records",
            value=f"{total_records:,}",
            help="Combined PO and PA observations",
        )


def create_data_distribution_chart(po_df, pa_df):
    """Create data type distribution chart"""
    data = {
        "Data Type": ["Presence-Only (PO)", "Presence-Absence (PA)"],
        "Count": [po_df.shape[0], pa_df.shape[0]],
        "Color": ["#2E86AB", "#A23B72"],
    }

    fig = px.bar(
        x=data["Data Type"],
        y=data["Count"],
        color=data["Data Type"],
        color_discrete_map={
            "Presence-Only (PO)": "#2E86AB",
            "Presence-Absence (PA)": "#A23B72",
        },
        title="📊 Training Data Distribution",
        labels={"x": "Data Type", "y": "Number of Observations"},
    )

    fig.update_layout(showlegend=False, height=400, title_font_size=16, title_x=0.5)

    # Add value labels on bars
    fig.update_traces(
        texttemplate="%{y:,}",
        textposition="outside",
        textfont_size=12,
        textfont_color="black",
    )

    return fig


def create_geographic_distribution(po_sample, pa_sample):
    """Create interactive geographic distribution map"""
    fig = go.Figure()

    # Add PO data points
    fig.add_trace(
        go.Scattermapbox(
            lat=po_sample["lat"],
            lon=po_sample["lon"],
            mode="markers",
            marker=dict(size=4, color="#2E86AB", opacity=0.6),
            name="Presence-Only Data",
            text=[
                f"Species: {s}<br>Year: {y}<br>Publisher: {p}"
                for s, y, p in zip(
                    po_sample["speciesId"],
                    po_sample["year"],
                    po_sample["publisher"],
                )
            ],
            hovertemplate="<b>Presence-Only</b><br>%{text}<extra></extra>",
        )
    )

    # Add PA data points
    fig.add_trace(
        go.Scattermapbox(
            lat=pa_sample["lat"],
            lon=pa_sample["lon"],
            mode="markers",
            marker=dict(size=8, color="#A23B72", opacity=0.8),
            name="Presence-Absence Data",
            text=[
                f"Country: {c}<br>Year: {y}"
                for c, y in zip(pa_sample["country"], pa_sample["year"])
            ],
            hovertemplate="<b>Presence-Absence</b><br>%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="🗺️ Geographic Coverage Across Europe",
        mapbox=dict(style="open-street-map", center=dict(lat=50, lon=10), zoom=3),
        width=600,
        height=600,
        title_font_size=16,
        title_x=0.5,
    )

    return fig


def create_temporal_analysis(po_sample):
    """Create temporal distribution charts"""
    # Yearly distribution
    yearly_counts = po_sample.group_by("year").len().sort("year")

    fig_yearly = px.line(
        x=yearly_counts["year"],
        y=yearly_counts["len"],
        title="📅 Temporal Distribution of Observations",
        labels={"x": "Year", "y": "Number of Observations"},
        markers=True,
        line_shape="spline",
    )

    fig_yearly.update_traces(
        line_color="#2E86AB", marker_color="#2E86AB", marker_size=6
    )

    fig_yearly.update_layout(
        height=400, title_font_size=16, title_x=0.5, showlegend=False
    )

    return fig_yearly


def create_seasonal_analysis(po_sample):
    """Create seasonal distribution chart"""
    monthly_counts = po_sample.group_by("month").len().sort("month")

    # Create complete month range
    month_dict = dict(
        zip(monthly_counts["month"].to_list(), monthly_counts["len"].to_list())
    )
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_values = [month_dict.get(i, 0) for i in range(1, 13)]

    fig = px.bar(
        x=months,
        y=month_values,
        title="🌸 Seasonal Distribution of Observations",
        labels={"x": "Month", "y": "Number of Observations"},
        color=month_values,
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        height=400,
        title_font_size=16,
        title_x=0.5,
        showlegend=False,
        coloraxis_showscale=False,
    )

    return fig


def create_publisher_analysis(po_sample):
    """Create publisher distribution chart"""
    publisher_counts = (
        po_sample.group_by("publisher").len().sort("len", descending=True)
    )

    fig = px.pie(
        values=publisher_counts["len"],
        names=publisher_counts["publisher"],
        title="📚 Data Publishers Distribution",
        color_discrete_sequence=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Records: %{value:,}<br>Percentage: %{percent}<extra></extra>",
    )

    fig.update_layout(height=500, title_font_size=16, title_x=0.5)

    return fig


def create_country_analysis(pa_sample):
    """Create country distribution chart"""
    country_counts = (
        pa_sample.group_by("country").len().sort("len", descending=True).head(15)
    )

    fig = px.bar(
        x=country_counts["len"],
        y=country_counts["country"],
        orientation="h",
        title="🌍 PA Data Distribution by Country (Top 15)",
        labels={"x": "Number of Surveys", "y": "Country"},
        color=country_counts["len"],
        color_continuous_scale="Plasma",
    )

    fig.update_layout(
        height=600,
        title_font_size=16,
        title_x=0.5,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )

    return fig


def create_uncertainty_analysis(po_sample, pa_sample):
    """Create geographic uncertainty analysis"""
    fig = go.Figure()

    # Filter out null values
    po_uncertainty = po_sample.filter(pl.col("geoUncertaintyInM").is_not_null())[
        "geoUncertaintyInM"
    ]
    pa_uncertainty = pa_sample.filter(pl.col("geoUncertaintyInM").is_not_null())[
        "geoUncertaintyInM"
    ]

    if po_uncertainty:
        fig.add_trace(
            go.Histogram(
                x=po_uncertainty,
                name="PO Data",
                opacity=0.7,
                marker_color="#2E86AB",
                nbinsx=50,
            )
        )

    if pa_uncertainty:
        fig.add_trace(
            go.Histogram(
                x=pa_uncertainty,
                name="PA Data",
                opacity=0.7,
                marker_color="#A23B72",
                nbinsx=50,
            )
        )

    fig.update_layout(
        title="📍 Geographic Uncertainty Distribution",
        xaxis_title="Geographic Uncertainty (meters)",
        yaxis_title="Frequency",
        xaxis_type="log",
        barmode="overlay",
        height=400,
        title_font_size=16,
        title_x=0.5,
    )

    return fig


def create_species_richness_analysis(po_sample):
    """Create species richness analysis"""
    species_counts = po_sample.group_by("speciesId").len().sort("len", descending=True)
    observations_per_species = species_counts["len"].to_list()

    fig = px.histogram(
        x=observations_per_species,
        nbins=50,
        title=f"🦋 Species Observation Distribution ({len(observations_per_species):,} unique species)",
        labels={"x": "Observations per Species", "y": "Number of Species"},
        color_discrete_sequence=["#F18F01"],
    )

    fig.update_layout(
        yaxis_type="log", height=400, title_font_size=16, title_x=0.5, showlegend=False
    )

    return fig


def create_top_species_table(po_df):
    """Create top species table"""
    top_species = (
        po_df.group_by("speciesId").len().sort("len", descending=True).head(10)
    )

    # Convert to pandas for better display
    df_display = pd.DataFrame(
        {
            "Rank": range(1, 11),
            "Species ID": top_species["speciesId"].to_list(),
            "Observations": top_species["len"].to_list(),
        }
    )

    return df_display


def main():
    """Main Streamlit application"""

    # Header
    st.markdown(
        '<h1 class="main-header">🌿 GeoLifeCLEF25 Explorer</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Location-based Species Presence Prediction @ CVPR & LifeCLEF</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("🎛️ Controls")
    st.sidebar.markdown("---")

    # Load data
    po_df, pa_df = load_data()

    if po_df is None or pa_df is None:
        st.stop()

    # Navigation
    page = st.sidebar.selectbox(
        "📋 Select Analysis Type",
        ["📊 Dataset Overview", "🛰️ Satellite Analysis", "🔬 Species Deep Dive"],
    )

    if page == "📊 Dataset Overview":
        # Original dataset overview functionality
        st.sidebar.subheader("📊 Sampling Controls")
        sample_size_po = st.sidebar.slider(
            "PO Sample Size",
            min_value=1000,
            max_value=min(50000, po_df.shape[0]),
            value=min(10000, po_df.shape[0]),
            step=1000,
        )

        sample_size_pa = st.sidebar.slider(
            "PA Sample Size",
            min_value=500,
            max_value=min(10000, pa_df.shape[0]),
            value=min(5000, pa_df.shape[0]),
            step=500,
        )

        po_sample, pa_sample = prepare_sample_data(
            po_df, pa_df, sample_size_po, sample_size_pa
        )

        # Overview metrics
        st.markdown("## 📊 Dataset Overview")
        create_overview_metrics(po_df, pa_df)

        st.markdown("---")

        # Main visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_data_distribution_chart(po_df, pa_df), use_container_width=True
            )

        with col2:
            st.plotly_chart(
                create_temporal_analysis(po_sample), use_container_width=True
            )

        # Geographic distribution
        st.markdown("## 🗺️ Geographic Distribution")
        st.plotly_chart(
            create_geographic_distribution(po_sample, pa_sample),
            use_container_width=True,
        )

        # Temporal and seasonal analysis
        st.markdown("## 📅 Temporal Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_seasonal_analysis(po_sample), use_container_width=True
            )

        # with col2:
        #     st.plotly_chart(
        #         create_uncertainty_analysis(po_sample, pa_sample),
        #         use_container_width=True,
        #     )

        # Publisher and country analysis
        st.markdown("## 📚 Data Sources Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_publisher_analysis(po_sample), use_container_width=True
            )

        with col2:
            st.plotly_chart(
                create_country_analysis(pa_sample), use_container_width=True
            )

        # Species analysis
        st.markdown("## 🦋 Species Analysis")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(
                create_species_richness_analysis(po_sample), use_container_width=True
            )

        with col2:
            st.markdown("### 🏆 Top 10 Most Observed Species")
            top_species_df = create_top_species_table(po_df)
            st.dataframe(top_species_df, use_container_width=True)

        # Data quality section
        st.markdown("## 🔍 Data Quality Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            po_null_geo = po_df.filter(
                (pl.col("lat").is_null()) | (pl.col("lon").is_null())
            ).shape[0]
            po_null_pct = po_null_geo / po_df.shape[0] * 100
            st.metric(
                "Missing Coordinates (PO)", f"{po_null_geo:,}", f"{po_null_pct:.2f}%"
            )

        with col2:
            pa_null_geo = pa_df.filter(
                (pl.col("lat").is_null()) | (pl.col("lon").is_null())
            ).shape[0]
            pa_null_pct = pa_null_geo / pa_df.shape[0] * 100
            st.metric(
                "Missing Coordinates (PA)", f"{pa_null_geo:,}", f"{pa_null_pct:.2f}%"
            )

        with col3:
            year_range_po = f"{po_df['year'].min()} - {po_df['year'].max()}"
            st.metric(
                "Date Range (PO)",
                year_range_po,
                f"{po_df['year'].max() - po_df['year'].min()} years",
            )

        # Detailed statistics
        with st.expander("📈 Detailed Statistics"):
            st.markdown("### Presence-Only (PO) Data")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Total Observations:** {po_df.shape[0]:,}")
                st.write(f"**Unique Species:** {po_df['speciesId'].n_unique():,}")
                st.write(
                    f"**Date Range:** {po_df['year'].min()} - {po_df['year'].max()}"
                )
                st.write(
                    f"**Publishers:** {', '.join(po_df['publisher'].unique().to_list())}"
                )

            with col2:
                st.write(
                    f"**Latitude Range:** {po_df['lat'].min():.2f}° to {po_df['lat'].max():.2f}°"
                )
                st.write(
                    f"**Longitude Range:** {po_df['lon'].min():.2f}° to {po_df['lon'].max():.2f}°"
                )

            st.markdown("### Presence-Absence (PA) Data")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Total Surveys:** {pa_df.shape[0]:,}")
                st.write(
                    f"**Date Range:** {pa_df['year'].min()} - {pa_df['year'].max()}"
                )
                st.write(
                    f"**Countries:** {', '.join(pa_df['country'].unique().to_list())}"
                )

            with col2:
                st.write(
                    f"**Latitude Range:** {pa_df['lat'].min():.2f}° to {pa_df['lat'].max():.2f}°"
                )
                st.write(
                    f"**Longitude Range:** {pa_df['lon'].min():.2f}° to {pa_df['lon'].max():.2f}°"
                )

    elif page == "🛰️ Satellite Analysis":
        st.markdown("## 🛰️ Satellite Patch Analysis")

        # Satellite analysis controls
        st.sidebar.subheader("🛰️ Satellite Controls")

        # Species selection
        available_species = (
            po_df["speciesId"].unique().sample(n=50, seed=0).to_list()
        )  # Limit for performance

        selected_species = st.sidebar.selectbox(
            "🦋 Select Species ID",
            available_species,
            help="Choose a species to analyze satellite patches",
        )

        max_patches = st.sidebar.slider(
            "Maximum Patches to Analyze",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of satellite patches to load and analyze",
        )

        if st.sidebar.button("🚀 Analyze Satellite Patches"):
            # Find available survey IDs for the selected species
            with st.spinner(
                f"Finding satellite patches for species {selected_species}..."
            ):
                available_surveys = find_available_survey_ids_for_species(
                    po_df, pa_df, selected_species, max_patches
                )

            if not available_surveys:
                st.error(
                    f"❌ No satellite patches found for species {selected_species}"
                )
                st.info(
                    "Try a different species or check if satellite data is available."
                )
                return

            st.success(
                f"✅ Found {len(available_surveys)} satellite patches for species {selected_species}"
            )

            # Process each satellite patch
            for i, survey_id in enumerate(available_surveys):
                st.markdown(f"### 🛰️ Satellite Patch {i + 1}: Survey ID {survey_id}")

                # Load satellite image
                with st.spinner(f"Loading satellite image {survey_id}..."):
                    raw_bands, processed_bands, metadata = load_satellite_image(
                        survey_id
                    )

                if processed_bands is None:
                    st.error(
                        f"❌ Could not load satellite image for survey {survey_id}"
                    )
                    continue

                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(
                    [
                        "🖼️ RGB Images",
                        "📊 Spectral Bands",
                        "🌿 Spectral Indices",
                        "📈 Statistics",
                    ]
                )

                with tab1:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🌈 True Color (RGB)**")
                        rgb_composite = create_satellite_rgb_composite(processed_bands)

                        if rgb_composite is not None:
                            fig_rgb = px.imshow(rgb_composite)
                            fig_rgb.update_layout(
                                height=64 * 10,
                                width=64 * 10,
                            )
                            st.plotly_chart(fig_rgb, use_container_width=True)

                    with col2:
                        st.markdown("**🔴 False Color (NIR-Red-Green)**")
                        false_color = create_false_color_composite(processed_bands)
                        if false_color is not None:
                            fig_false = px.imshow(false_color)
                            fig_false.update_layout(
                                height=64 * 10,
                                width=64 * 10,
                            )
                            st.plotly_chart(fig_false, use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                with tab2:
                    st.markdown("**📊 Individual Spectral Bands**")
                    bands_plot = create_spectral_band_plots(processed_bands, survey_id)
                    if bands_plot:
                        st.plotly_chart(bands_plot, use_container_width=True)

                with tab3:
                    st.markdown("**🌿 Spectral Indices Analysis**")

                    # Calculate spectral indices
                    indices = calculate_comprehensive_spectral_indices(
                        raw_bands, use_processed=False
                    )

                    if indices:
                        # Create spectral indices plot
                        indices_plot = create_spectral_indices_plots(indices, survey_id)
                        if indices_plot:
                            st.plotly_chart(indices_plot, use_container_width=True)

                        # Vegetation health analysis
                        st.markdown("**🌱 Vegetation Health Assessment**")
                        health_analysis = analyze_vegetation_health(indices)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "NDVI Assessment",
                                f"{health_analysis['values']['NDVI']:.3f}",
                                health_analysis["NDVI"],
                            )

                        with col2:
                            st.metric(
                                "SAVI Assessment",
                                f"{health_analysis['values']['SAVI']:.3f}",
                                health_analysis["SAVI"],
                            )

                        with col3:
                            st.metric(
                                "SR Assessment",
                                f"{health_analysis['values']['SR']:.3f}",
                                health_analysis["SR"],
                            )

                with tab4:
                    st.markdown("**📈 Detailed Statistics**")

                    if indices:
                        # Statistics chart
                        stats_plot, stats_df = create_spectral_statistics_chart(indices)
                        st.plotly_chart(stats_plot, use_container_width=True)

                        # Statistics table
                        st.markdown("**📋 Spectral Indices Statistics Table**")
                        st.dataframe(stats_df, use_container_width=True)

                        # Interpretation guide
                        with st.expander("📖 Interpretation Guide"):
                            st.markdown("""
                            **🌿 Spectral Indices Interpretation:**

                            - **NDVI** (-1 to +1): Higher values indicate more vegetation
                            - **NDWI** (-1 to +1): Higher values indicate more water content
                            - **SAVI** (-1.5 to +1.5): Soil-adjusted vegetation index, reduces soil background effects
                            - **SR** (0 to ∞): Simple ratio NIR/Red, healthy vegetation typically 2-8
                            - **EVI**: Enhanced vegetation index, less sensitive to atmospheric effects
                            - **GNDVI**: Green-based vegetation index, sensitive to chlorophyll
                            - **ARVI**: Resistant to atmospheric effects
                            - **MSAVI**: Modified soil-adjusted vegetation index
                            """)

    elif page == "🔬 Species Deep Dive":
        st.markdown("## 🔬 Species Deep Dive Analysis")

        # Species selection for deep dive
        available_species = po_df["speciesId"].unique().sample(50, seed=0).to_list()

        selected_species = st.selectbox(
            "🦋 Select Species for Deep Analysis", available_species
        )

        if selected_species:
            # Species information
            species_po_count = po_df.filter(
                pl.col("speciesId") == selected_species
            ).shape[0]

            species_pa_count = pa_df.filter(
                pl.col("speciesId") == selected_species
            ).shape[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("PO Observations", f"{species_po_count:,}")

            with col2:
                st.metric("PA Surveys", f"{species_pa_count:,}")

            with col3:
                satellite_surveys = find_available_survey_ids_for_species(
                    po_df, pa_df, selected_species, 20
                )
                st.metric("Available Satellite Patches", f"{len(satellite_surveys)}")

            # Geographic distribution for this species
            species_data = po_df.filter(pl.col("speciesId") == selected_species)

            fig_species_map = px.scatter_mapbox(
                species_data,
                lat="lat",
                lon="lon",
                size_max=15,
                zoom=3,
                title=f"Geographic Distribution - Species {selected_species}",
                mapbox_style="open-street-map",
                width=600,
                height=600,
            )

            st.plotly_chart(fig_species_map, use_container_width=True)


if __name__ == "__main__":
    main()