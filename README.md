# GeoLifeCLEF25

https://www.kaggle.com/competitions/geolifeclef-2025

## Ausführung und Installation

### Installation notwendiger Libraries
`pip install -r requirements.txt`

### Ausführung des Interface zur Datenvisualisierung
`streamlit run interface.py`

- Browser sollte unter http://localhost:8501/ ein Interface zur Visualisierung der Daten anzeigen

- Work in Progress, funktioniert möglicherweise noch nicht, aber ist im Pflichtenheft auch optional

### Ausführung des Jupyter Notebooks mit dem Kernel, in welchem die Requirements installiert wurden
- Jupyter Notebook `sentinel-2-data-processing-and-normalization.ipynb`: Bereitgestellt durch Veranstalter der Challenge, visualisiert Datenaufbereitungsmethoden
- Jupyter Notebook `Modelltraining.ipynb`: Eigenes Notebook zum Trainieren des Modells

## Projektstruktur:
- `research` enthält gelesene Forschungsarbeiten der Vorjahre der Challenge
- `src` enthält den Quellcode des Projekts abgesehen vom Interface
  - `dataset`: enthält den Quellcode zur Datenverarbeitung
  - `model`: enthält den Quellcode zum Modelltraining
  - `helpers.py`: Helferfunktionen
- `interface.py`: Interface zur Datenvisualisierung
- `requirements.txt`: Auflistung notwendiger Libraries
- `Modelltraining.ipynb`: Eigenes Notebook zum Trainieren des Modells
- `sentinel-2-data-processing-and-normalization.ipynb`: Bereitgestellt durch Veranstalter der Challenge, visualisiert Datenaufbereitungsmethoden

## Data

### Presence-Absence
- 100k surveys
- 10k species

### Presence-Only
- 5 Mio. observations

### Metadata (labels)
- PA in `PresenceAbsenceSurveys/GLC25_PA_metadata_train.csv`
- PO in `PresenceOnlyOccurences/GLC25_PO_metadata_train.csv`

### Environmental data

**1. Satellite Image Patches**
- 4-band (R, G, B, NIR) TIFF in `./SatellitePatches/`
- 64x64 pixels
- 10m resolution
- Access via surveyId
  - `surveyId = XXXABCD` $\rightarrow$ `…/CD/AB/XXXXABCD.jpeg`
  - `surveyId` of occurrence from CSV
  - e.g.: `surveyId = 1234567`
    - path: `./SatellitePatches/67/45/3016745.tiff`
  - or `surveyId = 1`
    - path: `./SatellitePatches/1/1/1.tiff`

**2. Satellite Time Series**
- Time Series of satellite median point values over each season since winter 1999 for each observation
- For six satellite bands (R, G, B, NIR, SWIR1, SWIR2)
- Format 1: CSV
  - 6 CSV files, one per band
  - Rows corresponding to `surveyId`
  - Columns for the 84 seasons from winter 2000 until autumn 2020
- Format 2: TimeSeries-Cubes
  - CSV aggregated into 3d tensors with axes and BAND, QUARTER, YEAR
- Access through `./SatelliteTimeSeries/`

**3. Monthly climatic rasters**
- Four climatic variables computed monthly
- Mean, min, max temp. and total precipitation
- Jan 2000 - Dec 2019
- 960 low-res rasters
- Fomat 1: CSV
  - One per raster, referenced though the `surveyId`
- Format 2: TimeSeries-Cubes
  - CSV aggregated into 3d tensors with axes and RASTER-TYPE, YEAR, MONTH
- Access: `/EnvironmentalRasters/Climate/Climatic_Monthly_2000-2019`

**4. Environmental rasters**
- Additional environmental data such as GeoTIFF rasters and scalar values extracted from them
- CSV files, one per band raster type, i.e. Climate, SoilGrids, Elevation, LandCover, HumanFootprint
1. Bioclimatic
   - 19 low-res rasters across Europe
   - 30 arcsec (about 1km) resolution
   - lon/lat coordinates (WGS84)
   - GeoTIFF with compression or CSV with extracted values
   - Access: `/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010`
2. SoilGrids
   - 9 pedologic low-res 
3. Elevation
   - 1 low-res raster across Europe
   - 30 arcsec (about 1km) resolution
   - lon/lat coordinates (WGS84)
   - GeoTIFF with compression or CSV with extracted values
   - Access: `/EnvironmentalRasters/Elevation`
4. LandCover
   - 1 low-res raster across Europe
   - 30 arcsec (about 1km) resolution