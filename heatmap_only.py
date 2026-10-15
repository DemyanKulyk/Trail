from pathlib import Path
import folium
from folium.plugins import HeatMap
import fitdecode
import warnings

FIT_DIR = Path("./real_data_fit")
OUTPUT_DIR = Path("./cartes_sorties")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Crée le dossier s'il n'existe pas
HTML_OUT = OUTPUT_DIR / "heatmap_frequentation.html"


SEMICIRCLE = 180.0 / (2 ** 31)

# Seuil maximum de points pour garantir la fluidité du navigateur
MAX_HTML_POINTS = 50000

all_coordinates = []
processed_files_count = 0

# 1. Extraction brute de toutes les données
fit_files = list(FIT_DIR.glob("*.fit"))
if not fit_files:
    print("Aucun fichier .fit trouvé.")
    exit()

for file_path in fit_files:
    try:
        with fitdecode.FitReader(file_path) as fit:
            processed_files_count += 1
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record":
                    raw_lat = frame.get_value("position_lat") if frame.has_field("position_lat") else None
                    raw_lon = frame.get_value("position_long") if frame.has_field("position_long") else None

                    if raw_lat is not None and raw_lon is not None:
                        all_coordinates.append([raw_lat * SEMICIRCLE, raw_lon * SEMICIRCLE])
    except Exception as e:
        print(f"Erreur fichier {file_path.name} : {e}")

total_points_bruts = len(all_coordinates)
if total_points_bruts == 0:
    print("Aucune coordonnée GPS extraite.")
    exit()

# 2. Sous-échantillonnage dynamique (Optimisation)
if total_points_bruts > MAX_HTML_POINTS:
    step = (total_points_bruts // MAX_HTML_POINTS) + 1
    display_coordinates = all_coordinates[::step]
else:
    display_coordinates = all_coordinates

total_points_affiches = len(display_coordinates)

# 3. Calcul du centre
avg_lat = sum(c[0] for c in display_coordinates) / total_points_affiches
avg_lon = sum(c[1] for c in display_coordinates) / total_points_affiches

# 4. Initialisation de la carte
m = folium.Map(
    location=[avg_lat, avg_lon],
    zoom_start=15,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, HERE, Garmin, USGS, NGA"
)

# 5. Rendu HeatMap
HeatMap(
    display_coordinates,
    radius=12,
    blur=15,
    min_opacity=0.3,
    max_zoom=1
).add_to(m)

# 6. Interface de synthèse
style_commun = """
    background: white; 
    border: 2px solid #2B2D42; 
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    color: #111;
    z-index: 9999;
"""

ratio_compression = round((1 - (total_points_affiches / total_points_bruts)) * 100, 1)

info_box_html = f"""
<div style="
    position: fixed; top: 15px; right: 15px; width: 280px;
    padding: 12px 16px; font-size: 13px; line-height: 1.55;
    {style_commun}
">
    <div style="font-weight: 700; font-size: 15px; margin-bottom: 8px; border-bottom: 1px solid #E0E0E0; padding-bottom: 4px;">
        Heatmap des Passages (Optimisée)
    </div>
    <div style="margin: 3px 0;"><b>Courses analysées :</b> {processed_files_count}</div>
    <div style="margin: 3px 0;"><b>Points bruts (fichiers) :</b> {total_points_bruts}</div>
    <div style="margin: 3px 0;"><b>Points rendus (DOM) :</b> {total_points_affiches}</div>
    <div style="margin: 3px 0; color: #EF233C; font-weight: 600;">
        Compression : -{ratio_compression}%
    </div>
</div>
"""

m.get_root().html.add_child(folium.Element(info_box_html))

# 7. Export
m.save(str(HTML_OUT))
print(f"Carte enregistrée : {HTML_OUT.resolve()}") # resolve() affiche le chemin absolu complet
print(f"-> {total_points_bruts} points bruts réduits à {total_points_affiches} pour l'affichage.")


# Masquer les warnings spécifiques de fitdecode sur les champs non standard
warnings.filterwarnings(
    "ignore", category=UserWarning, module="fitdecode.reader"
)