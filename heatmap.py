from pathlib import Path
from math import cos, radians, floor
from collections import defaultdict
import warnings

import folium
from folium.plugins import HeatMap
import fitdecode

# Masquer les warnings de fitdecode sur les champs non standard
warnings.filterwarnings("ignore", category=UserWarning, module="fitdecode.reader")

FIT_DIR = Path("./real_data_fit")
OUTPUT_DIR = Path("./cartes_sorties")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_OUT = OUTPUT_DIR / "heatmap.html"

SEMICIRCLE = 180.0 / (2 ** 31)
METERS_PER_DEG_LAT = 111_320

# --- Paramètres de comptage des passages ---
CELL_SIZE_M = 15          # taille d'une cellule de la grille (mètres)
MIN_GAP_S = 90            # absence minimale (secondes) avant qu'un retour compte comme un nouveau passage
WEIGHT_PERCENTILE = 95    # les cellules au-dessus de ce percentile sont affichées à intensité max
MIN_MOVE_M = 3            # un point à moins de X m du dernier point compté = personne immobile (tremblement GPS), ignoré. 0 = désactivé

# --- Paramètres de rendu continu ---
MAX_INTERP_GAP_S = 15     # on relie deux points GPS seulement s'ils sont séparés de moins de X secondes
RADIUS_FACTOR = 1.5       # rayon de lissage = RADIUS_FACTOR x taille de cellule (en mètres réels, suit le zoom)

# 1. Extraction des traces (une liste de points par fichier)
fit_files = list(FIT_DIR.glob("*.fit"))
if not fit_files:
    print("Aucun fichier .fit trouvé.")
    raise SystemExit

tracks = []
total_points_bruts = 0

for file_path in fit_files:
    track = []
    try:
        with fitdecode.FitReader(file_path) as fit:
            for frame in fit:
                if not (isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record"):
                    continue
                if not (frame.has_field("position_lat")
                        and frame.has_field("position_long")
                        and frame.has_field("timestamp")):
                    continue

                raw_lat = frame.get_value("position_lat")
                raw_lon = frame.get_value("position_long")
                ts = frame.get_value("timestamp")
                if raw_lat is None or raw_lon is None or ts is None:
                    continue

                track.append((ts, raw_lat * SEMICIRCLE, raw_lon * SEMICIRCLE))
    except Exception as e:
        print(f"Erreur fichier {file_path.name} : {e}")

    if track:
        track.sort(key=lambda p: p[0])
        tracks.append(track)
        total_points_bruts += len(track)

processed_files_count = len(tracks)
if total_points_bruts == 0:
    print("Aucune coordonnée GPS (avec horodatage) extraite.")
    raise SystemExit

# 2. Grille : cellules d'environ CELL_SIZE_M mètres
ref_lat = sum(p[1] for t in tracks for p in t) / total_points_bruts
dlat = CELL_SIZE_M / METERS_PER_DEG_LAT
dlon = dlat / cos(radians(ref_lat))  # correction de la longitude selon la latitude

# 3. Comptage des passages par cellule
def densify(track):
    """Ajoute des points intermédiaires entre deux points GPS consécutifs trop éloignés,
    pour que toutes les cellules traversées soient comptées (tracé sans trous).
    On ne relie pas deux points séparés par une longue pause (signal perdu, arrêt de l'enregistrement)."""
    step_m = CELL_SIZE_M / 2
    out = [track[0]]
    for (t1, la1, lo1), (t2, la2, lo2) in zip(track, track[1:]):
        dt = (t2 - t1).total_seconds()
        dy = (la2 - la1) * METERS_PER_DEG_LAT
        dx = (lo2 - lo1) * METERS_PER_DEG_LAT * cos(radians(ref_lat))
        dist = (dx * dx + dy * dy) ** 0.5
        if 0 < dt <= MAX_INTERP_GAP_S and dist > step_m:
            n = int(dist // step_m) + 1
            for k in range(1, n):
                f = k / n
                out.append((t1 + (t2 - t1) * f,
                            la1 + (la2 - la1) * f,
                            lo1 + (lo2 - lo1) * f))
        out.append((t2, la2, lo2))
    return out


passages = defaultdict(int)
runs_per_cell = defaultdict(set)  # cellule -> numéros des courses qui y sont passées (pour le diagnostic)

for run_id, track in enumerate(tracks):
    last_seen = {}  # cellule -> heure du dernier point reçu dans cette cellule (pour CE fichier)
    anchor = None   # dernier point compté
    for ts, lat, lon in densify(track):
        cell = (floor(lat / dlat), floor(lon / dlon))

        # Personne quasi immobile (tremblement GPS) : on ne crée pas de passage,
        # on prolonge juste la présence dans les cellules déjà visitées.
        if anchor is not None and MIN_MOVE_M > 0:
            dy = (lat - anchor[0]) * METERS_PER_DEG_LAT
            dx = (lon - anchor[1]) * METERS_PER_DEG_LAT * cos(radians(ref_lat))
            if (dx * dx + dy * dy) ** 0.5 < MIN_MOVE_M:
                if cell in last_seen:
                    last_seen[cell] = ts
                continue
        anchor = (lat, lon)

        prev = last_seen.get(cell)
        if prev is None or (ts - prev).total_seconds() > MIN_GAP_S:
            passages[cell] += 1  # nouvelle entrée après une absence suffisante
            runs_per_cell[cell].add(run_id)
        last_seen[cell] = ts     # mis à jour à chaque point : rester sur place = un seul passage

# 4. Poids normalisés (plafonnés au percentile choisi)
counts_sorted = sorted(passages.values())
idx = min(len(counts_sorted) - 1, int(len(counts_sorted) * WEIGHT_PERCENTILE / 100))
cap = max(1, counts_sorted[idx])

heat_data = []
for (i, j), n in passages.items():
    lat_c = (i + 0.5) * dlat
    lon_c = (j + 0.5) * dlon
    heat_data.append([lat_c, lon_c, min(n, cap) / cap])

total_cellules = len(heat_data)
total_passages = sum(passages.values())
max_passages = max(passages.values())

# 5. Centre de la carte
avg_lat = sum(p[0] for p in heat_data) / total_cellules
avg_lon = sum(p[1] for p in heat_data) / total_cellules

# 6. Carte
m = folium.Map(
    location=[avg_lat, avg_lon],
    zoom_start=15,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, HERE, Garmin, USGS, NGA"
)

heat = HeatMap(
    heat_data,
    radius=12,
    blur=15,
    min_opacity=0.3,
    max_zoom=1
)
heat.add_to(m)

# Le rayon de la heatmap est en pixels : on le recalcule à chaque zoom pour qu'il
# corresponde toujours à la même distance réelle (RADIUS_FACTOR x taille de cellule).
zoom_js = """
<script>
window.addEventListener('load', function () {
    var map = __MAP__;
    var layer = __HEAT__;
    function updateRadius() {
        var lat = map.getCenter().lat;
        var metersPerPixel = 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, map.getZoom());
        var r = Math.max(3, __RADIUS_M__ / metersPerPixel);
        layer.setOptions({radius: r, blur: r * 0.7});
    }
    map.on('zoomend', updateRadius);
    updateRadius();
});
</script>
""".replace("__MAP__", m.get_name()) \
   .replace("__HEAT__", heat.get_name()) \
   .replace("__RADIUS_M__", str(RADIUS_FACTOR * CELL_SIZE_M))

m.get_root().html.add_child(folium.Element(zoom_js))

# 7. Interface de synthèse
style_commun = """
    background: white;
    border: 2px solid #2B2D42;
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    color: #111;
    z-index: 9999;
"""

info_box_html = f"""
<div style="
    position: fixed; top: 15px; right: 15px; width: 290px;
    padding: 12px 16px; font-size: 13px; line-height: 1.55;
    {style_commun}
">
    <div style="font-weight: 700; font-size: 15px; margin-bottom: 8px; border-bottom: 1px solid #E0E0E0; padding-bottom: 4px;">
        Heatmap des Passages
    </div>
    <div style="margin: 3px 0;"><b>Courses analysées :</b> {processed_files_count}</div>
    <div style="margin: 3px 0;"><b>Tailles des cellules :</b> {CELL_SIZE_M}</div>
    <div style="margin: 3px 0;"><b>Max sur une cellule :</b> {max_passages}</div>
    <div style="margin: 3px 0; color: #6B6B6B;">
        Nouveau passage après {MIN_GAP_S} s d'absence
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(info_box_html))

# 8. Export
m.save(str(HTML_OUT))
print(f"Carte enregistrée : {HTML_OUT.resolve()}")
print(f"-> {total_points_bruts} points GPS -> {total_passages} passages sur {total_cellules} cellules.")
