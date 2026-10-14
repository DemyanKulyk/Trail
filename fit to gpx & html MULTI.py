import math
from pathlib import Path
import fitdecode
import folium
from branca.colormap import LinearColormap

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "data_fit_generated"
OUTPUT_MAP_PATH = SCRIPT_DIR / "heatmap_courses.html"

SEMICIRCLE_TO_DEGREES = 180.0 / (2**31)


def haversine(coord1, coord2):
  """Distance en mètres entre deux coordonnées (lat, lon)."""
  r = 6371000
  phi1, phi2 = math.radians(coord1[0]), math.radians(coord2[0])
  dphi = math.radians(coord2[0] - coord1[0])
  dlambda = math.radians(coord2[1] - coord1[1])

  a = (
      math.sin(dphi / 2) ** 2
      + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
  )
  return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def format_seconds(sec: float) -> str:
  m, s = divmod(int(sec), 60)
  h, m = divmod(m, 60)
  return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


all_tracks = []
all_points = []
total_distance_m = 0.0
total_duration_s = 0.0
all_speeds = []

fit_files = list(INPUT_DIR.glob("*.fit"))

if not fit_files:
    print(f"Aucun fichier .fit trouvé dans {INPUT_DIR.resolve()}")
    exit()

for fit_path in fit_files:

    coords = []
    timestamps = []

    with fitdecode.FitReader(str(fit_path)) as fit:

        for frame in fit:

            if (
                isinstance(frame, fitdecode.FitDataMessage)
                and frame.name == "record"
            ):

                if (
                    frame.has_field("position_lat")
                    and frame.has_field("position_long")
                ):

                    raw_lat = frame.get_value("position_lat")
                    raw_lon = frame.get_value("position_long")

                    if raw_lat is not None and raw_lon is not None:

                        lat = raw_lat * SEMICIRCLE_TO_DEGREES
                        lon = raw_lon * SEMICIRCLE_TO_DEGREES

                        coords.append((lat, lon))

                        if frame.has_field("timestamp"):
                            t = frame.get_value("timestamp")
                            timestamps.append(t)
                        else:
                            timestamps.append(None)

    if len(coords) >= 2:

        speeds = []

        for i in range(len(coords) - 1):

            distance = haversine(
                coords[i],
                coords[i + 1]
            )

            total_distance_m += distance

            t1 = timestamps[i]
            t2 = timestamps[i + 1]

            if t1 is not None and t2 is not None:

                dt = (t2 - t1).total_seconds()

                if dt > 0:

                    speed_mps = distance / dt
                    speed_kmh = speed_mps * 3.6

                else:
                    speed_kmh = 0

            else:
                speed_kmh = 0

            speeds.append(speed_kmh)

        all_tracks.append({
            "coords": coords,
            "speeds": speeds,
            "filename": fit_path.name
        })

        all_points.extend(coords)
        all_speeds.extend(speeds)

        if timestamps[0] is not None and timestamps[-1] is not None:

            file_duration = (
                timestamps[-1] - timestamps[0]
            ).total_seconds()

            total_duration_s += max(0, file_duration)

if not all_points:
  print("Erreur : Aucun point GPS valide trouvé dans les fichiers .fit.")
  exit()

# Fond Satellite Esri World Imagery (contraste maximal pour tracés lumineux, sans API)
m = folium.Map(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, Maxar, Earthstar Geographics, CNES/Airbus DS",
)

# TECHNIQUE HEATMAP VECTORIELLE (DOUBLE COUCHE) :
# Passe 1 : Halo diffus épais. S'accumule là où il y a du passage pour créer une zone incandescente
##for track in all_tracks:
  ##folium.PolyLine(
      ##locations=track,
      ##color="#FF3300",  # Rouge-orangé chaud
      ##weight=8,
      ##opacity=0.18,
  ##).add_to(m)
# --------------------------------------------------
# Speed color scale
# --------------------------------------------------

min_speed = 0
max_speed = 25
display_speed = min(speed, 25)
colormap = LinearColormap(
    colors=[
        "blue",
        "cyan",
        "lime",
        "yellow",
        "orange",
        "red"
    ],
    vmin=0,
    vmax=25,
    caption="Vitesse (km/h)"
)

colormap.add_to(m)


# --------------------------------------------------
# Draw every segment according to its speed
# --------------------------------------------------

for track in all_tracks:

    coords = track["coords"]
    speeds = track["speeds"]

    for i, speed in enumerate(speeds):

        folium.PolyLine(
            locations=[
                coords[i],
                coords[i + 1]
            ],
            color=colormap(min(25,speed)),
            weight=5,
            opacity=0.85,
            tooltip=f"{speed:.1f} km/h"
        ).add_to(m)

# Passe 2 : Noyau fin haute intensité
for track in all_tracks:
  folium.PolyLine(
      locations=track,
      color="#FFFF33",  # Jaune néon au centre
      weight=2,
      opacity=0.45,
  ).add_to(m)

# Recadrage automatique précis
lats = [p[0] for p in all_points]
lons = [p[1] for p in all_points]
m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

# Métriques globales
cumulated_km = total_distance_m / 1000
avg_speed = (
    (cumulated_km / (total_duration_s / 3600)) if total_duration_s > 0 else 0
)

info_box_html = f"""
<div style="
    position: fixed; 
    top: 20px; right: 20px; width: 280px; 
    background-color: rgba(18, 18, 20, 0.90); backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.15); 
    border-radius: 10px; z-index: 9999; padding: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px; line-height: 1.5; color: #ECEFF1;
">
    <div style="font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.12); padding-bottom: 6px; color: #FF5722;">
        Session Pilote • Heatmap
    </div>
    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
        <span style="color: #90A4AE;">Courses :</span> <b>{len(all_tracks)}</b>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
        <span style="color: #90A4AE;">Distance :</span> <b>{cumulated_km:.2f} km</b>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
        <span style="color: #90A4AE;">Temps cumulé :</span> <b>{format_seconds(total_duration_s)}</b>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
        <span style="color: #90A4AE;">Allure moy :</span> <b>{avg_speed:.1f} km/h</b>
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(info_box_html))

m.save(OUTPUT_MAP_PATH)
print(f"Heatmap stylisée générée avec succès : {OUTPUT_MAP_PATH}")