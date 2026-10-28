from pathlib import Path
import folium
import branca.colormap as cm
import fitdecode
import warnings

FIT_FILE = Path("./real_data_fit/22318015913_ACTIVITY.fit")
OUTPUT_DIR = Path("./cartes_sorties")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Crée le dossier s'il n'existe pas
HTML_OUT = OUTPUT_DIR / "onecourse.html"


SEMICIRCLE = 180.0 / (2 ** 31)

records = []
total_ascent = None
total_descent = None

# 0. Parsing du fichier .FIT
with fitdecode.FitReader(FIT_FILE) as fit:
    for frame in fit:
        if isinstance(frame, fitdecode.FitDataMessage):

            # Récupération du D+ et D- officiel Garmin s'il est présent dans session ou lap
            if frame.name in ("session", "lap"):
                if total_ascent is None and frame.has_field("total_ascent"):
                    total_ascent = frame.get_value("total_ascent")
                if total_descent is None and frame.has_field("total_descent"):
                    total_descent = frame.get_value("total_descent")

            # Récupération des séries temporelles
            elif frame.name == "record":
                raw_lat = frame.get_value("position_lat") if frame.has_field("position_lat") else None
                raw_lon = frame.get_value("position_long") if frame.has_field("position_long") else None
                speed_ms = frame.get_value("enhanced_speed") if frame.has_field("enhanced_speed") else frame.get_value(
                    "speed") if frame.has_field("speed") else None
                altitude = frame.get_value("enhanced_altitude") if frame.has_field(
                    "enhanced_altitude") else frame.get_value("altitude") if frame.has_field("altitude") else None
                distance = frame.get_value("distance") if frame.has_field("distance") else None
                timestamp = frame.get_value("timestamp") if frame.has_field("timestamp") else None

                if raw_lat is not None and raw_lon is not None and speed_ms is not None:
                    # Plafond à 8:00 min/km pour les phases lentes / arrêts
                    if speed_ms > 0.5:
                        pace_min_km = min(1000.0 / (speed_ms * 60.0), 8.0)
                    else:
                        pace_min_km = 8.0

                    records.append({
                        "lat": raw_lat * SEMICIRCLE,
                        "lon": raw_lon * SEMICIRCLE,
                        "speed_ms": speed_ms,
                        "pace": pace_min_km,
                        "altitude": altitude,
                        "distance": distance,
                        "time": timestamp
                    })

if len(records) < 2:
    print("Données insuffisantes.")
    exit()

# 2. Calcul des métriques globales
# Si total_ascent n'est pas dans l'en-tête session, calcul direct via la série altimétrique
if total_ascent is None or total_descent is None:
    calc_ascent = 0.0
    calc_descent = 0.0
    for i in range(len(records) - 1):
        alt1 = records[i]["altitude"]
        alt2 = records[i + 1]["altitude"]
        if alt1 is not None and alt2 is not None:
            diff = alt2 - alt1
            if diff > 0:
                calc_ascent += diff
            else:
                calc_descent += abs(diff)
    total_ascent = round(calc_ascent)
    total_descent = round(calc_descent)

# Distances et durées
total_dist_m = (records[-1]["distance"] or 0) - (records[0]["distance"] or 0)
if total_dist_m <= 0:
    total_dist_m = len(records) * 1.5  # fallback d'estimation

total_seconds = 0
if records[0]["time"] and records[-1]["time"]:
    total_seconds = (records[-1]["time"] - records[0]["time"]).total_seconds()

# 1. Calcul du temps en mouvement par intégration des deltas temporels réels (tolérant au Smart Recording)
moving_seconds = 0
for i in range(len(records) - 1):
    t1 = records[i]["time"]
    t2 = records[i + 1]["time"]
    speed = records[i]["speed_ms"]

    if t1 and t2:
        dt = (t2 - t1).total_seconds()
        # Filtre les sauts aberrants (> 30s) et compte uniquement si vitesse > 1.8 km/h
        if 0 < dt < 30 and speed > 0.5:
            moving_seconds += dt

# Fallback si aucun mouvement valide n'est détecté
if moving_seconds == 0:
    moving_seconds = total_seconds

# 2. Vitesses réelles recalculées
moving_speed_kmh = (total_dist_m / 1000) / (moving_seconds / 3600) if moving_seconds > 0 else 0
total_speed_kmh = (total_dist_m / 1000) / (total_seconds / 3600) if total_seconds > 0 else 0


def sec_to_pace_str(speed_kmh: float) -> str:
    if speed_kmh <= 0.5:
        return "-:--"
    pace_dec = 60.0 / speed_kmh
    m, s = divmod(int(pace_dec * 60), 60)
    return f"{m}:{s:02d} min/km"


def format_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


# 3. Colormap pour l'allure
min_pace = max(3.5, min(r["pace"] for r in records))
max_pace = min(7.5, max(r["pace"] for r in records))

colormap = cm.LinearColormap(
    colors=["#2DC653", "#FFD166", "#EF233C"],
    vmin=min_pace,
    vmax=max_pace,
    caption="Allure (min/km) : Vert = Rapide | Rouge = Lent"
)

# 4. Carte Folium
start_loc = [records[0]["lat"], records[0]["lon"]]
m = folium.Map(
    location=start_loc,
    zoom_start=14,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, HERE, Garmin, USGS, NGA"
)

# Micro-segments colorés
for i in range(len(records) - 1):
    pt1 = records[i]
    pt2 = records[i + 1]
    avg_pace = (pt1["pace"] + pt2["pace"]) / 2.0
    m_p, s_p = divmod(int(avg_pace * 60), 60)

    folium.PolyLine(
        locations=[(pt1["lat"], pt1["lon"]), (pt2["lat"], pt2["lon"])],
        color=colormap(avg_pace),
        weight=5,
        opacity=0.9,
        tooltip=f"{m_p}:{s_p:02d} min/km"
    ).add_to(m)

# Marqueurs
folium.Marker([records[0]["lat"], records[0]["lon"]], popup="Départ",
              icon=folium.Icon(color="green", icon="play")).add_to(m)
folium.Marker([records[-1]["lat"], records[-1]["lon"]], popup="Arrivée",
              icon=folium.Icon(color="red", icon="flag")).add_to(m)
# [Garde toute la première partie de ton script de parsing et de calcul inchangée]

# ...

# 3. Échelle de l'allure (valeurs de référence pour la légende)
min_pace = max(3.5, min(r["pace"] for r in records))
max_pace = min(7.5, max(r["pace"] for r in records))

# On garde le colormap pour colorer les tracés
colormap = cm.LinearColormap(
    colors=["#2DC653", "#FFD166", "#EF233C"],
    vmin=min_pace,
    vmax=max_pace
)

# 4. Carte Folium (toujours avec Esri pour éviter le 403)
start_loc = [records[0]["lat"], records[0]["lon"]]
m = folium.Map(
    location=start_loc,
    zoom_start=16,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, HERE, Garmin, USGS, NGA"
)

# Micro-segments colorés (Garder tel quel)
for i in range(len(records) - 1):
    pt1 = records[i]
    pt2 = records[i + 1]
    avg_pace = (pt1["pace"] + pt2["pace"]) / 2.0
    m_p, s_p = divmod(int(avg_pace * 60), 60)

    folium.PolyLine(
        locations=[(pt1["lat"], pt1["lon"]), (pt2["lat"], pt2["lon"])],
        color=colormap(avg_pace),
        weight=5,
        opacity=0.9,
        tooltip=f"{m_p}:{s_p:02d} min/km"
    ).add_to(m)

# Marqueurs Départ / Arrivée
folium.Marker([records[0]["lat"], records[0]["lon"]], popup="Départ", icon=folium.Icon(color="green", icon="play")).add_to(m)
folium.Marker([records[-1]["lat"], records[-1]["lon"]], popup="Arrivée", icon=folium.Icon(color="red", icon="flag")).add_to(m)


# 5. --- STYLE COMMUN POUR LES ENCARTS ---
style_commun = """
    background: white; 
    border: 2px solid #2B2D42; 
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    color: #111;
    z-index: 9999;
"""

# 6. --- ENCART DE SYNTHÈSE DE COURSE ---
info_box_html = f"""
<div style="
    position: fixed; top: 15px; right: 15px; width: 290px;
    padding: 12px 16px; font-size: 13px; line-height: 1.55;
    {style_commun}
">
    <div style="font-weight: 700; font-size: 15px; margin-bottom: 8px; border-bottom: 1px solid #E0E0E0; padding-bottom: 4px;">
        Synthèse de Course
    </div>
    <div style="margin: 3px 0;"><b>Distance :</b> {total_dist_m / 1000:.2f} km</div>
    <div style="margin: 3px 0;">
        <b>Durée :</b> {format_time(moving_seconds)} <span style="color: #6C757D; font-size: 11px;">(total : {format_time(total_seconds)})</span>
    </div>
    <div style="margin: 3px 0;">
        <b>Vitesse :</b> {moving_speed_kmh:.1f} km/h <span style="color: #6C757D; font-size: 11px;"></span>
    </div>
    <div style="margin: 3px 0;">
        <b>Allure moy :</b> {sec_to_pace_str(moving_speed_kmh)} <span style="color: #6C757D; font-size: 11px;"></span>
    </div>
    <div style="margin: 3px 0;">
        <b>Dénivelé :</b> +{int(total_ascent)} m / -{int(total_descent)} m
    </div>
</div>
"""

# 7. --- NOUVEL ENCART : ÉCHELLE DE VITESSE / ALLURE ---
# Formatage des valeurs min et max pour l'affichage de la légende
# Formatage rigoureux MM:SS pour les trois repères de la légende
def format_pace(pace_dec: float) -> str:
    m, s = divmod(int(round(pace_dec * 60)), 60)
    return f"{m}:{s:02d}"

pace_min_str = format_pace(min_pace)
pace_mid_str = format_pace((min_pace + max_pace) / 2.0)
pace_max_str = format_pace(max_pace)

legend_html = f"""
<div style="
    position: fixed; top: 210px; right: 15px; width: 290px;
    padding: 12px 16px; font-size: 12px;
    {style_commun}
">
    <div style="font-weight: 700; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid #E0E0E0; padding-bottom: 4px;">
        Échelle d'allure (min/km)
    </div>
    <!-- Dégradé de Vert (#2DC653) à Rouge (#EF233C) -->
    <div style="
        height: 14px; 
        background: linear-gradient(to right, #2DC653 0%, #FFD166 50%, #EF233C 100%); 
        border-radius: 4px;
        margin-bottom: 6px;
    "></div>
    <div style="display: flex; justify-content: space-between; font-weight: 500; font-size: 11px;">
        <span style="color: #2DC653;">{pace_min_str} </span>
        <span style="color: #6C757D;">{pace_mid_str}</span>
        <span style="color: #EF233C;">{pace_max_str} </span>
    </div>
</div>
"""

# Injection des deux encarts dans la carte
m.get_root().html.add_child(folium.Element(info_box_html))
m.get_root().html.add_child(folium.Element(legend_html))

m.save(str(HTML_OUT))
print(f"Carte enregistrée : {HTML_OUT.resolve()}") # resolve() affiche le chemin absolu complet

# Masquer les warnings spécifiques de fitdecode sur les champs non standard
warnings.filterwarnings(
    "ignore", category=UserWarning, module="fitdecode.reader"
)