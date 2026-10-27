from pathlib import Path
from math import cos, radians, ceil, floor
from collections import defaultdict
import warnings

import folium
from folium.plugins import HeatMap
import fitdecode

# Masquer les warnings de fitdecode sur les champs non standard
warnings.filterwarnings("ignore", category=UserWarning, module="fitdecode.reader")

FIT_DIR = Path("./real_data_fit")
OUTPUT_DIR = Path("./cartes_sorties")
HTML_OUT = OUTPUT_DIR / "heatmap_continue.html"

SEMICIRCLE = 180.0 / (2 ** 31)
METERS_PER_DEG_LAT = 111_320

# --- Principe : la chaleur est proportionnelle à la DISTANCE parcourue à chaque endroit ---
# Rester sur place = 0 m parcouru = aucune chaleur ajoutée.
# Repasser 3 fois au même endroit = 3 fois la longueur = 3 fois la chaleur.

STEP_M = 5                # un point n'est retenu que s'il est à >= X m du dernier point retenu.
                          # Élimine les arrêts et le tremblement GPS (la personne "ne bouge pas").
RENDER_STEP_M = 3         # espacement des points affichés le long du trajet (mètres)
MAX_INTERP_GAP_S = 15     # on ne relie pas deux points GPS séparés de plus de X secondes (pause, perte de signal)
RADIUS_M = 15             # rayon de lissage en mètres réels (suit le zoom)
WEIGHT_PERCENTILE = 98    # intensité maximale = ce percentile de fréquentation (plus bas = carte plus contrastée)
NORM_CELL_M = 10          # taille des cases utilisées UNIQUEMENT pour calibrer l'échelle de couleurs


def load_tracks():
    """Lit les fichiers .fit. Retourne une liste de traces [(timestamp, lat, lon), ...] et le nombre de points lus."""
    fit_files = list(FIT_DIR.glob("*.fit"))
    tracks = []
    total_points = 0

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
            total_points += len(track)

    return tracks, total_points


def dist_m(a_lat, a_lon, b_lat, b_lon, cos_lat):
    dy = (b_lat - a_lat) * METERS_PER_DEG_LAT
    dx = (b_lon - a_lon) * METERS_PER_DEG_LAT * cos_lat
    return (dx * dx + dy * dy) ** 0.5


def resample(track, cos_lat):
    """Transforme une trace en points régulièrement espacés le long du trajet réel.
    Retourne une liste de (lat, lon, distance_représentée_en_mètres)."""
    out = []
    anchor = track[0]   # dernier point retenu
    prev_ts = anchor[0]
    gap = False         # y a-t-il eu une coupure d'enregistrement depuis le dernier point retenu ?

    for p in track[1:]:
        ts, lat, lon = p
        if (ts - prev_ts).total_seconds() > MAX_INTERP_GAP_S:
            gap = True
        prev_ts = ts

        d = dist_m(anchor[1], anchor[2], lat, lon, cos_lat)
        if d < STEP_M:
            continue  # immobile ou tremblement GPS : ne rien ajouter

        if gap:
            # Coupure (pause, perte de signal) : on ne relie pas, on ajoute un seul point
            out.append((lat, lon, min(d, RENDER_STEP_M)))
        else:
            # Trajet continu : points intermédiaires, chacun pèse la longueur qu'il représente
            n = ceil(d / RENDER_STEP_M)
            for k in range(1, n + 1):
                f = k / n
                out.append((anchor[1] + (lat - anchor[1]) * f,
                            anchor[2] + (lon - anchor[2]) * f,
                            d / n))
        anchor = p
        gap = False

    return out


def build_heat_data(tracks):
    """Retourne (heat_data, stats). heat_data = [[lat, lon, poids], ...]"""
    n_points = sum(len(t) for t in tracks)
    ref_lat = sum(p[1] for t in tracks for p in t) / n_points
    cos_lat = cos(radians(ref_lat))

    points = []
    for t in tracks:
        points.extend(resample(t, cos_lat))
    total_m = sum(d for _, _, d in points)
    if not points:
        return [], {"total_km": 0, "cap": 1, "median_dt": 0}

    # Calibrage des couleurs : mètres parcourus par case de NORM_CELL_M ~ nombre de passages
    dlat = NORM_CELL_M / METERS_PER_DEG_LAT
    dlon = dlat / cos_lat
    meters_per_cell = defaultdict(float)
    for lat, lon, d in points:
        meters_per_cell[(floor(lat / dlat), floor(lon / dlon))] += d
    passes = sorted(m / NORM_CELL_M for m in meters_per_cell.values())
    idx = min(len(passes) - 1, int(len(passes) * WEIGHT_PERCENTILE / 100))
    cap = max(1.0, passes[idx])

    # Poids de chaque point + regroupement fin (3 m) pour alléger, sans effet visible
    a_lat = RENDER_STEP_M / METERS_PER_DEG_LAT
    a_lon = a_lat / cos_lat
    agg = {}
    for lat, lon, d in points:
        w = (d / NORM_CELL_M) / cap
        key = (floor(lat / a_lat), floor(lon / a_lon))
        c = agg.get(key)
        if c is None:
            agg[key] = [w, w * lat, w * lon]
        else:
            c[0] += w
            c[1] += w * lat
            c[2] += w * lon
    heat_data = [[c[1] / c[0], c[2] / c[0], c[0]] for c in agg.values()]

    intervals = sorted((t2 - t1).total_seconds()
                       for tr in tracks
                       for (t1, _, _), (t2, _, _) in zip(tr, tr[1:]))
    stats = {
        "total_km": total_m / 1000,
        "cap": cap,
        "median_dt": intervals[len(intervals) // 2] if intervals else 0,
        "n_display": len(heat_data),
    }
    return heat_data, stats


def write_map(heat_data, n_files, n_points_bruts, stats):
    avg_lat = sum(p[0] for p in heat_data) / len(heat_data)
    avg_lon = sum(p[1] for p in heat_data) / len(heat_data)

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=15,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, USGS, NGA"
    )

    # min_opacity=0 : le poids de chaque point compte vraiment (sinon un plancher fausserait l'échelle)
    heat = HeatMap(
        heat_data,
        radius=12,
        blur=15,
        min_opacity=0,
        max_zoom=1
    )
    heat.add_to(m)

    # Le rayon de la heatmap est en pixels : on le recalcule à chaque zoom pour qu'il
    # corresponde toujours à la même distance réelle (RADIUS_M).
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
   .replace("__RADIUS_M__", str(RADIUS_M))
    m.get_root().html.add_child(folium.Element(zoom_js))

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
    <div style="margin: 3px 0;"><b>Courses analysées :</b> {n_files}</div>
    <div style="margin: 3px 0;"><b>Points GPS lus :</b> {n_points_bruts}</div>
    <div style="margin: 3px 0;"><b>Points affichés :</b> {stats['n_display']}</div>
    <div style="margin: 3px 0;"><b>Distance cumulée :</b> {stats['total_km']:.0f} km</div>
    <div style="margin: 3px 0; color: #6B6B6B;">
        Rouge &asymp; {stats['cap']:.0f} passages ou plus
    </div>
</div>
"""
    m.get_root().html.add_child(folium.Element(info_box_html))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(HTML_OUT))
    print(f"Carte enregistrée : {HTML_OUT.resolve()}")


def main():
    tracks, n_points_bruts = load_tracks()
    if not tracks:
        print("Aucune coordonnée GPS (avec horodatage) extraite.")
        return

    heat_data, stats = build_heat_data(tracks)
    if not heat_data:
        print("Aucun déplacement détecté.")
        return

    write_map(heat_data, len(tracks), n_points_bruts, stats)
    print(f"-> {n_points_bruts} points GPS, {stats['total_km']:.0f} km cumulés, "
          f"{stats['n_display']} points affichés.")
    print(f"-> Couleur maximale à environ {stats['cap']:.0f} passages.")
    print(f"-> Intervalle médian entre deux points GPS : {stats['median_dt']:.0f} s")


if __name__ == "__main__":
    main()