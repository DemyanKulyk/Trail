"""
Heatmap "style Strava" à partir de fichiers .fit.

Principe : au lieu de poser un point par cellule (Leaflet.heat), on dessine les
trajets comme des LIGNES continues sur une grille raster fine (quelques mètres
par pixel), on additionne les courses (max 1 par pixel et par course), puis on
applique un flou gaussien. Le résultat est exporté sur une carte (folium ImageOverlay).

Dépendances : pip install numpy scipy pillow folium fitdecode
"""
from pathlib import Path
from math import cos, radians, pi, sqrt
import base64
import io
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
from PIL import Image
import folium
import fitdecode

# Masquer les warnings de fitdecode sur les champs non standard
warnings.filterwarnings("ignore", category=UserWarning, module="fitdecode.reader")

FIT_DIR = Path("./real_data_fit")
OUTPUT_DIR = Path("./cartes_sorties")
HTML_OUT = OUTPUT_DIR / "heatmap_v2.html"

SEMICIRCLE = 180.0 / (2 ** 31)

# --- Grille et lissage ---
PIXEL_M = 0.5  # taille d'un pixel (mètres). 2 = très net, 4-5 = plus léger
BLUR_M = 5.0  # sigma du flou gaussien (mètres). Plus grand = traits plus épais et plus fusionnés
SMOOTH_POINTS = 3  # moyenne glissante sur N points GPS avant tracé (réduit le bruit latéral). 1 = désactivé
MAX_SIDE_PX = 10000  # si la grille dépasse ce nombre de pixels de côté, PIXEL_M est augmenté automatiquement

# --- Filtrage des segments ---
MAX_GAP_S = 15  # on ne relie deux points que s'ils sont séparés de moins de X secondes
MAX_SPEED_MS = 12  # on ignore les segments plus rapides (sauts GPS aberrants), en m/s

# --- Rendu ---
PERCENTILE = 98  # les pixels au-dessus de ce percentile sont affichés à intensité max
GAMMA = 0.5  # 0.5 = racine carrée (compresse les écarts). 1 = linéaire
PALETTE = "violet"  # "violet" (proche de Strava sombre) ou "chaud"
OPACITY = 0.95

PALETTES = {
    "violet": [(0.0, (70, 60, 230)), (0.6, (125, 110, 255)), (1.0, (225, 235, 255))],
    "chaud": [(0.0, (120, 0, 200)), (0.35, (230, 30, 90)), (0.7, (255, 160, 0)), (1.0, (255, 255, 200))],
}

EARTH_R = 6378137.0


# ---------------------------------------------------------------- projection
def project(lat, lon, scale):
    x = EARTH_R * np.radians(lon) * scale
    y = EARTH_R * np.log(np.tan(pi / 4 + np.radians(lat) / 2)) * scale
    return x, y


def unproject(x, y, scale):
    lon = np.degrees(x / (EARTH_R * scale))
    lat = np.degrees(2 * np.arctan(np.exp(y / (EARTH_R * scale))) - pi / 2)
    return lat, lon


# ---------------------------------------------------------------- lecture
def load_tracks(fit_dir):
    tracks = []
    for file_path in sorted(fit_dir.glob("*.fit")):
        pts = []
        try:
            with fitdecode.FitReader(file_path) as fit:
                for frame in fit:
                    if not (isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record"):
                        continue
                    if not (frame.has_field("position_lat")
                            and frame.has_field("position_long")
                            and frame.has_field("timestamp")):
                        continue
                    la = frame.get_value("position_lat")
                    lo = frame.get_value("position_long")
                    ts = frame.get_value("timestamp")
                    if la is None or lo is None or ts is None:
                        continue
                    pts.append((ts, la * SEMICIRCLE, lo * SEMICIRCLE))
        except Exception as e:
            print(f"Erreur fichier {file_path.name} : {e}")

        if len(pts) >= 2:
            pts.sort(key=lambda p: p[0])
            t0 = pts[0][0]
            t = np.array([(p[0] - t0).total_seconds() for p in pts])
            lat = np.array([p[1] for p in pts])
            lon = np.array([p[2] for p in pts])
            tracks.append((t, lat, lon))
    return tracks


# ---------------------------------------------------------------- rastérisation
def rasterize(x, y, t, x0, ytop, pixel_m, shape):
    mask = np.zeros(shape, dtype=bool)
    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    dist = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, dist / dt, np.inf)
    ok = (dt > 0) & (dt <= MAX_GAP_S) & (speed <= MAX_SPEED_MS)
    if not ok.any():
        return mask

    n = np.where(ok, np.maximum(1, np.ceil(dist / (pixel_m * 0.5))).astype(int), 0)
    npts = np.where(ok, n + 1, 0)
    total = npts.sum()
    seg = np.repeat(np.arange(len(n)), npts)
    offsets = np.cumsum(npts) - npts
    k = np.arange(total) - np.repeat(offsets, npts)
    f = k / n[seg]

    xs = x[:-1][seg] + dx[seg] * f
    ys = y[:-1][seg] + dy[seg] * f
    ix = np.floor((xs - x0) / pixel_m).astype(int)
    iy = np.floor((ytop - ys) / pixel_m).astype(int)
    valid = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    mask[iy[valid], ix[valid]] = True
    return mask


def compute_heat(tracks):
    all_lat = np.concatenate([t[1] for t in tracks])
    ref_lat = float(all_lat.mean())
    scale = cos(radians(ref_lat))

    proj = []
    for t, lat, lon in tracks:
        if SMOOTH_POINTS > 1 and len(lat) >= SMOOTH_POINTS:
            lat = uniform_filter1d(lat, SMOOTH_POINTS, mode="nearest")
            lon = uniform_filter1d(lon, SMOOTH_POINTS, mode="nearest")
        x, y = project(lat, lon, scale)
        proj.append((t, x, y))

    all_x = np.concatenate([p[1] for p in proj])
    all_y = np.concatenate([p[2] for p in proj])

    pixel_m = PIXEL_M
    extent = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    if extent / pixel_m > MAX_SIDE_PX:
        pixel_m = extent / MAX_SIDE_PX
        print(f"Zone très étendue : pixel porté à {pixel_m:.1f} m pour limiter la taille de l'image.")

    margin = 4 * BLUR_M + 2 * pixel_m
    x0 = all_x.min() - margin
    ytop = all_y.max() + margin
    w = int(np.ceil((all_x.max() + margin - x0) / pixel_m))
    h = int(np.ceil((ytop - (all_y.min() - margin)) / pixel_m))

    H = np.zeros((h, w), dtype=np.float32)
    for t, x, y in proj:
        H += rasterize(x, y, t, x0, ytop, pixel_m, (h, w))

    max_runs_raw = float(H.max())

    sigma_px = BLUR_M / pixel_m
    H = gaussian_filter(H, sigma=sigma_px, mode="constant", truncate=3.0)

    line_peak = 1.0 / (sqrt(2 * pi) * sigma_px)

    south, west = unproject(x0, ytop - h * pixel_m, scale)
    north, east = unproject(x0 + w * pixel_m, ytop, scale)
    bounds = [[float(south), float(west)], [float(north), float(east)]]

    return H, line_peak, bounds, max_runs_raw, pixel_m


# ---------------------------------------------------------------- rendu image
def make_lut(stops, n=256):
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=float)
    tt = np.linspace(0, 1, n)
    return np.stack([np.interp(tt, pos, cols[:, c]) for c in range(3)], axis=1).astype(np.uint8)


def render_base64(H, line_peak):
    """Convertit la matrice de chaleur en chaîne base64 (PNG RGBA)."""
    core = H > 0.5 * line_peak
    ref = H[core] if core.any() else H.ravel()
    cap = max(float(np.percentile(ref, PERCENTILE)), line_peak)

    v = np.clip(H / cap, 0, 1)
    v[H < 0.05 * line_peak] = 0
    v = v ** GAMMA

    lut = make_lut(PALETTES[PALETTE])
    rgb = lut[(v * 255).astype(np.uint8)]
    alpha = (np.clip(v * 3, 0, 1) * 255 * OPACITY).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    # Génération de l'image en mémoire
    img = Image.fromarray(rgba, "RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode()

    return b64_str, cap / line_peak


# ---------------------------------------------------------------- carte
def build_map(bounds, b64_img, n_runs, saturation, pixel_m):
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]

    # 1. Limiter le zoom de la carte globale
    m = folium.Map(location=center, zoom_start=15, tiles=None, max_zoom=17)

    # 2. Appliquer la même limite de zoom à la couche satellite Esri
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, USGS, NGA",
        name="Satellite",
        max_zoom=17
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{b64_img}",
        bounds=bounds,
        opacity=1.0,
        name="Heatmap",
        interactive=False,
        zindex=5,
    ).add_to(m)

    #m.fit_bounds(bounds) #cette ligne permet de faire rentrer toutes les courses sur le 1er affichage, quitte à beaucoup dézoomer
    folium.LayerControl(collapsed=True).add_to(m)

    info_box_html = f"""
    <div style="position: fixed; top: 15px; right: 15px; width: 290px; padding: 12px 16px;
        font-size: 13px; line-height: 1.55; background: white; border: 2px solid #2B2D42;
        border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15); color: #111; z-index: 9999;">
        <div style="font-weight: 700; font-size: 15px; margin-bottom: 8px;
            border-bottom: 1px solid #E0E0E0; padding-bottom: 4px;">Heatmap des Passages</div>
        <div style="margin: 3px 0;"><b>Courses analysées :</b> {n_runs}</div>
        <div style="margin: 3px 0;"><b>Résolution :</b> {pixel_m:.1f} m / pixel</div>
        <div style="margin: 3px 0;"><b>Lissage :</b> {BLUR_M:g} m</div>
        <div style="margin: 3px 0;"><b>Intensité max à ≈</b> {saturation:.0f} passages</div>
        <div style="margin: 3px 0; color: #6B6B6B;">Max 1 passage par course et par pixel</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_box_html))
    return m


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(FIT_DIR)
    if not tracks:
        print("Aucune trace GPS (avec horodatage) trouvée dans", FIT_DIR.resolve())
        raise SystemExit
    n_points = sum(len(t[0]) for t in tracks)
    print(f"{len(tracks)} courses, {n_points} points GPS.")

    H, line_peak, bounds, max_raw, pixel_m = compute_heat(tracks)
    print(f"Grille : {H.shape[1]} x {H.shape[0]} pixels ({pixel_m:.1f} m/pixel).")

    b64_img, saturation = render_base64(H, line_peak)
    m = build_map(bounds, b64_img, len(tracks), saturation, pixel_m)
    m.save(str(HTML_OUT))

    print(f"Carte enregistrée : {HTML_OUT.resolve()}")


if __name__ == "__main__":
    main()