from pathlib import Path
import math

import fitdecode
import folium
import osmnx as ox

from shapely.geometry import Point, LineString
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

FIT_DIR = SCRIPT_DIR / "real_data_fit"

OUTPUT_DIR = SCRIPT_DIR / "cartes_sorties"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HTML_OUT = (
    OUTPUT_DIR
    / "runs_osm_map_matched.html"
)


# ============================================================
# PARAMETERS
# ============================================================

SEMICIRCLE = 180.0 / (2 ** 31)


# ------------------------------------------------------------
# GPS smoothing
# ------------------------------------------------------------

SMOOTH_WINDOW = 5


# ------------------------------------------------------------
# OSM download area
#
# Extra area around all GPS runs.
# ------------------------------------------------------------

OSM_MARGIN_M = 300


# ------------------------------------------------------------
# Maximum GPS -> OSM distance
#
# If a GPS point is farther than this from every OSM path,
# leave the point at its original position.
#
# Start with 25 m.
# ------------------------------------------------------------

MAX_SNAP_DISTANCE_M = 25


# ------------------------------------------------------------
# OSM network type
#
# "walk" gives walkable roads + pedestrian paths.
# ------------------------------------------------------------

OSM_NETWORK_TYPE = "walk"


# ------------------------------------------------------------
# Map appearance
# ------------------------------------------------------------

HALO_WEIGHT = 30
HALO_OPACITY = 0.06

LINE_WEIGHT = 3
LINE_OPACITY = 0.65


# ============================================================
# OSMNX SETTINGS
# ============================================================

# Keep downloaded OSM data cached locally.
# This makes subsequent executions much faster.

ox.settings.use_cache = True

ox.settings.log_console = True


# ============================================================
# SMOOTH GPS TRACK
# ============================================================

def smooth_track(
    coords,
    window=5
):
    """
    Smooth GPS coordinates using a moving average.
    """

    if (
        window <= 1
        or len(coords) < window
    ):
        return coords

    smoothed = []

    half_window = window // 2

    for i in range(len(coords)):

        start = max(
            0,
            i - half_window
        )

        end = min(
            len(coords),
            i + half_window + 1
        )

        local_points = (
            coords[start:end]
        )

        avg_lat = (
            sum(
                p[0]
                for p in local_points
            )
            / len(local_points)
        )

        avg_lon = (
            sum(
                p[1]
                for p in local_points
            )
            / len(local_points)
        )

        smoothed.append(
            (
                avg_lat,
                avg_lon
            )
        )

    return smoothed


# ============================================================
# READ ONE FIT FILE
# ============================================================

def read_fit_file(file_path):
    """
    Read latitude/longitude coordinates from a FIT file.
    """

    coords = []

    try:

        with fitdecode.FitReader(
            file_path
        ) as fit:

            for frame in fit:

                if not (
                    isinstance(
                        frame,
                        fitdecode.FitDataMessage
                    )
                    and frame.name == "record"
                ):
                    continue


                raw_lat = (
                    frame.get_value(
                        "position_lat"
                    )
                    if frame.has_field(
                        "position_lat"
                    )
                    else None
                )


                raw_lon = (
                    frame.get_value(
                        "position_long"
                    )
                    if frame.has_field(
                        "position_long"
                    )
                    else None
                )


                if (
                    raw_lat is None
                    or raw_lon is None
                ):
                    continue


                lat = (
                    raw_lat
                    * SEMICIRCLE
                )

                lon = (
                    raw_lon
                    * SEMICIRCLE
                )


                coords.append(
                    (
                        lat,
                        lon
                    )
                )


    except Exception as e:

        print(
            f"Erreur fichier "
            f"{file_path.name}: {e}"
        )

        return []


    return coords


# ============================================================
# CALCULATE GPS BOUNDING BOX
# ============================================================

def create_bbox(
    points,
    margin_m=300
):
    """
    Create an OSM bounding box around all GPS points.

    Returns:
        (left, bottom, right, top)
    """

    lats = [
        p[0]
        for p in points
    ]

    lons = [
        p[1]
        for p in points
    ]


    min_lat = min(lats)
    max_lat = max(lats)

    min_lon = min(lons)
    max_lon = max(lons)


    mean_lat = (
        sum(lats)
        / len(lats)
    )


    # Approximate metre -> degree conversion

    lat_margin = (
        margin_m
        / 111320
    )


    lon_margin = (
        margin_m
        /
        (
            111320
            * math.cos(
                math.radians(
                    mean_lat
                )
            )
        )
    )


    bottom = (
        min_lat
        - lat_margin
    )

    top = (
        max_lat
        + lat_margin
    )

    left = (
        min_lon
        - lon_margin
    )

    right = (
        max_lon
        + lon_margin
    )


    return (
        left,
        bottom,
        right,
        top
    )


# ============================================================
# GET EDGE GEOMETRY
# ============================================================

def get_edge_geometry(
    graph,
    u,
    v,
    key
):
    """
    Return the geometry of an OSM edge.

    Some edges have an explicit geometry.
    Straight edges sometimes do not, so we construct
    a LineString from their start/end nodes.
    """

    data = graph.edges[
        u,
        v,
        key
    ]


    geometry = data.get(
        "geometry"
    )


    if geometry is not None:
        return geometry


    node_u = graph.nodes[u]
    node_v = graph.nodes[v]


    return LineString([
        (
            node_u["x"],
            node_u["y"]
        ),
        (
            node_v["x"],
            node_v["y"]
        )
    ])


# ============================================================
# SNAP ONE RUN TO OPENSTREETMAP
# ============================================================

def snap_track_to_osm(
    coords,
    graph_projected,
    to_projected,
    to_wgs84,
    max_distance_m=25
):
    """
    Snap GPS points onto the closest OSM edge.

    The graph and GPS points are projected into metres,
    so distances are real metric distances.
    """

    if not coords:
        return []


    # --------------------------------------------------------
    # Convert GPS coordinates to projected metre coordinates
    # --------------------------------------------------------

    projected_points = []

    for lat, lon in coords:

        x, y = to_projected.transform(
            lon,
            lat
        )

        projected_points.append(
            (
                x,
                y
            )
        )


    xs = [
        p[0]
        for p in projected_points
    ]

    ys = [
        p[1]
        for p in projected_points
    ]


    # --------------------------------------------------------
    # Find nearest OSM edge for ALL points in one operation.
    #
    # This is much faster than querying one point at a time.
    # --------------------------------------------------------

    nearest_edges, distances = (
        ox.distance.nearest_edges(
            graph_projected,
            X=xs,
            Y=ys,
            return_dist=True
        )
    )


    snapped_coords = []


    # --------------------------------------------------------
    # Project each point exactly onto its corresponding edge
    # --------------------------------------------------------

    for (
        original_coord,
        projected_coord,
        edge_id,
        edge_distance
    ) in zip(
        coords,
        projected_points,
        nearest_edges,
        distances
    ):

        # ----------------------------------------------------
        # GPS too far from any mapped OSM path:
        # keep original coordinate
        # ----------------------------------------------------

        if (
            edge_distance
            > max_distance_m
        ):

            snapped_coords.append(
                original_coord
            )

            continue


        u = int(edge_id[0])
        v = int(edge_id[1])
        key = int(edge_id[2])


        edge_geometry = (
            get_edge_geometry(
                graph_projected,
                u,
                v,
                key
            )
        )


        gps_point = Point(
            projected_coord
        )


        # ----------------------------------------------------
        # Exact closest point on OSM line
        # ----------------------------------------------------

        distance_along_edge = (
            edge_geometry.project(
                gps_point
            )
        )


        snapped_point = (
            edge_geometry.interpolate(
                distance_along_edge
            )
        )


        # ----------------------------------------------------
        # Convert back to longitude / latitude
        # ----------------------------------------------------

        lon, lat = (
            to_wgs84.transform(
                snapped_point.x,
                snapped_point.y
            )
        )


        snapped_coords.append(
            (
                lat,
                lon
            )
        )


    return snapped_coords


# ============================================================
# 1. FIND FIT FILES
# ============================================================

fit_files = sorted(
    FIT_DIR.glob(
        "*.fit"
    )
)


if not fit_files:

    print(
        f"Aucun fichier .fit trouvé dans "
        f"{FIT_DIR.resolve()}"
    )

    raise SystemExit


print(
    f"{len(fit_files)} fichiers FIT trouvés."
)


# ============================================================
# 2. READ + SMOOTH ALL RUNS
# ============================================================

all_tracks = []

all_smoothed_points = []


for file_path in fit_files:

    print(
        f"Lecture : "
        f"{file_path.name}"
    )


    raw_coords = (
        read_fit_file(
            file_path
        )
    )


    if len(raw_coords) < 2:

        print(
            "  -> pas assez de points GPS"
        )

        continue


    coords_smoothed = (
        smooth_track(
            raw_coords,
            window=SMOOTH_WINDOW
        )
    )


    all_tracks.append({
        "name":
            file_path.name,

        "coords_original":
            raw_coords,

        "coords_smoothed":
            coords_smoothed,

        "coords":
            None
    })


    all_smoothed_points.extend(
        coords_smoothed
    )


print()

print(
    f"Courses valides : "
    f"{len(all_tracks)}"
)

print(
    f"Points GPS : "
    f"{len(all_smoothed_points)}"
)


if not all_tracks:

    print(
        "Aucune course GPS valide."
    )

    raise SystemExit


# ============================================================
# 3. CREATE OPENSTREETMAP BOUNDING BOX
# ============================================================

bbox = create_bbox(
    all_smoothed_points,
    margin_m=OSM_MARGIN_M
)


left, bottom, right, top = bbox


print()
print(
    "Zone OpenStreetMap :"
)

print(
    f"  left   = {left}"
)

print(
    f"  bottom = {bottom}"
)

print(
    f"  right  = {right}"
)

print(
    f"  top    = {top}"
)


# ============================================================
# 4. DOWNLOAD ACTUAL OSM WALKABLE NETWORK
# ============================================================

print()

print(
    "Téléchargement du réseau "
    "OpenStreetMap..."
)


graph = ox.graph.graph_from_bbox(
    bbox,
    network_type=OSM_NETWORK_TYPE,

    # Keep disconnected pedestrian paths too.
    retain_all=True,

    # Keep edges entering the bounding box.
    truncate_by_edge=True,

    simplify=True
)


print(
    f"Réseau OSM : "
    f"{len(graph.nodes)} noeuds, "
    f"{len(graph.edges)} segments."
)


# ============================================================
# 5. PROJECT OSM NETWORK TO METRES
# ============================================================

print(
    "Projection du réseau..."
)


graph_projected = (
    ox.projection.project_graph(
        graph
    )
)


projected_crs = (
    graph_projected.graph[
        "crs"
    ]
)


# GPS WGS84 -> local projected CRS

to_projected = (
    Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True
    )
)


# Local projected CRS -> GPS WGS84

to_wgs84 = (
    Transformer.from_crs(
        projected_crs,
        "EPSG:4326",
        always_xy=True
    )
)


# ============================================================
# 6. MAP-MATCH EACH RUN TO OSM
# ============================================================

print()

print(
    "Map matching des courses..."
)


all_points = []


for index, track in enumerate(
    all_tracks,
    start=1
):

    print(
        f"[{index}/{len(all_tracks)}] "
        f"{track['name']}"
    )


    snapped_coords = (
        snap_track_to_osm(
            track["coords_smoothed"],
            graph_projected,
            to_projected,
            to_wgs84,
            max_distance_m=(
                MAX_SNAP_DISTANCE_M
            )
        )
    )


    track["coords"] = (
        snapped_coords
    )


    all_points.extend(
        snapped_coords
    )


# ============================================================
# 7. INITIALIZE FOLIUM MAP
# ============================================================

avg_lat = (
    sum(
        p[0]
        for p in all_points
    )
    / len(all_points)
)


avg_lon = (
    sum(
        p[1]
        for p in all_points
    )
    / len(all_points)
)


m = folium.Map(

    location=[
        avg_lat,
        avg_lon
    ],

    zoom_start=15,

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr=(
        "Esri, HERE, Garmin, "
        "USGS, NGA"
    )
)


# ============================================================
# 8. DRAW MAP-MATCHED RUNS
# ============================================================

for track in all_tracks:

    coords = track["coords"]


    if len(coords) < 2:
        continue


    # --------------------------------------------------------
    # Transparent frequency halo
    # --------------------------------------------------------

    folium.PolyLine(

        locations=coords,

        color="#FF3300",

        weight=HALO_WEIGHT,

        opacity=HALO_OPACITY

    ).add_to(m)


    # --------------------------------------------------------
    # Visible run line
    # --------------------------------------------------------

    folium.PolyLine(

        locations=coords,

        color="#FFFF33",

        weight=LINE_WEIGHT,

        opacity=LINE_OPACITY,

        tooltip=track["name"]

    ).add_to(m)


# ============================================================
# 9. MAP BOUNDS
# ============================================================

lats = [
    p[0]
    for p in all_points
]

lons = [
    p[1]
    for p in all_points
]


m.fit_bounds([
    [
        min(lats),
        min(lons)
    ],
    [
        max(lats),
        max(lons)
    ]
])


# ============================================================
# 10. INFORMATION BOX
# ============================================================

style_commun = """
    background: white;
    border: 2px solid #2B2D42;
    border-radius: 8px;
    font-family: -apple-system,
                 BlinkMacSystemFont,
                 'Segoe UI',
                 Roboto,
                 sans-serif;
    box-shadow: 0 4px 12px
                rgba(0,0,0,0.15);
    color: #111;
    z-index: 9999;
"""


info_box_html = f"""
<div style="
    position: fixed;
    top: 15px;
    right: 15px;
    width: 300px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.55;
    {style_commun}
">

    <div style="
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 8px;
        border-bottom:
            1px solid #E0E0E0;
        padding-bottom: 4px;
    ">
        Passages GPS — OSM
    </div>

    <div style="margin: 3px 0;">
        <b>Courses analysées :</b>
        {len(all_tracks)}
    </div>

    <div style="margin: 3px 0;">
        <b>Points GPS :</b>
        {len(all_points)}
    </div>

    <div style="margin: 3px 0;">
        <b>Lissage :</b>
        {SMOOTH_WINDOW} points
    </div>

    <div style="margin: 3px 0;">
        <b>Map matching :</b>
        OpenStreetMap
    </div>

    <div style="margin: 3px 0;">
        <b>Réseau :</b>
        {OSM_NETWORK_TYPE}
    </div>

    <div style="margin: 3px 0;">
        <b>Distance max :</b>
        {MAX_SNAP_DISTANCE_M} m
    </div>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        info_box_html
    )
)


# ============================================================
# 11. SAVE
# ============================================================

m.save(
    str(
        HTML_OUT
    )
)


print()
print(
    "=================================="
)

print(
    "TERMINÉ"
)

print(
    "=================================="
)

print(
    f"Carte enregistrée : "
    f"{HTML_OUT.resolve()}"
)

print(
    f"Courses affichées : "
    f"{len(all_tracks)}"
)

print(
    f"Points affichés : "
    f"{len(all_points)}"
)