from pathlib import Path
import math

import folium
import fitdecode
from scipy.spatial import cKDTree


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

FIT_DIR = SCRIPT_DIR / "real_data_fit"
OUTPUT_DIR = SCRIPT_DIR / "cartes_sorties"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML_OUT = OUTPUT_DIR / "runs_lines_smooth_snap15m_fast.html"


# ============================================================
# PARAMETERS
# ============================================================

SEMICIRCLE = 180.0 / (2 ** 31)

# Moving-average smoothing
SMOOTH_WINDOW = 5

# If a GPS point is <= 5 m from an existing route,
# snap it onto that route.
SNAP_THRESHOLD_M = 10

# Create reference points roughly every 2 metres
# along previous routes.
REFERENCE_SPACING_M = 2

# Visual appearance
HALO_WEIGHT = 50
HALO_OPACITY = 0.08

LINE_WEIGHT = 2
LINE_OPACITY = 0.45

MAX_DIRECTION_DIFFERENCE = 45

# ============================================================
# DISTANCE
# ============================================================

def haversine(coord1, coord2):
    """
    Distance in metres between two GPS coordinates.
    """

    r = 6371000

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


# ============================================================
# SMOOTH GPS TRACK
# ============================================================

def smooth_track(coords, window=5):
    """
    Smooth GPS coordinates with a moving average.
    """

    if window <= 1 or len(coords) < window:
        return coords

    smoothed = []

    half_window = window // 2

    for i in range(len(coords)):

        start = max(0, i - half_window)
        end = min(
            len(coords),
            i + half_window + 1
        )

        local_points = coords[start:end]

        avg_lat = (
            sum(p[0] for p in local_points)
            / len(local_points)
        )

        avg_lon = (
            sum(p[1] for p in local_points)
            / len(local_points)
        )

        smoothed.append(
            (avg_lat, avg_lon)
        )

    return smoothed


# ============================================================
# GPS -> APPROXIMATE METRES
# ============================================================

def gps_to_xy(coords, ref_lat):
    """
    Convert GPS coordinates to approximate metre coordinates.

    Good enough for small local areas such as running routes.
    """

    metres_per_degree_lat = 111320

    metres_per_degree_lon = (
        111320
        * math.cos(math.radians(ref_lat))
    )

    result = []

    for lat, lon in coords:

        x = lon * metres_per_degree_lon
        y = lat * metres_per_degree_lat

        result.append((x, y))

    return result

def angle_difference(dx1, dy1, dx2, dy2):
    """
    Return the smallest angle difference in degrees
    between two direction vectors.
    """

    angle1 = math.degrees(
        math.atan2(dy1, dx1)
    )

    angle2 = math.degrees(
        math.atan2(dy2, dx2)
    )

    diff = abs(angle1 - angle2)

    if diff > 180:
        diff = 360 - diff

    return diff

def xy_to_gps(x, y, ref_lat):
    """Convert local metre coordinates back to GPS."""

    lat = y / 111320

    lon = x / (
        111320 * math.cos(math.radians(ref_lat))
    )

    return (lat, lon)

# ============================================================
# CREATE DENSE REFERENCE ROUTE
# ============================================================

def densify_track(coords, spacing_m=2):
    """
    Add intermediate GPS points so that previous routes
    are represented approximately every spacing_m metres.
    """

    if len(coords) < 2:
        return coords

    dense = []

    for i in range(len(coords) - 1):

        a = coords[i]
        b = coords[i + 1]

        distance = haversine(a, b)

        n = max(
            1,
            int(math.ceil(distance / spacing_m))
        )

        for j in range(n):

            t = j / n

            lat = (
                a[0]
                + t * (b[0] - a[0])
            )

            lon = (
                a[1]
                + t * (b[1] - a[1])
            )

            dense.append(
                (lat, lon)
            )

    dense.append(coords[-1])

    return dense


# ============================================================
# SNAP CURRENT RUN TO PREVIOUS ROUTES
# ============================================================

def snap_with_kdtree(
    coords,
    reference_tracks,
    threshold_m=10,
    k_candidates=10
):
    """
    Snap the current route onto the existing canonical network.

    Returns:
        snapped_coords
        snapped_mask

    snapped_mask[i] = True
        -> point matched an existing route

    snapped_mask[i] = False
        -> point represents potentially new territory
    """

    if not reference_tracks:
        return coords, [False] * len(coords)

    all_reference_points = [
        point
        for track in reference_tracks
        for point in track
    ]

    ref_lat = (
        sum(p[0] for p in all_reference_points)
        / len(all_reference_points)
    )

    current_xy = gps_to_xy(
        coords,
        ref_lat
    )

    # ========================================================
    # Build REAL segments from each reference track separately
    # ========================================================

    segments = []

    for track in reference_tracks:

        track_xy = gps_to_xy(
            track,
            ref_lat
        )

        for i in range(len(track_xy) - 1):

            segments.append(
                (
                    track_xy[i],
                    track_xy[i + 1]
                )
            )

    if not segments:
        return coords, [False] * len(coords)

    # ========================================================
    # KD-tree
    # ========================================================

    midpoints = [
        (
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2
        )
        for a, b in segments
    ]

    tree = cKDTree(midpoints)

    snapped = []
    snapped_mask = []

    for i, point_xy in enumerate(current_xy):

        px, py = point_xy

        # ====================================================
        # Direction of current run
        # ====================================================

        if i == 0:

            dx_run = (
                current_xy[i + 1][0]
                - current_xy[i][0]
            )

            dy_run = (
                current_xy[i + 1][1]
                - current_xy[i][1]
            )

        elif i == len(current_xy) - 1:

            dx_run = (
                current_xy[i][0]
                - current_xy[i - 1][0]
            )

            dy_run = (
                current_xy[i][1]
                - current_xy[i - 1][1]
            )

        else:

            dx_run = (
                current_xy[i + 1][0]
                - current_xy[i - 1][0]
            )

            dy_run = (
                current_xy[i + 1][1]
                - current_xy[i - 1][1]
            )

        # ====================================================
        # Candidate segments
        # ====================================================

        k = min(
            k_candidates,
            len(segments)
        )

        _, candidate_indices = tree.query(
            point_xy,
            k=k
        )

        if k == 1:
            candidate_indices = [candidate_indices]

        best_distance = float("inf")
        best_projection = None

        for index in candidate_indices:

            a, b = segments[index]

            ax, ay = a
            bx, by = b

            dx_seg = bx - ax
            dy_seg = by - ay

            # ================================================
            # Direction compatibility
            # ================================================

            direction_difference = angle_difference(
                dx_run,
                dy_run,
                dx_seg,
                dy_seg
            )

            direction_difference = min(
                direction_difference,
                abs(180 - direction_difference)
            )

            if (
                direction_difference
                > MAX_DIRECTION_DIFFERENCE
            ):
                continue

            segment_length_squared = (
                dx_seg * dx_seg
                + dy_seg * dy_seg
            )

            if segment_length_squared == 0:
                continue

            # ================================================
            # Project point onto segment
            # ================================================

            t = (
                (px - ax) * dx_seg
                + (py - ay) * dy_seg
            ) / segment_length_squared

            t = max(
                0,
                min(1, t)
            )

            proj_x = ax + t * dx_seg
            proj_y = ay + t * dy_seg

            distance = math.hypot(
                px - proj_x,
                py - proj_y
            )

            if distance < best_distance:

                best_distance = distance

                best_projection = (
                    proj_x,
                    proj_y
                )

        # ====================================================
        # SNAP
        # ====================================================

        if (
            best_projection is not None
            and best_distance <= threshold_m
        ):

            snapped.append(
                xy_to_gps(
                    best_projection[0],
                    best_projection[1],
                    ref_lat
                )
            )

            snapped_mask.append(True)

        else:

            snapped.append(
                coords[i]
            )

            snapped_mask.append(False)

    return snapped, snapped_mask

def extract_new_sections(
    coords,
    snapped_mask,
    min_points=4
):
    """
    Extract only consecutive portions that did NOT match
    the existing canonical route network.

    Very short unmatched pieces are ignored because
    they are usually GPS noise.
    """

    sections = []
    current_section = []

    for point, was_snapped in zip(
        coords,
        snapped_mask
    ):

        if not was_snapped:

            current_section.append(
                point
            )

        else:

            if len(current_section) >= min_points:

                sections.append(
                    current_section
                )

            current_section = []

    if len(current_section) >= min_points:

        sections.append(
            current_section
        )

    return sections

# ============================================================
# 1. FIND FIT FILES
# ============================================================

fit_files = sorted(
    FIT_DIR.glob("*.fit")
)

if not fit_files:

    print(
        f"Aucun fichier .fit trouvé dans "
        f"{FIT_DIR.resolve()}"
    )

    exit()


# ============================================================
# 2. READ AND PROCESS RUNS
# ============================================================

all_tracks = []
all_points = []

# Contains dense points representing all previous paths

reference_tracks = []
processed_files_count = 0


for file_path in fit_files:

    print(f"Traitement : {file_path.name}")

    coords = []

    try:

        with fitdecode.FitReader(file_path) as fit:

            for frame in fit:

                if (
                    isinstance(
                        frame,
                        fitdecode.FitDataMessage
                    )
                    and frame.name == "record"
                ):

                    raw_lat = (
                        frame.get_value("position_lat")
                        if frame.has_field("position_lat")
                        else None
                    )

                    raw_lon = (
                        frame.get_value("position_long")
                        if frame.has_field("position_long")
                        else None
                    )

                    if (
                        raw_lat is not None
                        and raw_lon is not None
                    ):

                        lat = raw_lat * SEMICIRCLE
                        lon = raw_lon * SEMICIRCLE

                        coords.append(
                            (lat, lon)
                        )

    except Exception as e:

        print(
            f"Erreur fichier "
            f"{file_path.name} : {e}"
        )

        continue


    if len(coords) < 2:
        continue


    processed_files_count += 1


    # --------------------------------------------------------
    # STEP 1: Smooth the GPS data
    # --------------------------------------------------------

    coords_smoothed = smooth_track(
        coords,
        window=SMOOTH_WINDOW
    )


    # --------------------------------------------------------
    # STEP 2: Snap to previous routes using KD-tree
    # --------------------------------------------------------

    coords_snapped, snapped_mask = snap_with_kdtree(
        coords_smoothed,
        reference_tracks,
        threshold_m=SNAP_THRESHOLD_M
    )


    # --------------------------------------------------------
    # STEP 3: Save this run
    # --------------------------------------------------------

    all_tracks.append({
    "name": file_path.name,

    # Keep original smoothed route.
    # We will snap ALL routes again at the end.
    "coords_original": coords_smoothed,

    "coords": coords_snapped
    })


    # --------------------------------------------------------
    # STEP 4:
    # Make this run available as a reference for future runs.
    #
    # Densifying it means the next run can snap to the route
    # even if the watch recorded its GPS points at different
    # positions along the street.
    # --------------------------------------------------------

    # ============================================================
    # BUILD CANONICAL ROUTE NETWORK
    # ============================================================

    if len(reference_tracks) == 0:

        # First run creates the initial canonical network
        dense_reference = densify_track(
            coords_smoothed,
            spacing_m=REFERENCE_SPACING_M
        )

        reference_tracks.append(
            dense_reference
        )

    else:

        # Only genuinely NEW sections are allowed
        # to expand the reference network.
        new_sections = extract_new_sections(
            coords_smoothed,
            snapped_mask,
            min_points=10
        )

        for section in new_sections:

            dense_section = densify_track(
                section,
                spacing_m=REFERENCE_SPACING_M
            )

            reference_tracks.append(
                dense_section
            )
# ============================================================
# FINAL PASS
#
# Now that the complete canonical network exists,
# snap EVERY run again to it.
# ============================================================

all_points = []

for track in all_tracks:

    final_coords, _ = snap_with_kdtree(
        track["coords_original"],
        reference_tracks,
        threshold_m=SNAP_THRESHOLD_M
    )

    track["coords"] = final_coords

    all_points.extend(
        final_coords
    )

# ============================================================
# CHECK DATA
# ============================================================

if not all_points:

    print(
        "Aucune coordonnée GPS valide extraite."
    )

    exit()


# ============================================================
# 3. INITIALIZE MAP
# ============================================================

avg_lat = (
    sum(p[0] for p in all_points)
    / len(all_points)
)

avg_lon = (
    sum(p[1] for p in all_points)
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
# 4. DRAW RUNS
# ============================================================

for track in all_tracks:

    coords = track["coords"]


    # Wide transparent layer
    folium.PolyLine(
        locations=coords,
        color="#FF3300",
        weight=HALO_WEIGHT,
        opacity=HALO_OPACITY
    ).add_to(m)


    # Thin visible route
    folium.PolyLine(
        locations=coords,
        color="#FFFF33",
        weight=LINE_WEIGHT,
        opacity=LINE_OPACITY,
        tooltip=track["name"]
    ).add_to(m)


# ============================================================
# 5. AUTOMATIC MAP BOUNDS
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
# 6. INFORMATION BOX
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
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    color: #111;
    z-index: 9999;
"""


info_box_html = f"""
<div style="
    position: fixed;
    top: 15px;
    right: 15px;
    width: 280px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.55;
    {style_commun}
">

    <div style="
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 8px;
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 4px;
    ">
        Passages GPS
    </div>

    <div style="margin: 3px 0;">
        <b>Courses analysées :</b>
        {processed_files_count}
    </div>

    <div style="margin: 3px 0;">
        <b>Points GPS :</b>
        {len(all_points)}
    </div>

    <div style="margin: 3px 0;">
        <b>Lissage :</b>
        fenêtre de {SMOOTH_WINDOW} points
    </div>

    <div style="margin: 3px 0;">
        <b>Snap :</b>
        {SNAP_THRESHOLD_M} m
    </div>

    <div style="margin: 3px 0;">
        <b>Référence :</b>
        ~1 point / {REFERENCE_SPACING_M} m
    </div>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        info_box_html
    )
)


# ============================================================
# 7. SAVE
# ============================================================

m.save(
    str(HTML_OUT)
)

print()
print(
    f"Carte enregistrée : "
    f"{HTML_OUT.resolve()}"
)

print(
    f"Courses affichées : "
    f"{processed_files_count}"
)

total_reference_points = sum(
    len(track)
    for track in reference_tracks
)

print(
    f"Points de référence : "
    f"{total_reference_points}"
)