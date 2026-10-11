from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.profile_type import (
    Activity,
    Event,
    EventType,
    FileType,
    Manufacturer,
    Sport,
    SubSport,
)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def interpolate_path(waypoints: list[tuple[float, float]], speed_mps: float):
    points = []
    total_dist = 0.0

    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        seg_dist = haversine_m(lat1, lon1, lat2, lon2)

        steps = max(1, int(seg_dist / speed_mps))
        step_speed = seg_dist / steps

        for s in range(steps):
            fraction = s / steps
            cur_lat = lat1 + fraction * (lat2 - lat1)
            cur_lon = lon1 + fraction * (lon2 - lon1)
            points.append((cur_lat, cur_lon, step_speed, total_dist))
            total_dist += step_speed

    last_lat, last_lon = waypoints[-1]
    points.append((last_lat, last_lon, 0.0, total_dist))
    return points


def generate_single_run_fit(
    filename: str,
    output_dir: Path,
    device_sn: int,
    start_time: datetime,
    waypoints: list[tuple[float, float]],
    speed_mps: float,
    base_hr: int,
):
    builder = FitFileBuilder()

    # 1. File ID
    file_id = FileIdMessage()
    file_id.type = FileType.ACTIVITY
    file_id.manufacturer = Manufacturer.GARMIN.value
    file_id.product = 3589  # FR55
    file_id.serial_number = device_sn
    file_id.time_created = int(start_time.timestamp() * 1000)
    builder.add(file_id)

    # 2. Timer Start
    timer_start = EventMessage()
    timer_start.timestamp = int(start_time.timestamp() * 1000)
    timer_start.event = Event.TIMER
    timer_start.event_type = EventType.START
    builder.add(timer_start)

    path = interpolate_path(waypoints, speed_mps)
    current_ts = int(start_time.timestamp())
    total_dist_m = 0.0

    # Records (1 Hz)
    for i, (lat, lon, spd, _) in enumerate(path):
        current_ts += 1
        total_dist_m += spd

        rec = RecordMessage()
        rec.timestamp = current_ts * 1000
        rec.position_lat = lat
        rec.position_long = lon
        rec.distance = round(total_dist_m, 2)
        rec.speed = round(spd, 2)
        rec.heart_rate = int(base_hr + 6 * math.sin(i / 30.0))
        rec.altitude = 373.0
        builder.add(rec)

    # 3. Lap Message unique (100% Run)
    total_duration = current_ts - int(start_time.timestamp())
    lap = LapMessage()
    lap.timestamp = current_ts * 1000
    lap.start_time = int(start_time.timestamp() * 1000)
    lap.start_position_lat = path[0][0]
    lap.start_position_long = path[0][1]
    lap.end_position_lat = path[-1][0]
    lap.end_position_long = path[-1][1]
    lap.total_elapsed_time = float(total_duration)
    lap.total_timer_time = float(total_duration)
    lap.total_distance = round(total_dist_m, 2)
    lap.sport = Sport.RUNNING
    builder.add(lap)

    # 4. Timer Stop
    timer_stop = EventMessage()
    timer_stop.timestamp = current_ts * 1000
    timer_stop.event = Event.TIMER
    timer_stop.event_type = EventType.STOP_ALL
    builder.add(timer_stop)

    # 5. Session Message
    session = SessionMessage()
    session.timestamp = current_ts * 1000
    session.start_time = int(start_time.timestamp() * 1000)
    session.start_position_lat = path[0][0]
    session.start_position_long = path[0][1]
    session.total_elapsed_time = float(total_duration)
    session.total_timer_time = float(total_duration)
    session.total_distance = round(total_dist_m, 2)
    session.sport = Sport.RUNNING
    session.sub_sport = SubSport.GENERIC
    session.total_cycles = 1
    builder.add(session)

    # 6. Activity Message
    activity = ActivityMessage()
    activity.timestamp = current_ts * 1000
    activity.num_sessions = 1
    activity.type = Activity.MANUAL
    builder.add(activity)

    # Export
    out_file = output_dir / filename
    fit_file = builder.build()
    with open(out_file, "wb") as f:
        f.write(fit_file.to_bytes())

    pace_min_km = (1000.0 / speed_mps) / 60.0
    pace_m = int(pace_min_km)
    pace_s = int((pace_min_km - pace_m) * 60)
    print(
        f"{filename:<25} | Distance: {total_dist_m/1000:.2f} km | Allure: {pace_m}:{pace_s:02d}/km | Durée: {total_duration/60:.1f} min"
    )


def main():
    target_dir = Path("data_fit_generated")
    target_dir.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2026, 10, 10, 9, 0, 0, tzinfo=timezone.utc)

    # --- TOPOLOGIE VIDY ---
    # Point clé : Tous les tracés empruntent la berge [46.5168, 6.6002 -> 46.5165, 6.5960 -> 46.5173, 6.5915]
    # pour créer une forte densité sur la heatmap le long du lac.

    runners = [
        {
            "filename": "runner_1_boucle_theatre.fit",
            "sn": 100001,
            "delay_min": 0,
            "speed_mps": 3.85,  # 4:20/km
            "base_hr": 160,
            "waypoints": [
                (46.5186, 6.6025),  # Stade Pierre-de-Coubertin
                (46.5168, 6.6002),  # Berges
                (46.5165, 6.5960),  # Quai Gustave Doret (TRONÇON COMMUN)
                (46.5173, 6.5915),  # Berges Ouest
                (46.5189, 6.5932),  # Théâtre de Vidy
                (46.5205, 6.5970),  # Bois-de-Vaux
                (46.5186, 6.6025),  # Retour Stade
            ],
        },
        {
            "filename": "runner_2_grand_tour_parc.fit",
            "sn": 100002,
            "delay_min": 2,
            "speed_mps": 3.45,  # 4:50/km
            "base_hr": 154,
            "waypoints": [
                (46.5186, 6.6025),  # Stade
                (46.5168, 6.6002),  # Berges
                (46.5165, 6.5960),  # Quai Gustave Doret (TRONÇON COMMUN)
                (46.5173, 6.5915),  # Berges Ouest
                (46.5200, 6.5900),  # Camping de Vidy
                (46.5215, 6.5940),  # Allées nord
                (46.5208, 6.6015),  # Avenue Pierre-de-Coubertin
                (46.5186, 6.6025),  # Retour Stade
            ],
        },
        {
            "filename": "runner_3_extension_venoge.fit",
            "sn": 100003,
            "delay_min": 3,
            "speed_mps": 4.17,  # 4:00/km
            "base_hr": 166,
            "waypoints": [
                (46.5186, 6.6025),  # Stade
                (46.5168, 6.6002),  # Berges
                (46.5165, 6.5960),  # Quai Gustave Doret (TRONÇON COMMUN)
                (46.5173, 6.5915),  # Berges Ouest
                (46.5180, 6.5820),  # Rives ouest vers Embouchure Chamberonne / UNIL
                (46.5210, 6.5845),  # Sentier du bord de rivière
                (46.5215, 6.5940),  # Jonction Bois-de-Vaux
                (46.5186, 6.6025),  # Retour Stade
            ],
        },
        {
            "filename": "runner_4_aller_retour_ouchy.fit",
            "sn": 100004,
            "delay_min": 5,
            "speed_mps": 3.12,  # 5:20/km
            "base_hr": 145,
            "waypoints": [
                (46.5186, 6.6025),  # Stade
                (46.5168, 6.6002),  # Berges
                (46.5165, 6.5960),  # Quai Gustave Doret (TRONÇON COMMUN)
                (46.5168, 6.6002),  # Demi-tour berges
                (46.5150, 6.6080),  # Prolongation Quai vers Bellerive / Ouchy
                (46.5145, 6.6120),  # Piscine de Bellerive
                (46.5160, 6.6070),  # Retour
                (46.5186, 6.6025),  # Stade
            ],
        },
        {
            "filename": "runner_5_interieur_parc.fit",
            "sn": 100005,
            "delay_min": 7,
            "speed_mps": 2.78,  # 6:00/km
            "base_hr": 140,
            "waypoints": [
                (46.5208, 6.6015),  # Départ Nord Stade
                (46.5186, 6.6025),  # Descente vers Stade
                (46.5168, 6.6002),  # Berges
                (46.5165, 6.5960),  # Quai Gustave Doret (TRONÇON COMMUN)
                (46.5189, 6.5932),  # Remontée Théâtre
                (46.5208, 6.6015),  # Clôture boucle nord
            ],
        },
    ]

    for r in runners:
        start_ts = base_time + timedelta(minutes=r["delay_min"])
        generate_single_run_fit(
            filename=r["filename"],
            output_dir=target_dir,
            device_sn=r["sn"],
            start_time=start_ts,
            waypoints=r["waypoints"],
            speed_mps=r["speed_mps"],
            base_hr=r["base_hr"],
        )


if __name__ == "__main__":
    main()