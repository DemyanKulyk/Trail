import fitdecode
from pathlib import Path

FIT_FILE = Path("./data_fit/activity3.fit")

seen_types = set()

with fitdecode.FitReader(FIT_FILE) as fit:
    for frame in fit:
        if isinstance(frame, fitdecode.FitDataMessage) and frame.name in ("session", "lap", "record"):
            if frame.name not in seen_types:
                # Lecture directe sans get_value() pour éviter le KeyError sur les champs propriétaires
                fields = [f.name for f in frame.fields if f.value is not None]
                print(f"\n[{frame.name.upper()}] ({len(fields)} champs détectés) :")
                print(fields)
                seen_types.add(frame.name)

            # Arrêt dès qu'on a vu au moins un exemplaire de chaque type
            if seen_types == {"session", "lap", "record"}:
                break