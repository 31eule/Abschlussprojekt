import json
from encryption import save_encrypted_json

# Pfad zur unverschlüsselten Originaldatei (falls vorhanden)
original_json_path = "data/person_db.json"

# Pfad zur verschlüsselten Datei mit neuem Schlüssel
encrypted_json_path = "data/person_db_encrypted.bin"

# Lade unverschlüsselte Daten
with open(original_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Speichere Daten mit neuem Schlüssel verschlüsselt ab
save_encrypted_json(encrypted_json_path, data)

print("Daten wurden erfolgreich mit dem neuen Schlüssel verschlüsselt gespeichert.")