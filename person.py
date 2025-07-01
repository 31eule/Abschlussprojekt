import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta

class Person:
    
    @staticmethod
    def load_person_data():
        """Lädt die Personendaten aus der JSON-Datei"""
        with open("data/person_db.json", "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def save_person_data(data):
        """Speichert die Personendaten zurück in die JSON-Datei"""
        with open("data/person_db.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    @staticmethod
    def get_person_list(person_data):
        """Gibt eine Liste aller Personennamen zurück"""
        list_of_names = []
        for eintrag in person_data:
            list_of_names.append(eintrag["lastname"] + ", " +  eintrag["firstname"])
        return list_of_names
    
    @staticmethod
    def find_person_data_by_name(suchstring):
        """Findet und gibt die Personendaten anhand von Nachname, Vorname zurück"""
        person_data = Person.load_person_data()
        if suchstring == "None":
            return {}

        two_names = suchstring.split(", ")
        vorname = two_names[1]
        nachname = two_names[0]

        for eintrag in person_data:
            if (eintrag["lastname"] == nachname and eintrag["firstname"] == vorname):
                return eintrag
        else:
            return {}
        
    def __init__(self, person_dict) -> None:
        self.id = person_dict["id"]
        self.firstname = person_dict["firstname"]
        self.lastname = person_dict["lastname"]
        self.date_of_birth = person_dict["date_of_birth"]
        self.picture_path = person_dict.get("picture_path", "data/pictures/none.jpg")
        self.gender = person_dict.get("gender", "unknown")
        self.height = person_dict["height"]  # in cm
        self.weight = person_dict["weight"]  # in kg
        self.ekg_tests = person_dict.get("ekg_tests", [])

    @staticmethod
    def load_by_id(pid, person_dict):
        for person_data in person_dict:
            if str(person_data["id"]) == str(pid):  # Vergleich als String für Sicherheit
                return Person(person_data)
        return None
    
    def calc_age(self):
        today_year = date.today().year
        return today_year - self.date_of_birth

    def calc_max_heart_rate(self):
        age = self.calc_age()
        if self.gender == "male":
            max_heart = 223 - 0.9 * age
        else:
            max_heart = 226 - age
        return max_heart
    
    @staticmethod
    def update_height_weight(patient_id, new_height, new_weight):
        try:
            # Personendaten laden
            person_list = Person.load_person_data()

            # Patient suchen und Werte aktualisieren
            updated = False
            for person in person_list:
                if str(person["id"]) == str(patient_id):
                    person["height"] = new_height
                    person["weight"] = new_weight
                    updated = True
                    break

            if not updated:
                return "❌ Patient nicht gefunden."

            # Daten speichern
            Person.save_person_data(person_list)
            return "✅ Größe und Gewicht wurden erfolgreich aktualisiert."

        except Exception as e:
            return f"❌ Fehler beim Aktualisieren der Daten: {e}"


    @staticmethod
    def add_ekg(patient_id, uploaded_file, ekg_date):
        try:
            # JSON laden
            with open("data/person_db.json", "r", encoding="utf-8") as f:
                person_list = json.load(f)

            # Passende Person finden
            for person in person_list:
                if str(person["id"]) == str(patient_id):
                    ekg_tests = person.get("ekg_tests", [])

                    # Neue eindeutige Test-ID
                    new_id = max((test["id"] for test in ekg_tests), default=0) + 1

                    # Zielordner erstellen, falls nicht vorhanden
                    folder = "data/ekg_data"
                    os.makedirs(folder, exist_ok=True)

                    # Dateiname und Pfad definieren
                    base_filename = f"{patient_id}_{new_id}.txt"
                    file_path = os.path.join(folder, base_filename)

                    # Datei speichern
                    with open(file_path, "wb") as out_file:
                        out_file.write(uploaded_file.getbuffer())

                    # Neues EKG zur Liste hinzufügen
                    person.setdefault("ekg_tests", []).append({
                        "id": new_id,
                        "date": ekg_date.strftime("%d.%m.%Y"),
                        "result_link": file_path.replace("\\", "/")
                    })

                    # JSON speichern
                    with open("data/person_db.json", "w", encoding="utf-8") as f:
                        json.dump(person_list, f, indent=4)

                    return f"✅ EKG gespeichert unter {file_path}"

            return "❌ Patient nicht gefunden"

        except Exception as e:
            return f"❌ Fehler beim Speichern der EKG-Datei: {e}"

    @staticmethod    
    def next_medication_time(times):
        now = datetime.now()
        min_diff = timedelta(days=1)
        next_time_str = None

        for t in times:
            try:
                med_time = datetime.strptime(t.strip(), "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                )
            except ValueError:
                continue
            
            if med_time < now:
                med_time += timedelta(days=1)

            diff = med_time - now
            if diff < min_diff:
                min_diff = diff
                next_time_str = t.strip()

        return next_time_str, int(min_diff.total_seconds() // 60)
    
    @staticmethod
    def get_next_medication_times(patient):
        med_file = f"data/medikation/medikation_{patient['id']}.csv"
        if not os.path.exists(med_file):
            return []

        try:
            df_med = pd.read_csv(med_file)
        except Exception:
            return []

        # Zeiten aller Medikamente gruppieren
        med_times = {}
        for _, row in df_med.iterrows():
            med = row['Medikament']
            zeit = row['Zeitpunkt']
            if med not in med_times:
                med_times[med] = []
            med_times[med].append(zeit)

        notifications = []
        for med, times in med_times.items():
            next_time, minutes_left = Person.next_medication_time(times)
            if next_time is not None:
                notifications.append((med, next_time, minutes_left))

        return notifications
    
    @staticmethod
    def check_medication_notifications(next_med_times):
        notifications = []
        for med, time_str, minutes_left in next_med_times:
            if 0 < minutes_left <= 10:
                notifications.append(f"⚠️ Einnahme von **{med}** in etwa {minutes_left} Minuten (um {time_str})")
        return notifications

            
if __name__ == "__main__":
    #print("This is a module with some functions to read the person data")
    persons = Person.load_person_data()
    person_names = Person.get_person_list(persons)
    #print(person_names)
    print(Person.find_person_data_by_name("Huber, Julian"))
    id = "001"
    #print(Person.load_by_id(id, persons))
    person = Person.load_by_id(id, persons)
    #print (Person.calc_age(Person, person))
    age = Person.calc_age(Person, person)
    print(age)
    print(Person.calc_max_heart_rate(Person, person, age))