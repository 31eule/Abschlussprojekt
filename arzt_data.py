import streamlit as st
import json
import os
import uuid
from PIL import Image
import pandas as pd
from ekgdata import EKGdata
from person import Person
from datetime import datetime

def load_patients(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_patients(json_path, patienten):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(patienten, f, indent=4, ensure_ascii=False)

def add_patient_form(json_path):
    col1, col2 = st.columns([9, 1])

    with col1:
        st.title("➕ Neuen Patienten hinzufügen")

    with col2:
        if st.button("⬅ Zurück"):
            st.session_state.page = "list"
            st.rerun()

    # Deine Eingabefelder bleiben unverändert
    firstname = st.text_input("Vorname")
    lastname = st.text_input("Nachname")
    date_of_birth = st.number_input("Geburtsjahr", min_value=1900, max_value=2100, step=1)
    gender = st.selectbox("Geschlecht", ["male", "female", "diverse"])
    height = st.number_input("Größe (in cm)", min_value=0, step=1)
    weight = st.number_input("Gewicht (in kg)", min_value=0.0, step=0.1)
    picture_path = st.text_input("Bildpfad (optional)", placeholder="z. B. data/pictures/p1.jpg")

    if st.button("Hinzufügen"):
        if firstname and lastname:
            patienten = load_patients(json_path)
            new_patient = {
                "id": str(uuid.uuid4())[:8],
                "firstname": firstname,
                "lastname": lastname,
                "date_of_birth": int(date_of_birth),
                "gender": gender,
                "height": int(height),
                "weight": float(weight),
                "picture_path": picture_path,
                "ekg_tests": []
            }
            patienten.append(new_patient)
            save_patients(json_path, patienten)
            st.success(f"Patient {firstname} {lastname} wurde hinzugefügt!")
            st.rerun()
        else:
            st.error("Vorname und Nachname sind erforderlich.")

def show_patient_list(patienten, json_path):
    st.title("👥 Patientenliste")

    if st.button("➕ Neuen Patienten hinzufügen"):
        st.session_state.page = "add"
        st.rerun()

    st.markdown("### Patientenübersicht")

    header_cols = st.columns([2, 2, 2, 1, 1])
    header_cols[0].markdown("**Vorname**")
    header_cols[1].markdown("**Nachname**")
    header_cols[2].markdown("**Geburtsjahr**")
    header_cols[3].markdown("**Details**")
    header_cols[4].markdown("**Löschen**")

    for patient in patienten:
        cols = st.columns([2, 2, 2, 1, 1])
        cols[0].write(patient["firstname"])
        cols[1].write(patient["lastname"])
        cols[2].write(patient["date_of_birth"])

        if cols[3].button("Details", key=f"details_{patient['id']}"):
            st.session_state.selected_patient_id = patient["id"]
            st.session_state.page = "details"
            st.rerun()

        if cols[4].button("Löschen", key=f"delete_{patient['id']}"):
            # Patienten entfernen und speichern
            patienten = [p for p in patienten if p["id"] != patient["id"]]
            save_patients(json_path, patienten)
            st.success(f"Patient {patient['firstname']} {patient['lastname']} wurde gelöscht.")
            st.rerun()

person_dict = Person.load_person_data()
person_names = Person.get_person_list(person_dict)

def show_patient_details(patient):
    st.title(f"🩺 {patient['firstname']} {patient['lastname']}")

    person = Person(patient)

    cols = st.columns([1, 2])  # 1 Teil für Bild, 2 Teile für Daten (etwas mehr Platz rechts)

    with cols[0]:
        if patient.get("picture_path") and os.path.exists(patient["picture_path"]):
            st.image(Image.open(patient["picture_path"]), width=200)

    with cols[1]:
        age = person.calc_age()
        max_hr = person.calc_max_heart_rate()

        st.markdown(f"""
        **Geschlecht:** {person.gender}  
        **Geburtsjahr:** {person.date_of_birth}  
        **Alter:** {age} Jahre  
        **Maximale Herzfrequenz:** {max_hr:.1f} bpm
        """)

        # Editable Felder für Größe und Gewicht
        new_height = st.number_input(
            "Größe (cm)",
            min_value=0,
            value=int(person.height),
            step=1,
            key="height_input"
        )

        new_weight = st.number_input(
            "Gewicht (kg)",
            min_value=0.0,
            value=float(person.weight),
            step=0.1,
            format="%.1f",
            key="weight_input"
        )

        if st.button("Größe und Gewicht speichern"):
            msg = Person.update_height_weight(person.id, new_height, new_weight)
            st.success(msg)
            # Aktualisiere Objekt im Speicher
            person.height = new_height
            person.weight = new_weight
            st.rerun()

    if person.ekg_tests:
        ekg_ids = [ekg["id"] for ekg in person.ekg_tests]

        if ('current_person_id_for_ekg' not in st.session_state or
            st.session_state.current_person_id_for_ekg != person.id or
            'selected_ekg_index' not in st.session_state):
            st.session_state.current_person_id_for_ekg = person.id
            st.session_state.selected_ekg_index = 0

        selected_index = st.selectbox(
            "EKG auswählen",
            options=range(len(ekg_ids)),
            index=st.session_state.selected_ekg_index,
            format_func=lambda x: str(x + 1),
            key="selected_ekg_index"
        )

        if selected_index != st.session_state.selected_ekg_index:
            st.session_state.selected_ekg_index = selected_index

        selected_ekg_id = ekg_ids[selected_index]
        ekg_dict = EKGdata.load_by_id(person.ekg_tests, selected_ekg_id)

        if ekg_dict:
            ekg = EKGdata(ekg_dict)
            threshold = 340
            peaks = ekg.find_peaks(threshold)

            # Initialisiere Sichtbereich
            if "visible_start" not in st.session_state:
                st.session_state.visible_start = 0
            if "visible_end" not in st.session_state:
                st.session_state.visible_end = 5000

            max_index = len(ekg.df)

            # Zeitbereich in Millisekunden auslesen
            min_time_ms = int(ekg.df["Zeit in ms"].min())
            max_time_ms = int(ekg.df["Zeit in ms"].max())

            # Eingabefeld für Startzeit (ms)
            st.markdown("#### 🕒 Sichtbereich manuell anpassen")
            manual_time = st.number_input(
                "Startzeit (in ms):",
                min_value=min_time_ms,
                max_value=max_time_ms - 100,
                value=int(ekg.df["Zeit in ms"].iloc[st.session_state.visible_start]),
                step=100
            )

            # Button zur Bestätigung
            if st.button("🔍 Bereich anzeigen"):
                closest_idx = (ekg.df["Zeit in ms"] - manual_time).abs().idxmin()
                st.session_state.visible_start = max(0, closest_idx)
                st.session_state.visible_end = min(closest_idx + 5000, max_index)

            visible_range = (st.session_state.visible_start, st.session_state.visible_end)
            # st.markdown(f"**Aktueller Indexbereich:** {visible_range[0]} – {visible_range[1]}")
            st.markdown(f"**Zeitbereich (ms):** {int(ekg.df['Zeit in ms'].iloc[visible_range[0]])} – {int(ekg.df['Zeit in ms'].iloc[visible_range[1]-1])}")

            # Plot anzeigen
            fig = ekg.plot_time_series(peaks, visible_range)
            st.plotly_chart(fig)

            estimated_hr = ekg.estimate_hr(peaks)
            st.markdown(f"**Geschätzte Herzfrequenz:** {estimated_hr:.1f} bpm")

            st.markdown("### 🧠 Analyse des EKGs:")

            with st.expander("Ergebnisse anzeigen"):
                if ekg.find_bradykardie():
                    st.warning("🟡 Bradykardie erkannt (Herzfrequenz < 60 bpm)")
                elif ekg.find_tachykardie():
                    st.warning("🔴 Tachykardie erkannt (Herzfrequenz > 100 bpm)")
                else:
                    st.success("✅ Normale Herzfrequenz")

                if ekg.find_atrial_fibrillation():
                    st.warning("🟡 Verdacht auf Vorhofflimmern (hohe RR-Variabilität)")

                st_status = ekg.detect_st_elevation()
                if st_status == "ST-Hebung":
                    st.error("🔴 ST-Streckenhebung erkannt – möglicher Myokardinfarkt")
                elif st_status == "ST-Senkung":
                    st.warning("🟠 ST-Streckensenkung erkannt")
                else:
                    st.success("✅ ST-Strecke im Normalbereich")

                extras = ekg.find_extrasystoles()
                if extras:
                    st.warning(f"🟡 {len(extras)} mögliche Extrasystolen erkannt")

                    # ➕ Zeitpunkt in ms extrahieren
                    extras_times = []
                    for peak_index, _ in extras:
                        try:
                            time = ekg.df["Zeit in ms"].iloc[peak_index]
                            extras_times.append(time)
                        except:
                            continue

                    # ➕ DataFrame für Anzeige
                    df_extras = pd.DataFrame({"Zeitpunkt (ms)": extras_times})
                    
                    # ➕ Nur die ersten 10 Zeilen anzeigen, aber scrollbar
                    st.markdown("**Zeitpunkte der Extrasystolen:**")
                    st.dataframe(df_extras.head(10), height=200)
                else:
                    st.success("✅ Keine Extrasystolen erkannt")

            st.markdown(f"**EKG-Datei:** {ekg_dict['result_link']}")
        else:
            st.warning("Kein EKG mit dieser ID gefunden.")
    else:
        st.warning("Keine EKG-Daten für diese Person vorhanden.")

    st.markdown("## 📤 Neues EKG hochladen")

    # Patient muss vorher ausgewählt worden sein
    selected_patient_id = st.session_state.get("selected_patient_id", None)
    if selected_patient_id is None:
        st.warning("Bitte zuerst einen Patienten auswählen.")
    else:
        uploaded_file = st.file_uploader("Wähle eine EKG CSV-Datei aus", type=["txt"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Datei erfolgreich geladen.")
                st.dataframe(df.head(10))

                # Eingabe optionales EKG-Datum (voreingestellt auf heute)
                ekg_date = st.date_input("Datum des EKGs", value=datetime.today())

                if st.button("EKG speichern"):
                    # Person-Instanz erstellen (optional, je nach Struktur deiner App)
                    person = Person.load_by_id(selected_patient_id, Person.load_person_data())
                    if person:
                        result_message = Person.add_ekg(selected_patient_id, uploaded_file, ekg_date)
                        st.success(result_message)
                        st.rerun()
                    else:
                        st.error("❌ Patient nicht gefunden.")

            except Exception as e:
                st.error(f"Fehler beim Lesen der Datei: {e}")

    st.markdown("## 💊 Medikation")

    # Sicherstellen, dass der Ordner existiert
    med_folder = "data/medikation"
    os.makedirs(med_folder, exist_ok=True)

    # Pfad zur Medikamenten-CSV-Datei
    med_file = os.path.join(med_folder, f"medikation_{patient['id']}.csv")

    # Bestehende Datei laden oder neue leere Tabelle erzeugen
    try:
        med_df = pd.read_csv(med_file)
    except FileNotFoundError:
        med_df = pd.DataFrame(columns=["Medikament", "Zeitpunkt", "Bemerkung"])

    with st.form("med_form"):
        col1, col2, col3 = st.columns([3, 2, 3])
        medikament = col1.text_input("Medikament")
        zeitpunkt = col2.time_input("Einnahmezeitpunkt")
        bemerkung = col3.text_input("Bemerkung (optional)")
        submitted = st.form_submit_button("Hinzufügen")

    if submitted and medikament:
        new_entry = {
            "Medikament": medikament,
            "Zeitpunkt": zeitpunkt.strftime("%H:%M"),
            "Bemerkung": bemerkung
        }
        med_df = pd.concat([med_df, pd.DataFrame([new_entry])], ignore_index=True)
        med_df.to_csv(med_file, index=False)
        st.success("💾 Medikationseintrag gespeichert")
        st.rerun()

    if not med_df.empty:
        st.markdown("### 📋 Bisherige Einträge:")
        st.dataframe(med_df, use_container_width=True)
    else:
        st.info("Noch keine Medikation erfasst.")

    if st.button("⬅ Zurück zur Liste"):
        st.session_state.page = "list"
        st.rerun()