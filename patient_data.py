import streamlit as st
import json
import os
from PIL import Image
import pandas as pd
from ekgdata import EKGdata
from person import Person
from weather import weather_app
import encryption as enc

def load_patients(json_path):
    return enc.load_encrypted_json(json_path)

def show_patient_header(patient):
    st.title(f"🩺 {patient['firstname']} {patient['lastname']}")

def show_patient_info(person, patient):
    cols = st.columns([1, 2])
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
            **Größe:** {person.height} cm  
            **Gewicht:** {person.weight} kg  
            **Maximale Herzfrequenz:** {max_hr:.1f} bpm
        """)

        uploaded_file = st.file_uploader(
            "📷 Neues Bild hochladen",
            type=["png", "jpg", "jpeg"],
            key=f"upload_patient_image_{patient['id']}"
        )

        # Neu: Lade-Flag prüfen
        if uploaded_file is not None and not st.session_state.get(f"image_uploaded_{patient['id']}", False):
            image_folder = "data/pictures"
            os.makedirs(image_folder, exist_ok=True)
            new_path = os.path.join(image_folder, f"{patient['id']}.jpg")

            image = Image.open(uploaded_file).convert("RGB")
            image.save(new_path)

            try:
                patients = enc.load_encrypted_json("data/person_db_encrypted.bin")
            except FileNotFoundError:
                patients = []

            for p in patients:
                if p["id"] == patient["id"]:
                    p["picture_path"] = new_path
                    break
            else:
                patients.append(patient)

            enc.save_encrypted_json("data/person_db_encrypted.bin", patients)

            st.success("✅ Bild dauerhaft ersetzt.")

            # Lade-Flag setzen, damit st.rerun nur einmal ausgelöst wird
            st.session_state[f"image_uploaded_{patient['id']}"] = True
            st.rerun()

        # Lade-Flag nach Neuladen zurücksetzen, damit weitere Uploads möglich sind
        if st.session_state.get(f"image_uploaded_{patient['id']}", False):
            st.session_state[f"image_uploaded_{patient['id']}"] = False

def show_medication(patient):
    med_file = f"data/medikation/medikation_{patient['id']}.csv"
    if os.path.exists(med_file):
        st.markdown("### 💊 Aktuelle Medikation")
        try:
            df_med = pd.read_csv(med_file, sep=",")
            df_med.fillna("-", inplace=True)
            st.dataframe(df_med, use_container_width=True)

            next_med_times = []
            for idx, row in df_med.iterrows():
                if "Zeitpunkt" in df_med.columns and row["Zeitpunkt"] != "-":
                    times = row["Zeitpunkt"].split(",")
                    next_time, minutes_left = Person.next_medication_time(times)
                    next_med_times.append((row["Medikament"], next_time, minutes_left))

            if next_med_times:
                st.markdown("### 🕒 Nächste Medikamenteneinnahmen")
                for med, time_str, minutes_left in next_med_times:
                    st.info(f"🕒 Nächste Einnahme von **{med}** um {time_str} ({minutes_left} Minuten verbleibend)")
                Person.check_medication_notifications(next_med_times)
            else:
                st.info("Keine bevorstehenden Einnahmezeiten gefunden.")

        except Exception as e:
            st.error(f"Fehler beim Laden der Medikation: {e}")
    else:
        st.info("Keine Medikationsdaten für diesen Patienten gefunden.")

def show_ekg_data(person):
    ekg = None
    if person.ekg_tests:
        st.markdown("### 📈 EKG-Daten")
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

        selected_ekg_id = ekg_ids[selected_index]
        ekg_dict = EKGdata.load_by_id(person.ekg_tests, selected_ekg_id)

        if ekg_dict:
            ekg = EKGdata(ekg_dict)
            threshold = 340
            peaks = ekg.find_peaks(threshold)

            if "visible_start" not in st.session_state:
                st.session_state.visible_start = 0
            if "visible_end" not in st.session_state:
                st.session_state.visible_end = 5000

            max_index = len(ekg.df)
            min_time_ms = int(ekg.df["Zeit in ms"].min())
            max_time_ms = int(ekg.df["Zeit in ms"].max())
            min_time_min = min_time_ms // 60000
            max_time_min = max_time_ms // 60000

            # Sichtbereich in Minuten statt ms
            st.markdown("#### 🕒 Sichtbereich manuell anpassen")
            current_start_min = int(ekg.df["Zeit in ms"].iloc[st.session_state.visible_start] // 60000)
            manual_time_min = st.number_input(
                "Startzeit (in Minuten):",
                min_value=min_time_min,
                max_value=max_time_min - 1,
                value=current_start_min,
                step=1
            )

            if st.button("🔍 Bereich anzeigen"):
                manual_time_ms = manual_time_min * 60000
                closest_idx = (ekg.df["Zeit in ms"] - manual_time_ms).abs().idxmin()
                st.session_state.visible_start = max(0, closest_idx)
                st.session_state.visible_end = min(closest_idx + 5000, max_index)

            visible_range = (st.session_state.visible_start, st.session_state.visible_end)
            start_ms = int(ekg.df['Zeit in ms'].iloc[visible_range[0]])
            end_ms = int(ekg.df['Zeit in ms'].iloc[visible_range[1]-1])
            start_min = start_ms / 60000
            end_min = end_ms / 60000
            st.markdown(f"**Zeitbereich:** {start_min:.2f} – {end_min:.2f} Minuten")

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
                    extras_times = []
                    for peak_index, _ in extras:
                        try:
                            time = ekg.df["Zeit in ms"].iloc[peak_index]
                            extras_times.append(time)
                        except:
                            continue

                    df_extras = pd.DataFrame({"Zeitpunkt (ms)": extras_times})
                    st.markdown("**Zeitpunkte der Extrasystolen:**")
                    st.dataframe(df_extras.head(10), height=200)
                else:
                    st.success("✅ Keine Extrasystolen erkannt")

            st.markdown(f"**EKG-Datei:** {ekg_dict['result_link']}")
        else:
            st.warning("Kein EKG mit dieser ID gefunden.")
    else:
        st.warning("Keine EKG-Daten für diese Person vorhanden.")
    return ekg

def show_weather_and_notifications(patient):
    with st.container():
        next_med_times = Person.get_next_medication_times(patient)
        if not next_med_times:
            next_med_times = []

        notifications = Person.check_medication_notifications(next_med_times)

        for note in notifications:
            st.warning(note)

    with st.container():
        weather_app()

def show_disease_info(ekg):
    st.markdown("### 🏥 Erkannte Krankheiten und ihre Bedeutung")
    if ekg:
        disease_info = ekg.get_disease_descriptions_and_recommendations()
    else:
        disease_info = []

    if not disease_info:
        st.success("✅ Keine Auffälligkeiten erkannt.")
    else:
        for item in disease_info:
            with st.expander(item["title"]):
                st.markdown(item["description"])
                st.markdown("**Empfehlungen:**")
                st.markdown(item["recommendations"])

def show_patient_details(patient):
    left_col, right_col = st.columns([3, 1])

    with left_col:
        show_patient_header(patient)
        person = Person(patient)
        show_patient_info(person, patient)
        show_medication(patient)
        ekg = show_ekg_data(person)

    with right_col:
        show_weather_and_notifications(patient)
        show_disease_info(ekg)