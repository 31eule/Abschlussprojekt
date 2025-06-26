import streamlit as st
import json
import os
import uuid
from PIL import Image
import pandas as pd
from ekgdata import EKGdata
from person import Person
from datetime import datetime
from weather import weather_app

def show_patient_details(patient):
    # Zwei Spalten: Links Patientendaten, rechts Wetter
    left_col, right_col = st.columns([3, 1])

    with left_col:
        st.title(f"🩺 {patient['firstname']} {patient['lastname']}")
        person = Person(patient)

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

                if "visible_start" not in st.session_state:
                    st.session_state.visible_start = 0
                if "visible_end" not in st.session_state:
                    st.session_state.visible_end = 5000

                max_index = len(ekg.df)
                min_time_ms = int(ekg.df["Zeit in ms"].min())
                max_time_ms = int(ekg.df["Zeit in ms"].max())

                st.markdown("#### 🕒 Sichtbereich manuell anpassen")
                manual_time = st.number_input(
                    "Startzeit (in ms):",
                    min_value=min_time_ms,
                    max_value=max_time_ms - 100,
                    value=int(ekg.df["Zeit in ms"].iloc[st.session_state.visible_start]),
                    step=100
                )

                if st.button("🔍 Bereich anzeigen"):
                    closest_idx = (ekg.df["Zeit in ms"] - manual_time).abs().idxmin()
                    st.session_state.visible_start = max(0, closest_idx)
                    st.session_state.visible_end = min(closest_idx + 5000, max_index)

                visible_range = (st.session_state.visible_start, st.session_state.visible_end)
                st.markdown(f"**Zeitbereich (ms):** {int(ekg.df['Zeit in ms'].iloc[visible_range[0]])} – {int(ekg.df['Zeit in ms'].iloc[visible_range[1]-1])}")

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

        # 💊 Medikation laden
        med_file = f"data/medikation/medikation_{person.id}.csv"
        if os.path.exists(med_file):
            st.markdown("### 💊 Aktuelle Medikation")
            try:
                df_med = pd.read_csv(med_file, sep=",")
                df_med.fillna("-", inplace=True)
                st.dataframe(df_med, use_container_width=True)
            except Exception as e:
                st.error(f"Fehler beim Laden der Medikation: {e}")
        else:
            st.info("Keine Medikationsdaten für diesen Patienten gefunden.")

    # Rechte Spalte: Wetter-Widget dauerhaft sichtbar
    with right_col:
        with st.container():
            weather_app()
