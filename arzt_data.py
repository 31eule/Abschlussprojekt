import streamlit as st
import os
import uuid
from PIL import Image
import pandas as pd
from ekgdata import EKGdata
from person import Person
from encryption import load_encrypted_json, save_encrypted_json

json_path = "data/person_db_encrypted.bin"

def load_patients(json_path):
    return load_encrypted_json(json_path)

def save_patients(json_path, patienten):
    save_encrypted_json(json_path, patienten)

def add_patient_form(json_path):
    col1, col2 = st.columns([9, 1])

    with col1:
        st.title("➕ Neuen Patienten hinzufügen")

    with col2:
        if st.button("⬅ Zurück"):
            st.session_state.page = "list"
            st.rerun()

    password = st.text_input("Passwort", type="password", placeholder="Passwort für den neuen Patienten")
    firstname = st.text_input("Vorname")
    lastname = st.text_input("Nachname")
    date_of_birth = st.number_input("Geburtsjahr", min_value=1900, max_value=2100, step=1)
    gender = st.selectbox("Geschlecht", ["male", "female", "diverse"])
    height = st.number_input("Größe (in cm)", min_value=0, step=1)
    weight = st.number_input("Gewicht (in kg)", min_value=0.0, step=0.1)
    uploaded_file = st.file_uploader("📷 Patientenbild hochladen", type=["jpg", "jpeg", "png"])

    if st.button("Hinzufügen"):
        if firstname and lastname:
            # Patientenliste laden (aktuellste Daten)
            patienten = load_patients(json_path)

            image_path = ""
            if uploaded_file is not None:
                image_folder = "data/patient_pictures"
                os.makedirs(image_folder, exist_ok=True)
                image_path = os.path.join(image_folder, f"{firstname.lower()}_{lastname.lower()}.jpg")
                image = Image.open(uploaded_file)
                image.save(image_path)

            new_patient = {
                "id": str(uuid.uuid4())[:8],
                "username": f"{firstname.lower()}.{lastname.lower()}",
                "role": "Patient",
                "password": password,
                "firstname": firstname,
                "lastname": lastname,
                "date_of_birth": int(date_of_birth),
                "gender": gender,
                "height": int(height),
                "weight": float(weight),
                "picture_path": image_path,
                "ekg_tests": []
            }

            patienten.append(new_patient)

            # Patientenliste speichern
            save_patients(json_path, patienten)

            # Patientenliste direkt neu laden und in Session speichern für frische Daten
            st.session_state["patienten"] = load_patients(json_path)

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

        confirm_key = f"confirm_delete_{patient['id']}"

        if st.session_state.get(confirm_key):
            st.warning(f"Willst du Patient {patient['firstname']} {patient['lastname']} wirklich löschen?")
            yes = cols[4].button("Ja, löschen", key=f"yes_delete_{patient['id']}")
            no = cols[4].button("Nein", key=f"no_delete_{patient['id']}")

            if yes:
                neue_patienten = [p for p in patienten if p["id"] != patient["id"]]
                save_patients(json_path, neue_patienten)

                # Nach Löschen auch die Session aktualisieren
                st.session_state["patienten"] = load_patients(json_path)

                st.success(f"Patient {patient['firstname']} {patient['lastname']} wurde gelöscht.")
                st.session_state.pop(confirm_key)
                st.rerun()

            if no:
                st.session_state.pop(confirm_key)
                st.rerun()

        else:
            if cols[4].button("Löschen", key=f"delete_{patient['id']}"):
                st.session_state[confirm_key] = True
                st.rerun()

# Beispiel wie du initial die Patienten aus Session-State oder Datei holst
if "patienten" not in st.session_state:
    json_path = "data/person_db_encrypted.bin"  # oder dein tatsächlicher Pfad
    st.session_state["patienten"] = load_patients(json_path)

# In deinem Hauptprogramm dann z.B. so aufrufen:
if st.session_state.get("page", "list") == "list":
    show_patient_list(st.session_state["patienten"], json_path)
elif st.session_state["page"] == "add":
    add_patient_form(json_path)

def show_person_header(person, patient):
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
        **Maximale Herzfrequenz:** {max_hr:.1f} bpm
        """)

        new_height = st.number_input("Größe (cm)", min_value=0, value=int(person.height), step=1, key="height_input")
        new_weight = st.number_input("Gewicht (kg)", min_value=0.0, value=float(person.weight), step=0.1, format="%.1f", key="weight_input")

        if st.button("Größe und Gewicht speichern"):
            msg = Person.update_height_weight(person.id, new_height, new_weight)
            st.success(msg)
            person.height = new_height
            person.weight = new_weight
            st.rerun()

def show_ekg_analysis(person):
    if not person.ekg_tests:
        st.warning("Keine EKG-Daten für diese Person vorhanden.")
        return

    ekg_ids = [ekg["id"] for ekg in person.ekg_tests]
    if 'current_person_id_for_ekg' not in st.session_state or st.session_state.current_person_id_for_ekg != person.id:
        st.session_state.current_person_id_for_ekg = person.id
        st.session_state.selected_ekg_index = 0

    selected_index = st.selectbox("EKG auswählen", options=range(len(ekg_ids)), index=st.session_state.selected_ekg_index,
                                  format_func=lambda x: str(x + 1), key="selected_ekg_index")

    selected_ekg_id = ekg_ids[selected_index]
    ekg_dict = EKGdata.load_by_id(person.ekg_tests, selected_ekg_id)

    if not ekg_dict:
        st.warning("Kein EKG mit dieser ID gefunden.")
        return

    ekg = EKGdata(ekg_dict)
    peaks = ekg.find_peaks(340)

    # Sichtbereich initialisieren
    st.session_state.visible_start = st.session_state.get("visible_start", 0)
    st.session_state.visible_end = st.session_state.get("visible_end", 5000)
    max_index = len(ekg.df)

    # Bereich manuell setzen (in Minuten)
    st.markdown("#### 🕒 Sichtbereich manuell anpassen")
    min_time_ms = int(ekg.df["Zeit in ms"].min())
    max_time_ms = int(ekg.df["Zeit in ms"].max())
    min_time_min = min_time_ms // 60000
    max_time_min = max_time_ms // 60000

    # Aktueller Startzeitpunkt in Minuten
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

    st.plotly_chart(ekg.plot_time_series(peaks, visible_range))
    st.markdown(f"**Geschätzte Herzfrequenz:** {ekg.estimate_hr(peaks):.1f} bpm")

    with st.expander("🧠 Analyse des EKGs"):
        analyze_ekg_signals(ekg, peaks)

    st.markdown(f"**EKG-Datei:** {ekg_dict['result_link']}")

def analyze_ekg_signals(ekg, peaks):
    if ekg.find_bradykardie():
        st.warning("🟡 Bradykardie erkannt (Herzfrequenz < 60 bpm)")
    elif ekg.find_tachykardie():
        st.warning("🔴 Tachykardie erkannt (Herzfrequenz > 100 bpm)")
    else:
        st.success("✅ Normale Herzfrequenz")

    if ekg.find_atrial_fibrillation():
        st.warning("🟡 Verdacht auf Vorhofflimmern")

    st_status = ekg.detect_st_elevation()
    if st_status == "ST-Hebung":
        st.error("🔴 ST-Hebung erkannt – möglicher Infarkt")
    elif st_status == "ST-Senkung":
        st.warning("🟠 ST-Senkung erkannt")
    else:
        st.success("✅ ST-Strecke im Normalbereich")

    extras = ekg.find_extrasystoles()
    if extras:
        times = [ekg.df["Zeit in ms"].iloc[i] for i, _ in extras if i < len(ekg.df)]
        df_extras = pd.DataFrame({"Zeitpunkt (ms)": times})
        st.warning(f"🟡 {len(times)} mögliche Extrasystolen erkannt")
        st.dataframe(df_extras.head(10), height=200)
    else:
        st.success("✅ Keine Extrasystolen erkannt")

def show_medication_section(patient):
    st.markdown("## 💊 Medikation")

    med_folder = "data/medikation"
    os.makedirs(med_folder, exist_ok=True)
    med_file = os.path.join(med_folder, f"medikation_{patient['id']}.csv")

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
            "Bemerkung": bemerkung if bemerkung.strip() else "-"
        }
        med_df = pd.concat([med_df, pd.DataFrame([new_entry])], ignore_index=True)
        med_df.to_csv(med_file, index=False)
        st.success("💾 Medikationseintrag gespeichert")
        st.rerun()

    if not med_df.empty:
        st.markdown("### 📋 Bisherige Einträge:")
        for idx, row in med_df.iterrows():
            bemerkung = row["Bemerkung"] if pd.notna(row["Bemerkung"]) and row["Bemerkung"].strip() else "-"
            cols = st.columns([3, 2, 3, 1])
            cols[0].markdown(f"**{row['Medikament']}**")
            cols[1].markdown(row["Zeitpunkt"])
            cols[2].markdown(bemerkung)
            if cols[3].button("🗑️", key=f"del_{idx}"):
                med_df = med_df.drop(index=idx).reset_index(drop=True)
                med_df.to_csv(med_file, index=False)
                st.success("❌ Eintrag gelöscht")
                st.rerun()
    else:
        st.info("Noch keine Medikation erfasst.")

def show_patient_details(patient):
    st.title(f"🩺 {patient['firstname']} {patient['lastname']}")
    person = Person(patient)

    show_person_header(person, patient)
    show_ekg_analysis(person)
    show_medication_section(patient)

    if st.button("⬅ Zurück zur Liste"):
        st.session_state.page = "list"
        st.rerun()

