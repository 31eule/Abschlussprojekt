import streamlit as st
import json
import os
import uuid
from PIL import Image
import pandas as pd

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

def show_patient_details(patient):
    st.title(f"🩺 {patient['firstname']} {patient['lastname']}")

    if patient.get("picture_path") and os.path.exists(patient["picture_path"]):
        st.image(Image.open(patient["picture_path"]), width=200)

    st.markdown(f"""
    **Geburtsjahr:** {patient['date_of_birth']}  
    **Geschlecht:** {patient['gender']}  
    """)

    st.subheader("📈 EKG-Tests")
    for test in patient.get("ekg_tests", []):
        st.markdown(f"- {test['date']}: [Ergebnis ansehen]({test['result_link']})")

    if st.button("⬅ Zurück zur Liste"):
        st.session_state.page = "list"
        st.rerun()