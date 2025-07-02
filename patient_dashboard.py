import streamlit as st
import os
from login import logout_button
import patient_data

def app():
    st.title("Patientenansicht")
    st.write("Hier sehen Sie Ihre EKG-Daten und Empfehlungen.")

    if "user_data" not in st.session_state:
        st.error("Nicht eingeloggt. Bitte melden Sie sich erneut an.")
        st.stop()

    # Alle Patienten laden (entschlüsselt)
    patients = patient_data.load_patients("data/person_db_encrypted.bin")
    if not patients:
        st.error("Patientendaten konnten nicht geladen werden.")
        return

    # Angenommen, user_data enthält die Patient-ID
    patient_id = st.session_state["user_data"].get("id")
    if not patient_id:
        st.error("Ungültige Benutzerdaten.")
        return

    # Einzelnen Patienten finden
    patient = next((p for p in patients if p["id"] == patient_id), None)
    if not patient:
        st.error("Patient nicht gefunden.")
        return

    # Patientendetails anzeigen
    patient_data.show_patient_details(patient)

    if st.session_state.get("logged_in"):
        logout_button()


