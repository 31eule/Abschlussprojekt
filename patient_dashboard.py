import streamlit as st
import os
from login import logout_button
import patient_data

def app():
    st.title("Patientenansicht")
    st.write("Hier sehen Sie Ihre EKG-Daten und Empfehlungen.")

    # Stelle sicher, dass der Benutzer eingeloggt ist
    if "user_data" not in st.session_state:
        st.error("Nicht eingeloggt. Bitte melden Sie sich erneut an.")
        st.stop()

    patient = st.session_state.user_data

    # Patienten-Dashboard anzeigen (aus patient_data.py)
    patient_data.show_patient_details(patient)
    
    if st.session_state.get("logged_in"):
        logout_button()


