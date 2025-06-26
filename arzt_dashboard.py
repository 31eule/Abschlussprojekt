import streamlit as st
import os
import arzt_data
from login import logout_button

def app():
    if "page" not in st.session_state:
        st.session_state.page = "list"
    if "selected_patient_id" not in st.session_state:
        st.session_state.selected_patient_id = None

    json_path = os.path.join("data", "person_db.json")

    patienten = arzt_data.load_patients(json_path)

    if st.session_state.page == "list":
        arzt_data.show_patient_list(patienten, json_path)

    elif st.session_state.page == "details":
        patient = next((p for p in patienten if p["id"] == st.session_state.selected_patient_id), None)
        if patient:
            arzt_data.show_patient_details(patient)
        else:
            st.error("Patient nicht gefunden.")
            if st.button("Zurück zur Liste"):
                st.session_state.page = "list"
                st.rerun()

    elif st.session_state.page == "add":
        arzt_data.add_patient_form(json_path)

    logout_button(alignment="bottom-right")
    
