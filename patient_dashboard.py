import streamlit as st
import json
from login import logout_button

def app():
    logout_button() 
    st.title("Patientenansicht")
    st.write("Hier sehen Sie Ihre EKG-Daten und Empfehlungen.")

def load_person_data():
    """A Function that knows where the person database is and returns a dictionary with the persons"""
    file = open("data/person_db.json")
    person_data = json.load(file)
    return person_data

