import streamlit as st
import json
import os

import streamlit as st

def logout_button(show_message=True):
    query_params = st.query_params 
    """
    Zeigt einen fest positionierten Logout-Button unten rechts, wenn der Benutzer eingeloggt ist.
    """

    # Logout-Handling zuerst prüfen
    if st.query_params.get("logout") == "true":
        st.session_state.clear()
        st.session_state["logged_in"] = False
        st.session_state["show_logout_message"] = show_message

        # URL zurücksetzen – logout=true entfernen
        query_params.clear()
        st.rerun()

    if st.session_state.get("logged_in", False):
        # Logout-Button anzeigen
        st.markdown("""
            <style>
                .logout-fixed {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 9999;
                }
                .logout-fixed button {
                    background-color: #e74c3c;
                    color: white;
                    border-radius: 8px;
                    padding: 0.5rem 1rem;
                    border: none;
                    cursor: pointer;
                    font-size: 16px;
                }
            </style>
            <div class="logout-fixed">
                <form action="" method="get">
                    <input type="hidden" name="logout" value="true" />
                    <button type="submit">🚪 Logout</button>
                </form>
            </div>
        """, unsafe_allow_html=True)

    # Optional: Logout-Meldung anzeigen
    if st.session_state.get("show_logout_message", False):
        st.markdown(
            """
            <div style='
                max-width: 100%;
                margin-top: 10px;
                background-color: #1a1a1a;
                padding: 1.5rem 2rem;
                border-radius: 12px;
                border: 1px solid #444;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.05);
                text-align: center;
                color: #ccc;
                font-size: 1.1rem;
                line-height: 1.5;
            '>
                <h2 style='color: #4CAF50; margin-bottom: 1rem;'>✅ Erfolgreich ausgeloggt!</h2>
                <p>Bitte lade die Seite neu und logge dich erneut ein.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

if "show_logout_message" not in st.session_state:
    st.session_state["show_logout_message"] = False

def load_person_data():
    try:
        with open("data/person_db.json", "r", encoding="utf-8") as f1, open("data/arzt.json", "r", encoding="utf-8") as f2:
            person_data_1 = json.load(f1)
            person_data_2 = json.load(f2)
        return person_data_1 + person_data_2  # Zusammenführen zu einer Liste
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Fehler beim Laden der Benutzerdaten: {e}")
        return []


def login():
    st.title("Login für Arzt / Patient")

    role_choice = st.selectbox("Rolle wählen", ["Bitte wählen", "Arzt", "Patient"])
    username = st.text_input("Benutzername")
    password = st.text_input("Passwort", type="password")

    if st.button("Einloggen"):
        users = load_person_data()
        user = next((u for u in users if u.get("username") == username), None)

        if user and user.get("password") == password and user.get("role") == role_choice:
         st.session_state["logged_in"] = True
         st.session_state["role"] = user["role"]
         st.session_state["username"] = username
         st.session_state["user_data"] = user
         st.session_state["show_logout_message"] = False  # ✅ Meldung zurücksetzen
         st.success(f"Eingeloggt als {user['role']}")
         st.rerun()
        else:
            st.error("Login fehlgeschlagen – prüfe Benutzername, Passwort und Rolle.")

if __name__ == "__main__": 
    persons = load_person_data()
    print(persons)