import streamlit as st
import json
import os

def logout_button(alignment="right", show_message=True):
    """
    Zeigt einen Logout-Button in der gewünschten Position.
    
    :param alignment: "left", "center", "right", "bottom-right"
    :param show_message: Ob eine Logout-Meldung nach Logout angezeigt wird
    """

    # Floating Position: Unten rechts
    if alignment == "bottom-right":
        st.markdown("""
            <style>
                .logout-fixed {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 1000;
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
                <form action="?logout=true" method="post">
                    <button type="submit">🚪 Logout</button>
                </form>
            </div>
        """, unsafe_allow_html=True)

        # Abfangen des "Logout" via Query-Param
        if st.query_params.get("logout") == "true":
            st.session_state.clear()
            st.session_state["logged_in"] = False
            st.session_state["show_logout_message"] = show_message
            st.rerun()

    else:
        # Leichte vertikale Verschiebung
        st.markdown("<div style='margin-top: -30px;'></div>", unsafe_allow_html=True)

        # Flexible Spaltenaufteilung je nach Ausrichtung
        if alignment == "right":
            cols = st.columns([6, 1])
        elif alignment == "center":
            cols = st.columns([4, 1, 4])
        else:  # "left"
            cols = st.columns([1, 6])

        with cols[1]:
            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.session_state["logged_in"] = False
                st.session_state["show_logout_message"] = show_message
                st.rerun()

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
            st.success(f"Eingeloggt als {user['role']}")
            st.rerun()
        else:
            st.error("Login fehlgeschlagen – prüfe Benutzername, Passwort und Rolle.")

if __name__ == "__main__": 
    persons = load_person_data()
    print(persons)