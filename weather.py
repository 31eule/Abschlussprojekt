import streamlit as st
import requests

st.title("🌤️ Wetter-App mit Benachrichtigung")

city = st.text_input("Stadt eingeben", "Berlin")
api_key = "8fb8e55c5aaa04fbc1ef0b2206e67e1b"  # <-- hier deinen OpenWeatherMap API-Key einfügen

if st.button("Wetter prüfen"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=de"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["main"].lower()
        temp = data["main"]["temp"]

        st.write(f"🌡️ Temperatur: {temp}°C")
        st.write(f"☁️ Wetterlage: {weather}")

        # "Push-artige" Toast-Nachricht bei Regen
        if "rain" in weather or "regen" in weather:
            st.toast("⛈️ Achtung: Es regnet in deiner Stadt!", icon="⚠️")
        else:
            st.toast("☀️ Kein Regen – alles gut!", icon="✅")

    else:
        st.error("Stadt nicht gefunden oder API-Problem.")

