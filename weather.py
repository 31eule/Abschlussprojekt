import streamlit as st
import requests

def weather_app():
    st.title("🌦️ Wetter")
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
            elif "snow" in weather or "schnee" in weather:
                st.toast("❄️ Achtung: Es schneit in deiner Stadt!", icon="⚠️")
            elif "fog" in weather or "nebel" in weather:
                st.toast("🌫️ Achtung: Es ist neblig in deiner Stadt!", icon="⚠️")
            else:
                st.toast("🌤️ Das Wetter sieht gut aus!", icon="✅")

            if temp < 0:
                st.toast("🥶 Achtung: Es ist sehr kalt in deiner Stadt! Es kann zur Glätte führen", icon="⚠️")
            elif temp > 30:
                st.toast("🌡️ Achtung: Es ist sehr heiß in deiner Stadt! Achte auf ausreichend Flüssigkeit", icon="⚠️")
            else:
                st.toast("🌡️ Das Wetter ist angenehm.", icon="✅")

        else:
            st.error("Stadt nicht gefunden oder API-Problem.")

