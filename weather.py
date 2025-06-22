import streamlit as st
import requests

st.title("🌦️ Wetter")

city = st.text_input("Stadt eingeben:", "Berlin")

api_key = "8fb8e55c5aaa04fbc1ef0b2206e67e1b"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=de"

if city:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        st.subheader(f"Wetter in {city}")
        st.write(f"🌡️ Temperatur: {data['main']['temp']} °C")
        st.write(f"☁️ Wetter: {data['weather'][0]['description'].capitalize()}")
        st.write(f"💧 Luftfeuchtigkeit: {data['main']['humidity']} %")
        st.write(f"🌬️ Windgeschwindigkeit: {data['wind']['speed']} m/s")
    else:
        st.error("Stadt nicht gefunden oder API-Probleme.")

