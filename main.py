import streamlit as st
import joblib
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_models():
    """Betölti és cache-eli a modelleket."""
    return {
        "GKP": joblib.load("modelgkp.pkl"),
        "DEF": joblib.load("modeldef.pkl"),
        "MID": joblib.load("modelmid.pkl"),
        "FWD": joblib.load("modelfwd.pkl")
    }

# Modellek betöltése
models = load_models()

# Dropdown-okhoz opciók
positions = ["GKP", "DEF", "MID", "FWD"]
teams = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
         'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
         'Liverpool', 'Luton', 'Man City', 'Man Utd', 'Newcastle',
         "Nott'm Forest", 'Sheffield Utd', 'Spurs', 'West Ham', 'Wolves']

st.title("Prediktált Expected Goals")

# Felhasználói inputok
position = st.selectbox("Poszt", positions)
team = st.selectbox("Csapat", teams)
creativity = st.slider("Kreativitás", min_value=0, max_value=1000, value=300, step=1)
now_cost = st.slider("Költség (now_cost)", min_value=30, max_value=130, value=70, step=1)
age = st.slider("Életkor", min_value=16, max_value=45, value=25, step=1)

# Predikció gomb
if st.button("Prediktálás"):
    # Input dataframe összeállítása
    input_df = pd.DataFrame([{ 
        "position": position,
        "team": team,
        "now_cost": float(now_cost),
        "age": int(age),
        "creativity": int(creativity)
    }])
    # Modell kiválasztása és predikció
    model = models[position]
    prediction = model.predict(input_df)[0]

    # Eredmény megjelenítése
    st.subheader("Expected goals")
    st.write(round(prediction, 2))