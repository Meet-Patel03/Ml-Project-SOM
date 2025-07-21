import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("dataset.csv")

df['ram'] = df['ram'].str.extract(r'(\d+)').astype(float)
df['storage'] = df['storage'].str.extract(r'(\d+)').astype(float)

df.loc[df['storage'] == 1.0, 'storage'] = 1024
df.loc[df['storage'] == 2.0, 'storage'] = 2048

df['ram_original'] = df['ram']
df['storage_original'] = df['storage']

scaler = MinMaxScaler()
df[['ram', 'storage']] = scaler.fit_transform(df[['ram', 'storage']])

feature_cols = ['os_brand', 'laptop_brand', 'processor_brand', 'usecases']
X = pd.get_dummies(df, columns=feature_cols, dtype=int)
X.drop(['laptop_id', 'name', 'price', 'processor', 'os', 'img_link',
        'display', 'rating', 'no_of_ratings', 'no_of_reviews',
        'ram_original', 'storage_original'], axis=1, inplace=True)

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X)

st.markdown("<h1 style='color: #FF00FF;'>💻 Laptop Recommender (with Images)</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='color: #00FFFF;'>Select OS Brand</h3>", unsafe_allow_html=True)
selected_os = st.selectbox("", df['os_brand'].unique())
if selected_os == "Windows":
    st.info("💡 Tip: Windows is the most compatible OS for general use, gaming, and productivity.")
elif selected_os == "MacOS":
    st.info("💡 Tip: MacOS is great for creative professionals and offers strong ecosystem integration.")
elif selected_os == "DOS":
    st.info("💡 Tip: DOS laptops come without a pre-installed OS — you’ll need to install Windows/Linux manually.")
elif selected_os == "ChromeOS":
    st.info("💡 Tip: ChromeOS is lightweight, secure, and best for web-based tasks — perfect for students and casual users.")

st.markdown("<h3 style='color: #00FFFF;'>Select Laptop Brand</h3>", unsafe_allow_html=True)
lap_brand = st.selectbox("", df['laptop_brand'].unique())

st.markdown("<h3 style='color: #00FFFF;'>Select Processor Brand</h3>", unsafe_allow_html=True)
proc_brand = st.selectbox("", df['processor_brand'].unique())
if "Apple M" in proc_brand:
    st.info("🍏 Tip: Apple M-series chips are highly efficient with great battery life and excellent performance for creative tasks.")
elif proc_brand == "AMD":
    st.info("🔥 Tip: AMD processors offer great multi-core performance and are excellent for multitasking and gaming.")
elif proc_brand == "Intel":
    st.info("💼 Tip: Intel processors are widely used and offer strong single-core performance — great for productivity and compatibility.")

st.markdown("<h3 style='color: #00FFFF;'>Select Usecase</h3>", unsafe_allow_html=True)
usecase = st.selectbox("", df['usecases'].unique())
usecase_cleaned = usecase.strip().lower()
usecase_tips = {
    "budget friendly": "💰 Tip: To get a budget-friendly laptop, consider lowering RAM and storage.",
    "business/professional": "📈 Tip: Business laptops benefit from premium build and security features.",
    "creative/design": "🎨 Tip: Creative work needs strong GPU and high-resolution displays.",
    "gaming": "🎮 Tip: For gaming, aim for high RAM, SSD storage, and a dedicated GPU.",
    "home/everyday use": "🏠 Tip: Everyday use laptops should be simple, light, and responsive.",
    "multimedia/entertainment": "🎧 Tip: For media, prioritize screen quality, speakers, and battery.",
    "student/education": "📚 Tip: Students need balanced performance, portability, and battery life.",
    "ultra-portable": "✈️ Tip: Ultra-portables should be thin, light, and have long battery backup."
}
if usecase_cleaned in usecase_tips:
    st.info(usecase_tips[usecase_cleaned])

st.markdown("<h3 style='color: #00FFFF;'>Select RAM (GB)</h3>", unsafe_allow_html=True)
ram = st.slider("", 4, 32, 8, step=4)
st.markdown("<h3 style='color: #00FFFF;'>Select Storage (GB)</h3>", unsafe_allow_html=True)
storage = st.slider("", 128, 2048, 1024, step=128)

if st.button("🔍 Recommend"):
    input_data = {
        f"os_brand_{selected_os}": 1,
        f"laptop_brand_{lap_brand}": 1,
        f"processor_brand_{proc_brand}": 1,
        f"usecases_{usecase}": 1,
        "ram": scaler.transform([[ram, storage]])[0][0],
        "storage": scaler.transform([[ram, storage]])[0][1]
    }

    input_vector = pd.DataFrame(data=np.zeros((1, X.shape[1])), columns=X.columns)
    for key, value in input_data.items():
        if key in input_vector.columns:
            input_vector.at[0, key] = value

    similarities = cosine_similarity(input_vector, X).flatten()
    top_indices = similarities.argsort()[::-1][:5]
    recommended = df.iloc[top_indices]

    st.markdown("<h2 style='color: #FF00FF;'>💡 Recommended Laptops</h2>", unsafe_allow_html=True)
    for _, row in recommended.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            if pd.notna(row.get('img_link', None)):
                st.image(row['img_link'], width=400)
        with col2:
            st.markdown(f"""
            ### {row['name']}
            - 💻 OS: {row['os_brand']}
            - 🏷 Brand: {row['laptop_brand']}
            - 🧠 Processor: {row['processor_brand']}
            - 🎯 Usecase: {row['usecases']}
            - 🧮 RAM: {int(round(row['ram_original']))} GB | 💾 Storage: {int(round(row['storage_original']))} GB
            - 📺 Display: {row.get('display', 'N/A')} inch
            - ⭐ Rating: {row.get('rating', 'N/A')} | 💰 Price: ₹{row.get('price', 'N/A')}
            """)
        st.markdown("---")
