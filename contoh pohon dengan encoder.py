from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Data dengan nama
nama = ["rayzan", "fanisa", "udin", "agus", "tatang", "ucok", "supri", "jajang"]
angka = [10, 20, 30, 0, 5, 1, 10, 5]
label = [1, 1, 0, 0, 1, 0, 1, 0]

# Ubah nama jadi angka
le = LabelEncoder()
nama_encoded = le.fit_transform(nama)

# Gabungkan nama encoded dan angka jadi fitur
data = [[n, a] for n, a in zip(nama_encoded, angka)]

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=42)

# Latih model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediksi dan akurasi untuk data test
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

# Visualisasi pohon
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['nama_encoded', 'angka'], class_names=['Tidak', 'Ya'], filled=True)
plt.title("Decision Tree dengan Nama (Encoded)")
plt.show()

print("Label Nama:", list(le.classes_))
print("Prediksi:", prediksi)
print("Akurasi:", akurasi)

# Prediksi data baru: ["dani", 10]
nama_baru = "dani"
angka_baru = 10

# Encode nama baru (kalau belum ada, harus di-handle)
if nama_baru in le.classes_:
    nama_baru_encoded = le.transform([nama_baru])[0]
else:
    # Kalau nama baru belum ada di label encoder, kasih kode unik baru
    # LabelEncoder gak otomatis bisa nambah, jadi kita kasih nilai manual
    nama_baru_encoded = max(nama_encoded) + 1

fitur_baru = [[nama_baru_encoded, angka_baru]]
prediksi_baru = model.predict(fitur_baru)

print(f"Prediksi untuk data baru {['dani', 10]}: {'Ya' if prediksi_baru[0]==1 else 'Tidak'}")
