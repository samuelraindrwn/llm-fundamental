# 🧠 Belajar Fundamental LLM dari Nol

> **Bongkar rahasia "otak" LLM, liat gimana cara dia mikir & generate teks, step by step, dari nol banget!**

## 📖 Apa ini?

Proyek ini adalah sebuah **Interactive Educational Web App** yang dibuat khusus untuk memahami cara kerja _Large Language Model_ (LLM) seperti GPT, tapi dalam skala mikro.

Beda dengan tutorial lain yang biasanya langsung pake library canggih kayak PyTorch atau TensorFlow, di sini kita belajar **"The Hard Way"** alias **Pure Math & Logic**.
Kita membangun Neural Network sederhana dari nol menggunakan JavaScript murni (Vanilla JS) untuk melatih model bahasa kecil langsung di browser kamu!

## ✨ Fitur Utama

- **Tanpa Library AI**: Tidak ada TensorFlow.js, tidak ada ONNX. Semua logika matriks (Matrix multiplication), aktivasi (ReLU, Softmax), dan Backpropagation ditulis manual dari nol.
- **Visualisasi Interaktif**: Setiap konsep abstrak divisualisasikan dengan animasi yang smooth:
  - **Tokenization**: Lihat bagaimana teks diubah jadi angka.
  - **Embedding**: Lihat representasi vektor warna-warni dari setiap karakter.
  - **Forward Pass**: Animasi aliran data dari input -> hidden layer -> output prediksi.
- **Training Real-time**: Latih model AI langsung di browser dan lihat grafik _Loss_ menurun secara real-time.
- **Generate Teks**: Setelah training selesai, coba generate teks baru berdasarkan pola yang sudah dipelajari AI.
- **Desain Premium**: Tampilan Modern Dark Mode dengan animasi glassmorphism yang memanjakan mata.

## 🛠️ Teknologi yang Digunakan

Proyek ini dibangun menggunakan stack web dasar tanpa framework frontend yang berat:

- **HTML5**: Struktur konten yang semantik.
- **CSS3**: Styling modern dengan CSS Variables, Flexbox/Grid, dan animasi Keyframes.
- **Vanilla JavaScript (ES6+)**: Otak dari aplikasi ini. Menangani semua logika UI dan komputasi Neural Network (Forward pass, Backward pass, SGD Optimization).
- **Chart.js**: Satu-satunya library eksternal, digunakan hanya untuk menggambar grafik Loss saat training.

## 📂 Struktur Project

- **`index.html`**: Halaman utama yang berisi struktur _storytelling_ per bab.
- **`style.css`**: Semua styling untuk membuat tampilan visual yang menarik dan animasi interaktif.
- **`app.js`**: "Mesin" utama project ini. Berisi implementasi Neural Network manual:
  - Inisialisasi Weights (Gaussian Random)
  - Training Loop (Epochs)
  - Math Functions (`dot`, `outer`, `transpose`, `softmax`, `relu`)
  - DOM Manipulation untuk sinkronisasi visual.
- **`akusukakoding.py`**: Prototype logika dalam bahasa Python. Ini adalah referensi matematika murni yang kemudian di-porting ke JavaScript di `app.js`. Anda bisa menjalankan file ini untuk melihat proses yang sama berjalan di terminal.

## 🚀 Cara Menjalankan

Karena proyek ini hanya menggunakan HTML, CSS, dan JS statis, cara menjalankannya sangat mudah:

1.  **Clone atau Download** repository ini.
2.  Buka file `index.html` menggunakan browser modern (Chrome, Edge, Firefox, Safari).
3.  **Rekomendasi**: Gunakan ekstensi **"Live Server"** di VS Code untuk pengalaman terbaik (agar tidak ada masalah CORS policy pada beberapa browser, meskipun secara umum file lokal pun bisa jalan).

## 📚 Kurikulum (Bab per Bab)

1.  **Prolog**: Intro masalah, input teks latihan.
2.  **Bab 1 (Tokenization)**: Mengubah karakter menjadi ID numerik (Angka).
3.  **Bab 2 (Embedding)**: Mengubah ID menjadi Vector (Deretan angka yang kaya makna).
4.  **Bab 3 (Training Data)**: Membuat pasangan soal-jawaban (Input 3 huruf -> Target 1 huruf).
5.  **Bab 4 (Neural Network)**: Penjelasan arsitektur Input Layer, Hidden Layer, dan Output Layer.
6.  **Bab 5 (Forward Pass)**: Simulasi AI melakukan prediksi dengan bobot awal (masih random/bego).
7.  **Bab 6 (Training)**: Proses melatih AI (Forward -> Hitung Error -> Backward -> Update Bobot) ribuan kali.
8.  **Bab 7 (Generate)**: Menguji hasil kecerdasan buatan kita untuk menulis teks!

## 👨‍💻 Author

Dibuat dengan ❤️ dan ☕. Code logic Python asli (`akusukakoding.py`) berfungsi sebagai blueprint matematika, yang kemudian diterjemahkan menjadi pengalaman web interaktif yang epik ini.

---

_Selamat belajar dan semoga tercerahkan bahwa AI itu sebenarnya cuma matematika (aljabar linear) yang diajarin sopan santun!_ 🤖
