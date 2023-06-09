import tkinter as tk
from tkinter import filedialog
from tkinter import Text
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from ttkthemes import ThemedTk
import textwrap

# Metni çerçeve içine al
text = """CNV, retina tabakasındaki anormal damar büyümesiyle 
ilişkilendirilen hastalıklar arasında yer alır 
ve yaşa bağlı makula dejenerasyonu ve miyopik
 korioretinal atrofi gibi ciddi görme kaybına neden olabilir.
Anti-Vasküler Endotel Büyüme Faktörü 
(anti-VEGF) ilaçları veya fotodinamik tedavi gibi 
yöntemler kullanılarak CNV'ye bağlı görme kaybının kontrol 
altına alınması ve görsel fonksiyonun korunması hedeflenir.
"""

text1 = """DME, makula adı verilen gözün merkezi bölgesinde 
sıvı birikimine neden olan bir durumdur. Bu sıvı birikimi, 
retina tabakasını etkiler ve görme kaybına yol açabilir. 
DME genellikle diyabetik retinopatinin ilerleyen aşamalarında 
ortaya çıkar.Erken teşhis ve tedavi, DME'nin ilerlemesini 
yavaşlatabilir ve görme kaybını önleyebilir.
"""

text2 = """Drusenler, gözdeki sarımsı beyaz lekeler veya 
birikimler olarak tanımlanır. Bunlar, retina pigment epiteli 
(RPE) tabakası ve Bruch membranı arasında birikmiş maddelerdir. 
Drusenler genellikle makula bölgesinde bulunur ve görme kaybına 
yol açabilir.
"""
wrapped = textwrap.fill(text, width=80)
cnv_text = f"```\n{wrapped}\n```"
wrapped_text1 = textwrap.fill(text1, width=80)
dme_text = f"```\n{wrapped_text1}\n```"
wrapped_text2 = textwrap.fill(text2, width=80)
drusen_text = f"```\n{wrapped_text2}\n```"

# Arayüzü oluşturma
root = ThemedTk(theme="arc")
root.title("Görüntü Sınıflandırma")
root.geometry("400x400")

# Görüntü boyutları
img_width, img_height = 150, 150
channels = 3

# Sınıf isimleri
class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Model dosyasının yolu
model_file_path = r"C:\Users\elifs\PycharmProjects\pythonProject3\pythonProject3\second_model.h5"

# Modelin varlığını kontrol etme
if not os.path.exists(model_file_path):
    result_label = tk.Label(root, text="Model bulunamadı.")
    result_label.pack(pady=10)
    root.mainloop()
    raise ValueError("Model bulunamadı")

# Modeli yükleme
model = load_model(model_file_path)



# Görüntü yükleme fonksiyonu
def load_image():
    global x, image  # x ve image'i global olarak tanımla
    file_path = filedialog.askopenfilename(
        filetypes=[("JPEG Files", "*.jpeg"), ("JPG Files", "*.jpg"), ("PNG Files", "*.png")])
    if file_path:
        if not file_path.lower().endswith((".jpeg", ".jpg", ".png")):
            result_label.config(text="Lütfen .jpeg, .jpg veya .png uzantılı bir resim yükleyiniz.")
            return
        image = Image.open(file_path).resize((img_width, img_height))
        image = image.convert("RGB")
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        x = preprocess_input(image_array)

# Sonucu gösterme fonksiyonu
def create_text_widget(parent, text, width, height):
    text_widget = Text(parent, wrap="word", width=width, height=height, padx=10, pady=10, relief="solid", borderwidth=2, highlightthickness=0)
    text_widget.insert("1.0", text)
    text_widget.config(state="disabled", bg="white", fg="black", bd=2, highlightbackground="#4A6A7D", highlightcolor="#4A6A7D")
    return text_widget

def show_result():
    if x is not None:  # Eğer x tanımlıysa
        predictions = model.predict(x)[0]
        predicted_class = class_names[np.argmax(predictions)]
        predicted_prob = np.max(predictions)

        if predicted_class == "DME":
            result_text.set(f"OCT görüntü sonucunuz %{round(predicted_prob * 100, 2)} oranında {predicted_class} hastalığıdır. \nEn kısa sürede doktorunuza danışmanızı öneririz.")
            dme_text_widget = create_text_widget(canvas, text1, 64, 8)
            dme_text_widget.place(x=310, y=500)
        elif predicted_class == "CNV":
            result_text.set(f"OCT görüntü sonucunuz %{round(predicted_prob * 100, 2)} oranında {predicted_class} hastalığıdır. \nEn kısa sürede doktorunuza danışmanızı öneririz.")
            cnv_text_widget = create_text_widget(canvas, text, 64, 8)
            cnv_text_widget.place(x=310, y=500)
        elif predicted_class == "DRUSEN":
            result_text.set(f"OCT görüntü sonucunuz %{round(predicted_prob * 100, 2)} oranında {predicted_class} hastalığıdır. \nEn kısa sürede doktorunuza danışmanızı öneririz.")
            drusen_text_widget = create_text_widget(canvas, text2, 64, 8)
            drusen_text_widget.place(x=310, y=500)
        elif predicted_class == "NORMAL":
            result_text.set(f"OCT görüntü sonucunuz %{round(predicted_prob * 100, 2)} oranında sağlıklıdır. \nEn kısa sürede teyit amaçlı doktorunuza danışmanızı öneririz.")

        # Görüntüyü ekrana ekleme
        display_image = ImageTk.PhotoImage(image)
        image_label.config(image=display_image)
        image_label.image = display_image
    else:
        result_text.set("Lütfen önce bir görüntü yükleyin.")

# Arka plan rengi
background_color = "#FFFFFF"  # Beyaz

# Arka plana net bir renk eklemek için Canvas kullanımı
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight(), bg=background_color)
canvas.pack(fill="both", expand=True)

# Arka plana bir görsel eklemek için Label kullanımı
background_image = Image.open("background.png")
background_image = background_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(canvas, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# "Görüntü Yükle" butonunu oluşturma
load_button = tk.Button(canvas, text=" Görüntü Yükle ", command=load_image, bg="#5F7E95", fg="white", font=("Arial", 12, "bold"), relief="solid", borderwidth=0, highlightthickness=9)
load_button.place(x=100, y=100)
load_button.config(highlightbackground="#4A6A7D", highlightcolor="#4A6A7D")

# Sonucu göster butonunu oluşturma
result_button = tk.Button(canvas, text="Sonucu Göster", command=show_result, bg="#5F7E95", fg="white", font=("Arial", 12, "bold"), relief="solid", borderwidth=0, highlightthickness=9)
result_button.place(x=100, y=170)
result_button.config(highlightbackground="#4A6A7D", highlightcolor="#4A6A7D")

# Sonucu gösterecek etiket
result_text = tk.StringVar()
result_label = tk.Label(canvas, textvariable=result_text, bg=background_color, font=("Arial", 16))  # Arka plan rengini beyaz olarak ayarla
result_label.place(x=300, y=350)  # Sonucu gösteren etiketin konumunu güncelle

# Görüntüyü gösterecek etiket
image_label = tk.Label(canvas, bg=background_color)  # Arka plan rengini beyaz olarak ayarla
image_label.place(x=100, y=350)

# Arayüzü başlatma
root.mainloop()
