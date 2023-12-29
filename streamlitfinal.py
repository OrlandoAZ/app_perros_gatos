# pip install streamlit
## pip install pillow
#pip install --upgrade keras
#pip install --upgrade tensorflow
#pip install opencv-python
#pip install matplotlib
#pip install scikit-image

import streamlit as st
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage import io

# Cargar el modelo de Keras
model = load_model('keras_model.h5')

# Crear el directorio temporal si no existe
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

# Función de preprocesamiento y clasificación de la imagen
def clasificar_imagen(imagen_path):
    img_array = io.imread(imagen_path) / 255.0
    img_resized = ImageOps.fit(Image.fromarray((img_array * 255).astype(np.uint8)), (224, 224), Image.Resampling.LANCZOS)
    img_array_resized = np.asarray(img_resized)
    normalized_image_array = (img_array_resized.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    pred = model.predict(data)[0]
    return pred[0]

# Configuración para desactivar la advertencia
st.set_option('deprecation.showPyplotGlobalUse', False)

# Encabezado
st.title('MODELO PARA PERROS Y GATOS')

# Sección de carga de imágenes
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar temporalmente el archivo
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Mostrar la imagen
    lena_rgb = io.imread(temp_path) / 255.0

    # Crear una figura de Matplotlib y pasarla a st.pyplot()
    fig, ax = plt.subplots()
    ax.imshow(lena_rgb)
    ax.set_title("Imagen seleccionada")
    ax.axis('off')
    st.pyplot(fig)

    # Clasificar la imagen
    pred = clasificar_imagen(temp_path)

    # Mostrar resultado de la clasificación
    if pred < 0.5:
        st.markdown(f'<p style="color: green; font-size: 24px;">La imagen es un perro con una probabilidad de {round(1-pred, 4)}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color: red; font-size: 24px;">La imagen es un gato con una probabilidad de {round(pred, 4)}</p>', unsafe_allow_html=True)
    # Eliminar el archivo temporal después de usarlo
    os.remove(temp_path)
