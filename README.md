# Profesor de Inglés con IA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/TU_USUARIO/profesor-ingles-ia.svg)](https://github.com/TU_USUARIO/profesor-ingles-ia/issues)
[![GitHub stars](https://img.shields.io/github/stars/TU_USUARIO/profesor-ingles-ia.svg)](https://github.com/TU_USUARIO/profesor-ingles-ia/stargazers)

![image_fx_](https://github.com/user-attachments/assets/6d6ea8c6-237d-41dd-9505-b215657b2d92)

## Descripción

Este proyecto utiliza modelos de inteligencia artificial de última generación para crear un tutor de inglés interactivo. Combina las capacidades de **Seamless Communication** de Meta AI para la traducción y síntesis de voz, con la potencia de **LangChain** y los modelos de lenguaje generativos (LLMs) de **Ollama** o **Google AI** (como **Gemini**) para crear una experiencia de aprendizaje conversacional.

Los usuarios pueden practicar su inglés hablando o escribiendo, recibiendo respuestas en audio (tanto en inglés como en español) y texto.

## Características

-   **Traducción de voz a texto:** Convierte la voz del usuario (en español o inglés) a texto utilizando el modelo SeamlessM4Tv2 de Meta AI.
-   **Interacción con LLM:** Utiliza LangChain para gestionar la conversación y un LLM (Ollama o Gemini de Google AI) como "cerebro" del tutor, proporcionando respuestas educativas y contextuales.
-   **Generación de audio:** Convierte las respuestas de texto del LLM a audio en inglés y español utilizando SeamlessM4Tv2.
-   **Interfaz gráfica amigable:** Desarrollado con Gradio para facilitar la interacción del usuario.
-   **Memoria conversacional:** Recuerda las últimas interacciones para mantener un contexto en la conversación.
-   **Facilidad de uso:** Con una interfaz intuitiva que puede ser usada por cualquier persona.

## Herramientas de Google Utilizadas

Este proyecto utiliza las siguientes herramientas de Google:

-   **Gemma2** Un modelo de AI Open Source que se ejecuta en local, para mi caso uso Gemma 2 de 9B parametros
-   **Ollama:** para ejecutar modelos de lenguaje generativo, si bien no es una herramienta de google, lo uso para ejecutar modelos como Gemma 2 en local
-   **Google AI (Gemini):** Un potente modelo de lenguaje generativo utilizado como el LLM principal para la interacción conversacional (opcional, se puede usar Ollama).
-   **Google Colab:** (Opcional) Entorno de Jupyter Notebooks en la nube que facilita la experimentación y el desarrollo con acceso a GPUs.

Para usar **Gemini**, necesitas una API key de Google AI Studio. Puedes obtenerla aquí: [Google AI Studio](https://aistudio.google.com/).

Gemma lo puedes encontrar en [Huggingface](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) o en [Ollama](https://ollama.com/library/gemma2)

## Requisitos Previos

-   Python 3.9 o superior
-   pip (gestor de paquetes de Python)
-   Una cuenta de GitHub (para clonar el repositorio)
-   (Opcional) Una API key de Google AI Studio si quieres usar Gemini, de no ser asi usar Gemma2

## Instalación

1. **Clona el repositorio:**

    ```bash
    git clone https://github.com/alarcon7a/english-teacher-ai.git
    cd english-teacher-ai
    ```

2. **Crea un entorno virtual (recomendado):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\\Scripts\\activate
    ```

3. **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuración de la API Key de Google AI Studio (Opcional)

Si vas a usar Gemini, exporta tu API key como una variable de entorno:

```bash
export GOOGLE_API_KEY="TU_API_KEY_AQUÍ"
```
O si estas en Colab podrias hacerlo asi:

```bash
from google.colab import userdata
import os
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
```
## Uso
1. **Ejecuta Gemma con Ollama**
```bash
   ollama run gemma2
```
2. **Ejecuta la aplicación:**
```bash
   python teacher.py
```
Esto iniciará la interfaz de Gradio en tu navegador.

3. Podrias tambien ejecutar el notebook [English_teacher_with_gemini.ipynb ](https://github.com/alarcon7a/english-teacher-ai/blob/main/English_teacher_with_gemini.ipynb) paso a paso como guia para el uso de Gemini como tutor de ingles.

### Interactúa con el tutor:

Haz clic en el botón de grabación o sube un archivo de audio en español o inglés.

El tutor transcribirá tu voz a texto, procesará la entrada con el LLM y generará una respuesta.

Escucharás la respuesta en audio (español e inglés) y verás la transcripción en la interfaz.

## Contribuciones

¡Las contribuciones son bienvenidas! Si quieres mejorar este proyecto, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama con tu feature o bugfix: git checkout -b feature/mi-nueva-feature
3. Realiza tus cambios y haz commit: git commit -m "Añadir mi nueva feature"
4. Haz push a tu rama: git push origin feature/mi-nueva-feature
5. Abre un Pull Request en GitHub.
