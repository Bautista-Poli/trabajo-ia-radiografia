# Chatbot de turnos de radiología: NLU con LLM (Ollama + RAG)

Modelo basado en LLM del Trabajo Final de Ingeniería Informática. Extrae los datos del paciente (DNI, estudio, obra social, sede) de mensajes en lenguaje natural, usando un LLM local servido por [Ollama](https://ollama.ai) con RAG TF-IDF sobre terminología SNOMED CT.

## Estructura

```
modelos/llm/   ← código del modelo, tests y evaluación (ver modelos/llm/README.md)
CSVs/          ← datos de referencia: estudios SNOMED, coberturas, preparaciones
```

## Requisitos

- Python 3.12
- Ollama corriendo en `localhost:11434` con el modelo `qwen2.5:7b`:
  ```bash
  ollama pull qwen2.5:7b
  ```

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

Ejecutar desde la raíz del repositorio:

```bash
python -m modelos.llm.playground_llm
```

Los detalles del modelo, los tests y la evaluación están en [`modelos/llm/README.md`](modelos/llm/README.md).
