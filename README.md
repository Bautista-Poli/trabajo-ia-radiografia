# Turnos de radiografía en lenguaje natural: LLM local + RAG sobre SNOMED CT

Trabajo final de la materia **Inteligencia Artificial**.

## Integrantes

- Alvarez Poli, Bautista
- del Valle, Facundo
- Cardinale, Ignacio
- Kloster, Agustín

---

## Contexto

Cualquiera que haya sacado un turno para una radiografía conoce el problema: uno escribe como habla ("placa de la panza", "rx de rodilla") y el sistema de turnos le devuelve una lista de cientos de estudios con nombres técnicos que casi nadie entiende. Este proyecto nace de esa fricción.

Para la materia lo planteamos como un problema de IA que se pueda medir: tomar cada mensaje del paciente y asignarlo a un estudio oficial del catálogo, o darnos cuenta de que falta información para hacerlo.

- **Dominio:** salud, más precisamente la gestión administrativa de turnos de radiografía.
- **Cómo se resuelve hoy:** con menús en forma de árbol de decisión o con un buscador que exige el nombre exacto. Si el paciente no conoce la terminología formal, termina hablando con un administrativo.
- **Por qué es difícil:** solo en radiografía simple, la Edición Argentina de SNOMED CT tiene 481 procedimientos que se pueden pedir por turno. Además, mucha gente escribe con errores de tipeo y manda en un mismo mensaje el estudio, el DNI y la obra social.

## Formulación

En una frase: queremos ayudar a los pacientes que piden turnos de radiología por mensajería y chocan con sistemas que les exigen conocer el nombre técnico de entre 481 estudios. Para eso exploramos una solución apoyada en la terminología oficial SNOMED CT Argentina y en variantes coloquiales de cada estudio. El sistema responde con un estudio del catálogo, con "ambiguo" (junto con la aclaración que hay que pedir) o con "sin estudio".

Vamos a considerar que funciona si, sobre mensajes que nunca vio, acierta más que un buscador por similitud sin modelo de lenguaje, y si lo hace sin asignar estudios equivocados. Todo esto con tres condiciones:

1. que corra localmente, para que los datos de salud no salgan a terceros;
2. que se limite a radiografía simple;
3. que responda en pocos segundos.

## Definición del problema de IA

Técnicamente, es una clasificación multiclase con muchísimas clases y muy pocos ejemplos de cada una, a la que se suma la extracción de otros datos del mismo mensaje.

| Elemento | Definición |
|---|---|
| Usuario principal | El paciente que pide un turno por mensajería |
| Usuario secundario | El personal administrativo, que recibe la solicitud ya ordenada |
| Entrada | El mensaje tal como lo escribió el paciente, la última pregunta que hizo el sistema y los datos que ya se completaron |
| Salida | Uno de 481 estudios SNOMED, `AMBIGUO` (con tipo: falta lado, falta segmento, zona amplia, falta zona) o `null`; además DNI, cobertura y sede |
| Tipo de problema | Clasificación supervisada multiclase + extracción de entidades, resuelta con recuperación + generación restringida |
| Métrica principal | Exactitud estricta (el estudio exacto), con F1 de `AMBIGUO` y de `null` como apoyo |
| Criterio de éxito | Superar al mejor baseline sin modelo de lenguaje en mensajes nuevos, sin perder precisión en los estudios |

## Datos y auditoría

Conviene decirlo de entrada: trabajamos con terminología oficial y con variantes sintéticas. Todavía no tenemos mensajes reales de pacientes, y esa es la mayor limitación del proyecto en cuanto a representatividad.

- **Catálogo de estudios:** SNOMED CT Edición Argentina (2026-02-20). De 734 conceptos nos quedamos con 547 procedimientos, y de ellos con los 481 que se pueden pedir por turno.
- **Variantes coloquiales:** unas 3.300, que generamos con un modelo de lenguaje en español rioplatense (alrededor de 7 por estudio).
- **Tablas auxiliares:** 47 coberturas médicas con su código RNOS y una tabla con la preparación que requiere cada estudio.
- **Conjuntos de evaluación:** una suite de 80 casos para desarrollo y un held-out de 60 mensajes nuevos, todos etiquetados con el estudio exacto. Hay además un segundo held-out de 35 mensajes, reservado para una única evaluación final.

## Arquitectura

```
mensaje del paciente
      │
      ▼
Recuperación TF-IDF (n-gramas de caracteres 3-5)  ──►  top-5 estudios candidatos del catálogo
      │
      ▼
LLM local (qwen2.5:7b vía Ollama, temperatura 0)
  · prompt con reglas de ambigüedad + ejemplos few-shot de casos fuera de dominio
  · salida JSON restringida por schema: el estudio solo puede ser un candidato, AMBIGUO o null
      │
      ▼
Post-proceso determinístico
  · validación de DNI (7-8 dígitos)
  · normalización del estudio al preferred_term del catálogo
  · regla de lateralidad: zona par sin lado → AMBIGUO (falta_lado)
      │
      ▼
estudio | AMBIGUO (+ tipo) | null   +   DNI, obra social, sede
```

> Nota: esta es la configuración evaluada, implementada en `modelos/llm/evaluacion/comparar_configuraciones.py`. El playground conversacional (`ExtractorLLM`) todavía usa la configuración base (recuperación por palabras, top-3, sin salida restringida). Unificar los dos está entre los próximos pasos.

## Resultados

Exactitud estricta del estudio sobre la suite de desarrollo (80 casos) y el held-out 1 (60 mensajes que el sistema nunca vio):

| Configuración | Exactitud suite | Exactitud held-out |
|---|---|---|
| TF-IDF por palabras (sin LLM) | 0.21 | 0.32 |
| TF-IDF por n-gramas de caracteres (sin LLM) | 0.24 | 0.45 |
| TF-IDF por n-gramas con umbrales calibrados (sin LLM) | 0.69 | 0.53 |
| LLM sin recuperación | 0.51 | 0.22 |
| LLM + recuperación por palabras | 0.69 | 0.65 |
| LLM + recuperación por n-gramas | 0.72 | 0.70 |
| + salida restringida a los candidatos | 0.72 | 0.72 |
| + ejemplos de casos fuera de dominio (**elegida**) | 0.79 | 0.75 |
| + regla de lateralidad post-LLM | 0.84 | 0.80 |

La configuración elegida supera al mejor baseline sin LLM (0.53) por 22 puntos en el held-out, con F1 de `null` = 1.00 y una latencia mediana de unos 4 segundos por mensaje en una MacBook Pro M4 Pro. El detalle completo (P/R/F1 macro, F1 de `AMBIGUO` por tipo, recall de la recuperación y latencias) está en [`modelos/llm/evaluacion/resultados/comparacion_resumen.md`](modelos/llm/evaluacion/resultados/comparacion_resumen.md).

## Estructura del repositorio

```
modelos/llm/               ← código del modelo (ver modelos/llm/README.md)
modelos/llm/tests/         ← suite de tests NLU y tests unitarios
modelos/llm/evaluacion/    ← sets etiquetados, scripts de evaluación y resultados
CSVs/                      ← catálogo SNOMED con variantes, coberturas y preparaciones
```

## Instalación y uso

Requisitos: Python 3.12 y [Ollama](https://ollama.ai) corriendo en `localhost:11434`.

```bash
ollama pull qwen2.5:7b

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Desde la raíz del repositorio:

```bash
python -m modelos.llm.playground_llm                          # conversación interactiva
python -m modelos.llm.tests.test_suite                         # suite de tests NLU (requiere Ollama)
python -m unittest modelos.llm.tests.test_lateralidad -v       # tests unitarios (sin Ollama)
python -m modelos.llm.evaluacion.comparar_configuraciones      # tabla de resultados
```

Los detalles del modelo, los catálogos y el protocolo de evaluación están en [`modelos/llm/README.md`](modelos/llm/README.md). El registro de avance del proyecto está en [`docs/avance_proyecto.docx`](docs/avance_proyecto.docx).
