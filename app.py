import streamlit as st
import anthropic
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re, base64, os

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDE Social Advisor",
    page_icon="🌿",
    layout="wide"
)

# ── TRANSLATIONS ───────────────────────────────────────────────────────────────
LANG = {
    "es": {
        "hero_badge": "🌿 EDE Fundazioa · Reto BGT",
        "hero_title": "Asesor de Apoyo Social",
        "hero_desc": "Selecciona un modo — describe un caso de cliente o formula una pregunta de investigación.<br>El sistema recupera casos históricos similares y genera un informe personalizado con Claude.",
        "mode_label": "⚙️ Seleccionar Modo",
        "mode_case": "🟢 Asesoría de Casos",
        "mode_prog": "🔵 Investigación de Programa",
        "mode_case_info": "**Modo Asesoría de Casos** — Rellena el formulario con los datos del cliente. El sistema anonimizará automáticamente antes de buscar casos similares y generar un informe.",
        "mode_prog_info": "**Modo Investigación de Programa** — Haz una pregunta amplia sobre los registros históricos. Claude analizará patrones y presentará conclusiones a nivel de programa.",
        "step1_case": "✍️ Paso 1 — Introducir datos del cliente",
        "step1_prog": "🔵 Paso 1 — Escribe tu pregunta de investigación",
        "tab_form": "📋 Formulario estructurado",
        "tab_upload": "📎 Subir archivo de caso",
        "form_ref": "Referencia del caso (auto-generada)",
        "form_name": "Nombre del cliente",
        "form_name_hint": "Nombre completo tal como aparece en el expediente",
        "form_age": "Edad",
        "form_nationality": "Nacionalidad",
        "form_situation": "Situación actual",
        "form_situation_hint": "Estado legal, vivienda, idioma, situación familiar...",
        "form_background": "Antecedentes e historial",
        "form_background_hint": "Cuándo llegó, educación, familia, experiencia laboral, motivación...",
        "form_risks": "Factores de riesgo identificados",
        "form_risks_hint": "Barreras, vulnerabilidades, preocupaciones de salud mental...",
        "form_intervention": "Última intervención",
        "form_intervention_hint": "Qué servicios o apoyos se han ofrecido hasta ahora...",
        "form_outcome": "Resultado / Estado",
        "form_outcome_hint": "Estado actual del caso y qué se busca...",
        "anonymise_btn": "🔒 Anonimizar con IA",
        "anonymise_spinner": "Claude está anonimizando el caso...",
        "anon_preview_title": "📋 Paso 2 — Revisión del texto anonimizado",
        "anon_preview_desc": "El texto siguiente ha sido anonimizado automáticamente. Revisa los cambios resaltados y edita si es necesario antes de enviar.",
        "anon_edit_label": "Editar texto anonimizado si es necesario",
        "run_case": "✅ Confirmar y buscar casos similares",
        "run_prog": "🔍 Buscar Registros y Generar Informe de Investigación",
        "reset": "🔄 Nuevo Caso",
        "warn_empty_form": "Por favor, rellena al menos el nombre, la edad y la situación actual antes de anonimizar.",
        "warn_empty_prog": "Por favor, introduce una pregunta de investigación antes de continuar.",
        "warn_anon_first": "Por favor, anonimiza el caso primero antes de enviarlo.",
        "log_encoding": "Codificando el perfil del caso en un vector de búsqueda...",
        "log_encoded": "✅ Perfil del caso codificado correctamente.",
        "log_searching": "Buscando registros similares en la colección Qdrant...",
        "log_found": "✅ Recuperados {} registros de Qdrant.",
        "spinner_case": "Claude está leyendo los registros y redactando tu informe de asesoría...",
        "spinner_prog": "Claude está analizando los registros y redactando tu informe de investigación...",
        "step2_results": "📂 Paso 3 — Registros históricos más similares",
        "step3_case": "📋 Paso 4 — Informe de Asesoría de Claude",
        "step3_prog": "📊 Informe de Investigación de Programa",
        "download_case": "⬇️ Descargar Informe de Asesoría (.txt)",
        "download_prog": "⬇️ Descargar Informe de Investigación (.txt)",
        "file_case": "ede_informe_asesoria.txt",
        "file_prog": "ede_informe_investigacion.txt",
        "sidebar_title": "🌿 EDE Social Advisor",
        "sidebar_sub": "BGT 4ª Edición · EDE Fundazioa<br>Inteligencia de Casos con RAG",
        "sidebar_how": "**📋 Cómo usar esta herramienta**",
        "sidebar_steps": [
            ("<strong>① Selecciona un modo</strong>", "Asesoría de Casos para clientes individuales. Investigación de Programa para consultas amplias."),
            ("<strong>② Rellena el formulario</strong>", "Introduce los datos del cliente de forma natural en cada campo."),
            ("<strong>③ Anonimiza con IA</strong>", "Claude elimina automáticamente todos los datos identificativos."),
            ("<strong>④ Revisa y confirma</strong>", "Comprueba el texto anonimizado y envía cuando estés listo."),
            ("<strong>⑤ Lee la asesoría</strong>", "Claude genera un informe estructurado basado en casos similares."),
        ],
        "sidebar_about": "**ℹ️ Privacidad y RGPD**",
        "sidebar_tip": "Esta herramienta incluye <strong>anonimización automática por IA</strong> antes de cualquier envío. Ningún dato identificativo llega a la base de datos Qdrant ni a Claude.<br><br>Desarrollado con: <strong>Streamlit · Qdrant · sentence-transformers · Claude API</strong>",
        "sidebar_footer": "EDE Fundazioa · BGT 4ª Edición · 2025",
        "no_records": "No se encontraron registros coincidentes. Tu colección Qdrant puede estar vacía.",
        "prog_placeholder": "Ejemplo: ¿Qué pasos de integración lingüística y marcos de apoyo psikológico hemos utilizado con éxito para mujeres solicitantes de asilo aisladas?",
        "prog_hint": "<strong>Modo Investigación de Programa</strong> — Haz una pregunta amplia sobre patrones, marcos o resultados en los registros históricos de EDE.",
        "case_sections": [
            ("FACTORES DE RIESGO Y BARRERAS", "⚠️ Factores de Riesgo y Barreras"),
            ("LECCIONES DE LOS REGISTROS HISTÓRICOS", "📖 Lecciones de los Registros Históricos"),
            ("INTERVENCIONES RECOMENDADAS", "✅ Intervenciones Recomendadas"),
            ("LAGUNAS DE INFORMACIÓN", "🔍 Lagunas de Información"),
        ],
        "prog_sections": [
            ("PATRONES CLAVE IDENTIFICADOS", "📊 Patrones Clave Identificados"),
            ("MARCOS Y ENFOQUES EN USO", "🧩 Marcos y Enfoques en Uso"),
            ("RECOMENDACIONES PARA EL DESARROLLO DEL PROGRAMA", "✅ Recomendaciones para el Desarrollo del Programa"),
        ],
        "case_fallback_title": "📋 Informe de Asesoría",
        "prog_fallback_title": "📊 Informe de Investigación",
        "download_header_case": "INFORME DE ASESORÍA DE APOYO SOCIAL — EDE FUNDAZIOA",
        "download_header_prog": "INFORME DE INVESTIGACIÓN DE PROGRAMA — EDE FUNDAZIOA",
        "download_input_label": "CASO ANONIMIZADO",
        "download_brief_label": "INFORME",
        "anon_system": """Eres un asistente especializado en anonimización de datos para una organización de servicios sociales en el País Vasco, España, que debe cumplir el RGPD.

Tu única tarea es reescribir el texto del caso sustituyendo todos los datos identificativos, manteniendo intacto todo el contenido clínico y social.

REGLAS DE ANONIMIZACIÓN (aplica todas sin excepción):
- Nombres propios de clientes → código de referencia ya asignado (por ejemplo: Cliente EDE-A7)
- Edad exacta → banda de edad (18-25, 26-35, 36-45, 46-55, 56-65, 65+)
- Nacionalidad específica → región geográfica amplia (África Occidental, Europa del Este, Oriente Medio, América Latina, Asia Central, etc.)
- Fechas exactas o períodos específicos → términos relativos ("recientemente", "en los últimos meses", "hace más de un año")
- Nombres de albergues, centros, calles o barrios específicos → términos genéricos ("centro de acogida local", "centro comunitario", "barrio de residencia")
- Nombres de trabajadores sociales, médicos u otros profesionales → roles genéricos ("trabajador/a social de referencia", "profesional sanitario")
- Nombres de familiares → roles familiares ("cónyuge", "hijo/a", "madre", "padre")
- Cualquier otro dato que pudiera identificar a una persona específica → término genérico equivalente

MANTÉN SIN CAMBIOS:
- Factores de riesgo y barreras (lenguaje, situación legal, salud mental, etc.)
- Historial de intervenciones y servicios recibidos
- Motivaciones, habilidades y puntos fuertes del cliente
- Resultado y estado del caso
- Todo el contenido clínico y de trabajo social

FORMATO DE RESPUESTA:
Devuelve ÚNICAMENTE el texto reescrito en el mismo formato que recibiste (campos etiquetados). No añadas explicaciones, notas ni comentarios. No uses comillas.""",
        "system_case": """Eres un asesor experimentado de trabajo social que apoya a los trabajadores de casos en EDE Fundazioa, en el País Vasco, España.

Recibirás:
1. Un nuevo perfil de cliente anonimizado escrito por un trabajador social
2. Extractos recuperados de informes de casos históricos similares al nuevo caso

Un perfil de cliente válido es cualquier entrada que describa la situación o necesidades de un individuo — aunque sea brevemente.

Solo rechaza si la entrada NO contiene ningún individuo identificable — por ejemplo, si es una pregunta de investigación general.

Cuando se envíe un perfil de cliente válido, estructura tu respuesta usando exactamente estas cuatro secciones etiquetadas:

FACTORES DE RIESGO Y BARRERAS
[Identifica los principales factores de riesgo y barreras del cliente a partir de su perfil]

LECCIONES DE LOS REGISTROS HISTÓRICOS
[Extrae lecciones específicas de los registros históricos recuperados — refiérete a ellos por número, p. ej. Registro 1, Registro 2]

INTERVENCIONES RECOMENDADAS
[Recomienda 2-3 opciones de intervención priorizadas con una justificación clara para cada una]

LAGUNAS DE INFORMACIÓN
[Señala los detalles que faltan y que el trabajador social debe recopilar antes de proceder]

Sé conciso, profesional y basado en evidencias. Escribe en español claro, adecuado para un trabajador social de primera línea.
NO inventes detalles del caso que no estén en los registros proporcionados o en el perfil del cliente.""",
        "system_prog": """Eres un analista de programa senior que apoya a EDE Fundazioa, una organización de servicios sociales en el País Vasco, España.

Recibirás:
1. Una pregunta de investigación a nivel de programa
2. Extractos recuperados de informes de casos históricos relevantes

Estructura tu respuesta usando exactamente estas tres secciones etiquetadas:

PATRONES CLAVE IDENTIFICADOS
[Resume los principales patrones, tendencias o temas relevantes para la pregunta de investigación]

MARCOS Y ENFOQUES EN USO
[Identifica marcos, métodos o enfoques específicos que aparecen en los registros]

RECOMENDACIONES PARA EL DESARROLLO DEL PROGRAMA
[Ofrece 2-3 recomendaciones accionables para el diseño de programas o la prestación de servicios de EDE]

Sé analítico, basado en evidencias y escribe en español claro.""",
    },

    "en": {
        "hero_badge": "🌿 EDE Fundazioa · BGT Challenge",
        "hero_title": "Social Support Advisor",
        "hero_desc": "Select a mode below — fill in a client form or ask a programme-level research question.<br>The system anonymises automatically, retrieves similar historical cases and generates a tailored brief via Claude.",
        "mode_label": "⚙️ Select Mode",
        "mode_case": "🟢 Case Advisory",
        "mode_prog": "🔵 Programme Research",
        "mode_case_info": "**Case Advisory Mode** — Fill in the structured client form. The system will anonymise automatically before searching similar cases and generating a report.",
        "mode_prog_info": "**Programme Research Mode** — Ask a broad question across all historical records. Claude will analyse patterns and surface programme-level insights.",
        "step1_case": "✍️ Step 1 — Enter client details",
        "step1_prog": "🔵 Step 1 — Enter your research question",
        "tab_form": "📋 Structured form",
        "tab_upload": "📎 Upload case file",
        "form_ref": "Case reference (auto-generated)",
        "form_name": "Client name",
        "form_name_hint": "Full name as it appears in the file",
        "form_age": "Age",
        "form_nationality": "Nationality",
        "form_situation": "Current situation",
        "form_situation_hint": "Legal status, housing, language, family situation...",
        "form_background": "Background & history",
        "form_background_hint": "When they arrived, education, family, work experience, motivation...",
        "form_risks": "Identified risk factors",
        "form_risks_hint": "Barriers, vulnerabilities, mental health concerns...",
        "form_intervention": "Last intervention",
        "form_intervention_hint": "What services or support have been offered so far...",
        "form_outcome": "Outcome / Status",
        "form_outcome_hint": "Current case status and what is being sought...",
        "anonymise_btn": "🔒 Anonymise with AI",
        "anonymise_spinner": "Claude is anonymising the case...",
        "anon_preview_title": "📋 Step 2 — Review anonymised text",
        "anon_preview_desc": "The text below has been automatically anonymised. Review the highlighted changes and edit if needed before submitting.",
        "anon_edit_label": "Edit anonymised text if needed",
        "run_case": "✅ Confirm and find similar cases",
        "run_prog": "🔍 Search Records & Generate Research Brief",
        "reset": "🔄 New Case",
        "warn_empty_form": "Please fill in at least the name, age, and current situation before anonymising.",
        "warn_empty_prog": "Please enter a research question before running.",
        "warn_anon_first": "Please anonymise the case first before submitting.",
        "log_encoding": "Encoding your case profile into a search vector...",
        "log_encoded": "✅ Case profile encoded successfully.",
        "log_searching": "Searching Qdrant collection for similar records...",
        "log_found": "✅ Retrieved {} records from Qdrant.",
        "spinner_case": "Claude is reading the records and writing your advisory brief...",
        "spinner_prog": "Claude is analysing the records and writing your research brief...",
        "step2_results": "📂 Step 3 — Most similar historical records",
        "step3_case": "📋 Step 4 — Claude Advisory Brief",
        "step3_prog": "📊 Programme Research Brief",
        "download_case": "⬇️ Download Advisory Brief (.txt)",
        "download_prog": "⬇️ Download Research Brief (.txt)",
        "file_case": "ede_advisory_brief.txt",
        "file_prog": "ede_research_brief.txt",
        "sidebar_title": "🌿 EDE Social Advisor",
        "sidebar_sub": "BGT 4th Edition · EDE Fundazioa<br>RAG-Powered Case Intelligence",
        "sidebar_how": "**📋 How to use this tool**",
        "sidebar_steps": [
            ("<strong>① Select a mode</strong>", "Case Advisory for individual clients. Programme Research for broad cross-case queries."),
            ("<strong>② Fill in the form</strong>", "Enter client details naturally in each field."),
            ("<strong>③ Anonymise with AI</strong>", "Claude automatically removes all identifying data."),
            ("<strong>④ Review and confirm</strong>", "Check the anonymised text and submit when ready."),
            ("<strong>⑤ Read the advisory</strong>", "Claude generates a structured brief based on similar cases."),
        ],
        "sidebar_about": "**ℹ️ Privacy & GDPR**",
        "sidebar_tip": "This tool includes <strong>automatic AI anonymisation</strong> before any submission. No identifying data reaches the Qdrant database or Claude.<br><br>Built with: <strong>Streamlit · Qdrant · sentence-transformers · Claude API</strong>",
        "sidebar_footer": "EDE Fundazioa · BGT 4th Edition · 2025",
        "no_records": "No matching records found. Your Qdrant collection may be empty.",
        "prog_placeholder": "Example: Based on our historical records, what specific language integration steps and psychological support frameworks have we successfully used for isolated female asylum seekers?",
        "prog_hint": "<strong>Programme Research Mode</strong> — Ask a broad question about patterns, frameworks, or outcomes across EDE's historical case records.",
        "case_sections": [
            ("RISK FACTORS & BARRIERS", "⚠️ Risk Factors & Barriers"),
            ("LESSONS FROM HISTORICAL RECORDS", "📖 Lessons from Historical Records"),
            ("RECOMMENDED INTERVENTIONS", "✅ Recommended Interventions"),
            ("INFORMATION GAPS", "🔍 Information Gaps"),
        ],
        "prog_sections": [
            ("KEY PATTERNS IDENTIFIED", "📊 Key Patterns Identified"),
            ("FRAMEWORKS & APPROACHES IN USE", "🧩 Frameworks & Approaches in Use"),
            ("RECOMMENDATIONS FOR PROGRAMME DEVELOPMENT", "✅ Recommendations for Programme Development"),
        ],
        "case_fallback_title": "📋 Advisory Brief",
        "prog_fallback_title": "📊 Research Brief",
        "download_header_case": "EDE FUNDAZIOA — SOCIAL SUPPORT ADVISORY BRIEF",
        "download_header_prog": "EDE FUNDAZIOA — PROGRAMME RESEARCH BRIEF",
        "download_input_label": "ANONYMISED CASE",
        "download_brief_label": "BRIEF",
        "anon_system": """You are a data anonymisation assistant for a social services organisation in the Basque Country, Spain, which must comply with GDPR.

Your sole task is to rewrite the case text by replacing all identifying data while keeping all clinical and social content intact.

ANONYMISATION RULES (apply all without exception):
- Client proper names → the already-assigned reference code (e.g. Client EDE-A7)
- Exact age → age band (18-25, 26-35, 36-45, 46-55, 56-65, 65+)
- Specific nationality → broad geographic region (West African, Eastern European, Middle Eastern, Latin American, Central Asian, etc.)
- Exact dates or specific time periods → relative terms ("recently", "in recent months", "over a year ago")
- Names of specific shelters, centres, streets or neighbourhoods → generic terms ("local reception centre", "community centre", "area of residence")
- Names of social workers, doctors or other professionals → generic roles ("referring social worker", "healthcare professional")
- Names of family members → family roles ("spouse", "child", "mother", "father")
- Any other data that could identify a specific person → equivalent generic term

KEEP UNCHANGED:
- Risk factors and barriers (language, legal status, mental health, etc.)
- Intervention history and services received
- Client's motivations, skills and strengths
- Case outcome and status
- All clinical and social work content

RESPONSE FORMAT:
Return ONLY the rewritten text in the same format you received (labelled fields). Do not add explanations, notes or comments. Do not use quotation marks.""",
        "system_case": """You are an experienced social work advisor supporting caseworkers at EDE Fundazioa in the Basque Country, Spain.

You will receive:
1. A new anonymised client profile written by a social worker
2. Retrieved excerpts from historical case reports similar to the new case

A valid client profile is any input that describes an individual's situation or presenting needs — even briefly.

Only refuse if the input contains NO identifiable individual at all — for example if it is a general research question.

When a valid client profile is submitted, structure your response using exactly these four labelled sections:

RISK FACTORS & BARRIERS
[Identify the client's key risk factors and barriers from their profile]

LESSONS FROM HISTORICAL RECORDS
[Draw specific lessons from the retrieved historical records — reference them by number e.g. Record 1, Record 2]

RECOMMENDED INTERVENTIONS
[Recommend 2–3 prioritised intervention options with clear rationale for each]

INFORMATION GAPS
[Note any missing details the social worker should gather before proceeding]

Be concise, professional, and evidence-based. Write in plain English suitable for a frontline social worker.
Do NOT invent case details that are not in the provided records or client profile.""",
        "system_prog": """You are a senior programme analyst supporting EDE Fundazioa, a social services organisation in the Basque Country, Spain.

You will receive:
1. A programme-level research question from a staff member or manager
2. Retrieved excerpts from historical case reports that are relevant to the question

Structure your response using exactly these three labelled sections:

KEY PATTERNS IDENTIFIED
[Summarise the main patterns, trends, or themes relevant to the research question]

FRAMEWORKS & APPROACHES IN USE
[Identify specific frameworks, methods, or approaches that appear in the records]

RECOMMENDATIONS FOR PROGRAMME DEVELOPMENT
[Offer 2–3 actionable recommendations for EDE's programme design or service delivery]

Be analytical, evidence-based, and write in plain English suitable for a programme manager or supervisor.""",
    },

    "eu": {
        "hero_badge": "🌿 EDE Fundazioa · BGT Erronka",
        "hero_title": "Gizarte Laguntzaren Aholkularia",
        "hero_desc": "Hautatu modu bat — bete bezeroaren inprimakia edo egin programa-mailako ikerketa galdera bat.<br>Sistemak automatikoki anonimizatzen du, antzeko kasu historikoak berreskuratzen ditu eta txosten pertsonalizatua sortzen du.",
        "mode_label": "⚙️ Hautatu Modua",
        "mode_case": "🟢 Kasuen Aholkularitza",
        "mode_prog": "🔵 Programaren Ikerketa",
        "mode_case_info": "**Kasuen Aholkularitza Modua** — Bete bezeroaren inprimaki egituratua. Sistemak automatikoki anonimizatuko du antzeko kasuak bilatu eta txostena sortu aurretik.",
        "mode_prog_info": "**Programaren Ikerketa Modua** — Egin galdera zabal bat erregistro historikoen gainean. Claudek ereduak aztertuko ditu eta programa-mailako ondorioak aurkeztuko ditu.",
        "step1_case": "✍️ 1. urratsa — Sartu bezeroaren datuak",
        "step1_prog": "🔵 1. urratsa — Idatzi zure ikerketa galdera",
        "tab_form": "📋 Inprimaki egituratua",
        "tab_upload": "📎 Kargatu kasuen fitxategia",
        "form_ref": "Kasuaren erreferentzia (automatikoki sortua)",
        "form_name": "Bezeroaren izena",
        "form_name_hint": "Izen osoa fitxategian agertzen den bezala",
        "form_age": "Adina",
        "form_nationality": "Nazionalitatea",
        "form_situation": "Egungo egoera",
        "form_situation_hint": "Egoera juridikoa, etxebizitza, hizkuntza, familia egoera...",
        "form_background": "Aurrekariak eta historia",
        "form_background_hint": "Noiz iritsi zen, hezkuntza, familia, lan esperientzia, motibazioa...",
        "form_risks": "Identifikatutako arrisku-faktoreak",
        "form_risks_hint": "Oztopoak, ahultasunak, osasun mental kezkak...",
        "form_intervention": "Azken esku-hartzea",
        "form_intervention_hint": "Orain arte zer zerbitzu edo laguntza eskaini zaion...",
        "form_outcome": "Emaitza / Egoera",
        "form_outcome_hint": "Kasuaren egungo egoera eta zer bilatzen den...",
        "anonymise_btn": "🔒 Anonimizatu AIrekin",
        "anonymise_spinner": "Claude kasua anonimizatzen ari da...",
        "anon_preview_title": "📋 2. urratsa — Anonimizatutako testua berrikusi",
        "anon_preview_desc": "Beheko testua automatikoki anonimizatu da. Berrikusi nabarmendu diren aldaketak eta editatu behar bada bidali aurretik.",
        "anon_edit_label": "Editatu anonimizatutako testua behar bada",
        "run_case": "✅ Berretsi eta bilatu antzeko kasuak",
        "run_prog": "🔍 Erregistroak Bilatu eta Ikerketa Txostena Sortu",
        "reset": "🔄 Kasu Berria",
        "warn_empty_form": "Mesedez, bete gutxienez izena, adina eta egungo egoera anonimizatu aurretik.",
        "warn_empty_prog": "Mesedez, sartu ikerketa galdera aurrera jarraitu aurretik.",
        "warn_anon_first": "Mesedez, anonimizatu kasua lehenik bidali aurretik.",
        "log_encoding": "Kasuaren profila bilaketa-bektore batean kodetzen...",
        "log_encoded": "✅ Kasuaren profila ongi kodetu da.",
        "log_searching": "Qdrant bilduman antzeko erregistroak bilatzen...",
        "log_found": "✅ {} erregistro berreskuratu dira Qdrantetik.",
        "spinner_case": "Claude erregistroak irakurtzen eta zure aholkularitza txostena idazten ari da...",
        "spinner_prog": "Claude erregistroak aztertzen eta zure ikerketa txostena idazten ari da...",
        "step2_results": "📂 3. urratsa — Antzeko erregistro historikoak",
        "step3_case": "📋 4. urratsa — Clauderen Aholkularitza Txostena",
        "step3_prog": "📊 Programaren Ikerketa Txostena",
        "download_case": "⬇️ Deskargatu Aholkularitza Txostena (.txt)",
        "download_prog": "⬇️ Deskargatu Ikerketa Txostena (.txt)",
        "file_case": "ede_aholkularitza_txostena.txt",
        "file_prog": "ede_ikerketa_txostena.txt",
        "sidebar_title": "🌿 EDE Social Advisor",
        "sidebar_sub": "BGT 4. Edizioa · EDE Fundazioa<br>RAG bidezko Kasuen Adimena",
        "sidebar_how": "**📋 Nola erabili tresna hau**",
        "sidebar_steps": [
            ("<strong>① Hautatu modua</strong>", "Kasuen Aholkularitza banakako bezeroentzat. Programaren Ikerketa galdera zabalentzat."),
            ("<strong>② Bete inprimakia</strong>", "Sartu bezeroaren datuak modu naturalean eremu bakoitzean."),
            ("<strong>③ Anonimizatu AIrekin</strong>", "Claudek automatikoki kentzen ditu datu identifikagarri guztiak."),
            ("<strong>④ Berrikusi eta berretsi</strong>", "Egiaztatu anonimizatutako testua eta bidali prest zaudenean."),
            ("<strong>⑤ Irakurri aholkularitza</strong>", "Claudek antzeko kasuetan oinarritutako txosten egituratu bat sortzen du."),
        ],
        "sidebar_about": "**ℹ️ Pribatutasuna eta DBEB**",
        "sidebar_tip": "Tresna honek <strong>IA bidezko anonimizazio automatikoa</strong> du edozein bidalketa baino lehen. Datu identifikagarririk ez da Qdrant datu-basera edo Claudera iristen.<br><br>Eraikita: <strong>Streamlit · Qdrant · sentence-transformers · Claude API</strong>",
        "sidebar_footer": "EDE Fundazioa · BGT 4. Edizioa · 2025",
        "no_records": "Ez da erregistro bat ere aurkitu. Zure Qdrant bilduma hutsik egon daiteke.",
        "prog_placeholder": "Adibidea: Gure erregistro historikoen arabera, zein hizkuntza-integrazio urrats eta laguntza psikologikorako esparru erabili ditugu arrakastaz isolatutako emakume asilo-eskatzaileekin?",
        "prog_hint": "<strong>Programaren Ikerketa Modua</strong> — Egin galdera zabal bat EDEren kasu-erregistro historikoen ereduei, esparruei edo emaitzei buruz.",
        "case_sections": [
            ("ARRISKU-FAKTOREAK ETA OZTOPOAK", "⚠️ Arrisku-faktoreak eta Oztopoak"),
            ("ERREGISTRO HISTORIKOETATIK IKASGAIAK", "📖 Erregistro Historikoetatik Ikasgaiak"),
            ("GOMENDATUTAKO ESKU-HARTZEAK", "✅ Gomendatutako Esku-hartzeak"),
            ("INFORMAZIO HUTSUNEAK", "🔍 Informazio Hutsuneak"),
        ],
        "prog_sections": [
            ("IDENTIFIKATUTAKO FUNTSEZKO EREDUAK", "📊 Identifikatutako Funtsezko Ereduak"),
            ("ERABILERAN DAUDEN ESPARRUAK ETA IKUSPEGIAK", "🧩 Erabileran dauden Esparruak eta Ikuspegiak"),
            ("PROGRAMAREN GARAPENERAKO GOMENDIOAK", "✅ Programaren Garapenerako Gomendioak"),
        ],
        "case_fallback_title": "📋 Aholkularitza Txostena",
        "prog_fallback_title": "📊 Ikerketa Txostena",
        "download_header_case": "EDE FUNDAZIOA — GIZARTE LAGUNTZAREN AHOLKULARITZA TXOSTENA",
        "download_header_prog": "EDE FUNDAZIOA — PROGRAMAREN IKERKETA TXOSTENA",
        "download_input_label": "ANONIMIZATUTAKO KASUA",
        "download_brief_label": "TXOSTENA",
        "anon_system": """Zu DBEBa bete behar duen Euskal Herriko gizarte-zerbitzu erakunde batentzako datuen anonimizazio laguntzaile bat zara.

Zure zeregina bakarra da kasu-testua berrido idaztea, datu identifikagarri guztiak ordezkatuz, eduki kliniko eta soziala osorik mantenduz.

ANONIMIZAZIO ARAUAK (guztiak aplikatu salbuespenik gabe):
- Bezeroen izen propioak → dagoeneko esleitutako erreferentzia kodea (adib. Bezeroa EDE-A7)
- Adin zehatza → adin banda (18-25, 26-35, 36-45, 46-55, 56-65, 65+)
- Nazionalitate zehatza → eskualde geografiko zabal (Afrika Mendebaldea, Europa Ekialdea, Ekialde Hurbila, Latin Amerika, Asia Erdiko, etab.)
- Data zehatzak edo aldi zehatzak → termino erlatiboak ("duela gutxi", "azken hilabeteetan", "duela urte bat baino gehiago")
- Albergue, zentro, kale edo auzo zehatz baten izenak → termino generikoak ("tokiko harrera zentroa", "komunitate zentroa", "bizileku eremua")
- Gizarte-langile, mediku edo beste profesionalen izenak → rol generikoak ("erreferentziako gizarte-langilea", "osasun profesionala")
- Familiakideen izenak → familia rolak ("ezkontidea", "seme-alaba", "ama", "aita")
- Pertsona zehatza identifikatu lezakeen beste edozein datu → baliokide termino generikoa

ALDATU GABE MANTENDU:
- Arrisku-faktoreak eta oztopoak
- Esku-hartze historia eta jasotako zerbitzuak
- Bezeroaren motibazioak, gaitasunak eta indarguneak
- Kasuaren emaitza eta egoera
- Eduki kliniko eta gizarte-lan guztia

ERANTZUN FORMATUA:
Itzuli SOILIK berridatzitako testua jaso duzun formatu berean (etiketatutako eremuak). Ez gehitu azalpenak, oharrak edo iruzkinak.""",
        "system_case": """Zu Euskal Herrian, Espainian, EDE Fundazioan lan egiten duten gizarte-langileei laguntzen dien gizarte-lanaren aholkulari esperientziadun bat zara.

Jasoko duzu:
1. Gizarte-langile batek idatzitako bezero berri baten profil anonimizatua
2. Kasu berriari antzeko erregistro historikoetatik ateratako zatiak

Estruktura ezazu zure erantzuna zehazki lau atal hauetan:

ARRISKU-FAKTOREAK ETA OZTOPOAK
[Identifikatu bezeroaren arrisku-faktore eta oztopo nagusiak]

ERREGISTRO HISTORIKOETATIK IKASGAIAK
[Atera ikasgai zehatzak berreskuratutako erregistro historikoetatik]

GOMENDATUTAKO ESKU-HARTZEAK
[Gomendatu 2-3 esku-hartze aukera lehentasunezkoak]

INFORMAZIO HUTSUNEAK
[Adierazi gizarte-langileak aurrera joan aurretik bildu beharreko xehetasun falta]""",
        "system_prog": """Zu EDE Fundazioa, Euskal Herriko gizarte-zerbitzu erakunde bati laguntzen dion programa-analistalari nagusi bat zara.

Estruktura ezazu zure erantzuna zehazki hiru atal hauetan:

IDENTIFIKATUTAKO FUNTSEZKO EREDUAK
[Laburtu ikerketa galderarekin zerikusia duten eredu nagusiak]

ERABILERAN DAUDEN ESPARRUAK ETA IKUSPEGIAK
[Identifikatu erregistroetan agertzen diren esparru eta metodoak]

PROGRAMAREN GARAPENERAKO GOMENDIOAK
[Eskaini 2-3 gomendio ekintza-bideragarri EDEren programa-diseinurako]""",
    },
}

# ── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #F4F7F4; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 900px; }
.ede-hero { background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%); border-radius: 18px; padding: 40px 36px 32px; margin-bottom: 32px; text-align: center; box-shadow: 0 4px 24px rgba(27,67,50,0.18); }
.ede-hero h1 { font-family: 'DM Serif Display', serif; color: #D8F3DC; font-size: 34px; margin: 0 0 10px 0; letter-spacing: -0.5px; }
.ede-hero p { color: #95D5B2; font-size: 14.5px; margin: 0; line-height: 1.6; }
.ede-badge { display: inline-block; background: rgba(255,255,255,0.12); color: #B7E4C7; font-size: 11px; font-weight: 700; padding: 4px 14px; border-radius: 20px; margin-bottom: 16px; letter-spacing: 1px; text-transform: uppercase; border: 1px solid rgba(183,228,199,0.3); }

/* ── INTAKE FORM ── */
.form-card { background: #ffffff; border: 1px solid #C8E6C9; border-radius: 16px; padding: 28px 30px; margin-bottom: 20px; box-shadow: 0 2px 12px rgba(64,145,108,0.07); }
.form-header { background: linear-gradient(135deg, #D85A2A 0%, #E07040 100%); border-radius: 10px; padding: 12px 20px; margin-bottom: 20px; }
.form-header-text { color: #ffffff; font-weight: 700; font-size: 15px; letter-spacing: 0.5px; }
.form-ref-badge { display: inline-block; background: #E8F5E9; color: #2D6A4F; font-size: 13px; font-weight: 700; padding: 6px 16px; border-radius: 8px; border: 1px solid #A5D6A7; margin-bottom: 20px; font-family: monospace; }
.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.form-field-label { font-size: 11px; font-weight: 700; color: #2D6A4F; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.field-hint { font-size: 11px; color: #6B9E7A; margin-top: 4px; font-style: italic; }

/* ── ANONYMISATION PREVIEW ── */
.anon-card { background: #ffffff; border: 1px solid #BFD7ED; border-radius: 16px; padding: 28px 30px; margin-bottom: 20px; box-shadow: 0 2px 12px rgba(24,95,165,0.07); }
.anon-header { background: linear-gradient(135deg, #185FA5 0%, #2478C8 100%); border-radius: 10px; padding: 12px 20px; margin-bottom: 16px; }
.anon-header-text { color: #ffffff; font-weight: 700; font-size: 15px; }
.anon-desc { font-size: 13.5px; color: #1B3A5C; background: #EEF4FB; border-radius: 10px; padding: 12px 16px; margin-bottom: 16px; line-height: 1.6; border-left: 4px solid #185FA5; }
.gdpr-badge { display: inline-flex; align-items: center; gap: 6px; background: #E8F5E9; color: #1B4332; font-size: 11px; font-weight: 700; padding: 5px 12px; border-radius: 20px; border: 1px solid #A5D6A7; margin-bottom: 16px; }

/* ── STEP LABELS ── */
.step-label { font-size: 11px; font-weight: 700; color: #2D6A4F; text-transform: uppercase; letter-spacing: 1.2px; margin: 32px 0 10px 0; display: flex; align-items: center; gap: 10px; }
.step-label::after { content: ''; flex: 1; height: 1px; background: #C8E6C9; }
.step-label-blue { font-size: 11px; font-weight: 700; color: #1B3A5C; text-transform: uppercase; letter-spacing: 1.2px; margin: 32px 0 10px 0; display: flex; align-items: center; gap: 10px; }
.step-label-blue::after { content: ''; flex: 1; height: 1px; background: #BFD7ED; }

/* ── RESULTS ── */
.chunk-card { background: #ffffff; border: 1px solid #C8E6C9; border-left: 5px solid #40916C; border-radius: 12px; padding: 16px 20px; margin-bottom: 14px; font-size: 13.5px; line-height: 1.75; color: #1F2937; box-shadow: 0 1px 6px rgba(64,145,108,0.07); }
.chunk-meta { font-size: 11px; font-weight: 700; color: #40916C; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 10px; }
.sim-bar-wrap { flex: 1; background: #E8F5E9; border-radius: 6px; height: 7px; overflow: hidden; max-width: 160px; }
.sim-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #40916C, #74C69D); }
.advisory-section { background: #ffffff; border: 1px solid #C8E6C9; border-radius: 14px; padding: 28px 30px; margin-bottom: 16px; box-shadow: 0 2px 12px rgba(64,145,108,0.08); }
.advisory-section-title { font-size: 12px; font-weight: 700; color: #2D6A4F; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #E8F5E9; }
.advisory-body { font-size: 14.5px; line-height: 1.85; color: #111827; white-space: pre-wrap; }
.programme-section { background: #ffffff; border: 1px solid #BFD7ED; border-radius: 14px; padding: 28px 30px; margin-bottom: 16px; box-shadow: 0 2px 12px rgba(24,95,165,0.08); }
.programme-section-title { font-size: 12px; font-weight: 700; color: #1B3A5C; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #EEF4FB; }
.programme-body { font-size: 14.5px; line-height: 1.85; color: #111827; white-space: pre-wrap; }
.error-box { background: #FEF2F2; border: 1px solid #FECACA; border-left: 5px solid #EF4444; border-radius: 10px; padding: 16px 20px; font-size: 13.5px; color: #7F1D1D; line-height: 1.7; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] { background-color: #1B4332; }
section[data-testid="stSidebar"] * { color: #D8F3DC !important; }
section[data-testid="stSidebar"] .sidebar-title { font-family: 'DM Serif Display', serif; font-size: 20px; color: #D8F3DC !important; margin-bottom: 4px; }
section[data-testid="stSidebar"] .sidebar-sub { font-size: 12px; color: #95D5B2 !important; margin-bottom: 20px; line-height: 1.5; }
section[data-testid="stSidebar"] hr { border-color: #2D6A4F !important; margin: 16px 0; }
section[data-testid="stSidebar"] .sidebar-step { background: rgba(255,255,255,0.07); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; font-size: 13px; line-height: 1.6; color: #B7E4C7 !important; }
section[data-testid="stSidebar"] .sidebar-step strong { color: #D8F3DC !important; display: block; margin-bottom: 3px; }
section[data-testid="stSidebar"] .sidebar-tip { background: rgba(64,145,108,0.25); border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #95D5B2 !important; line-height: 1.6; margin-top: 16px; }

/* ── BUTTONS ── */
div[data-testid="stButton"] > button[kind="primary"] { background: linear-gradient(135deg, #2D6A4F, #40916C); border: none; border-radius: 10px; color: white; font-weight: 600; font-size: 15px; padding: 0.65rem 1.2rem; box-shadow: 0 3px 12px rgba(45,106,79,0.35); transition: all 0.2s ease; }
div[data-testid="stButton"] > button[kind="primary"]:hover { background: linear-gradient(135deg, #1B4332, #2D6A4F); transform: translateY(-1px); }
div[data-testid="stButton"] > button[kind="secondary"] { border: 1.5px solid #40916C; color: #2D6A4F; border-radius: 10px; font-weight: 600; background: white; transition: all 0.2s ease; }
div[data-testid="stButton"] > button[kind="secondary"]:hover { background: #E8F5E9; }
div[data-testid="stTabs"] button { font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 13.5px; color: #2D6A4F; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #1B4332; border-bottom-color: #40916C; }
.upload-hint { background: #EEF7EE; border: 2px dashed #74C69D; border-radius: 14px; padding: 22px 24px; margin-bottom: 16px; font-size: 13.5px; color: #2D6A4F; line-height: 1.7; }
.programme-hint { background: #EEF4FB; border: 2px dashed #74A9D5; border-radius: 14px; padding: 22px 24px; margin-bottom: 16px; font-size: 13.5px; color: #1B3A5C; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
import random, string

def gen_ref():
    letters = random.choices(string.ascii_uppercase, k=2)
    digits  = random.choices(string.digits, k=2)
    return f"EDE-{''.join(letters)}{''.join(digits)}"

for key, default in [
    ("result_ready", False),
    ("advisory", ""),
    ("client_case_snapshot", ""),
    ("retrieved_chunks", []),
    ("app_mode", "Case Advisory"),
    ("language", "es"),
    ("case_ref", gen_ref()),
    ("anonymised_text", ""),
    ("anon_ready", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qdrant():
    qdrant_url = (
        os.getenv("QDRANT_URL")
        or st.secrets.get("QDRANT_URL", "http://localhost:6333")
    )
    qdrant_key = (
        os.getenv("QDRANT_API_KEY")
        or st.secrets.get("QDRANT_API_KEY", None)
    )
    return QdrantClient(url=qdrant_url, api_key=qdrant_key)

@st.cache_resource
def load_claude():
    api_key = (
        os.getenv("ANTHROPIC_API_KEY")
        or st.secrets.get("ANTHROPIC_API_KEY", None)
    )
    return anthropic.Anthropic(api_key=api_key)

try:
    embedder = load_embedder()
    qdrant   = load_qdrant()
    claude   = load_claude()
except Exception as e:
    st.markdown(f'<div class="error-box">⚠️ <strong>Error:</strong><br><br>{str(e)}</div>', unsafe_allow_html=True)
    st.stop()

COLLECTION = "social_work_reports"
TOP_K      = 3

# ── HELPERS ────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc).strip()

def extract_text_from_image_via_claude(file_bytes, media_type):
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    response = claude.messages.create(
        model="claude-sonnet-4-6", max_tokens=1200,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": "This is a social work case file image. Extract all the text content from it and return it as clean, structured plain text. Preserve field names and their values. Do not add interpretation — only extract what is written."}
        ]}]
    )
    return response.content[0].text.strip()

def md_to_html(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    return text.replace('\n', '<br>')

def extract_section(text, start_key, all_keys):
    start_idx = text.upper().find(start_key)
    if start_idx == -1:
        return None
    content_start = text.find("\n", start_idx) + 1
    end_idx = len(text)
    for k, _ in all_keys:
        if k == start_key:
            continue
        idx = text.upper().find(k, content_start)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    return text[content_start:end_idx].strip()

def highlight_changes(original, anonymised):
    """Return anonymised text with simple word-level highlights where it differs from original."""
    orig_words = original.lower().split()
    anon_words = anonymised.split()
    # Simple heuristic: mark words in anonymised that don't appear in original as changed
    orig_set = set(original.lower().split())
    highlighted = []
    for word in anon_words:
        clean = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
        if clean and clean not in orig_set and len(clean) > 2:
            highlighted.append(f'<mark style="background:#FFF3CD;border-radius:3px;padding:1px 3px;">{word}</mark>')
        else:
            highlighted.append(word)
    return " ".join(highlighted)

def build_case_text_from_form(ref, name, age, nationality, situation, background, risks, intervention, outcome):
    """Assemble a structured text blob from form fields for anonymisation and RAG."""
    return f"""CASE REFERENCE: {ref}
CLIENT NAME: {name}
AGE / NATIONALITY: {age} / {nationality}
CURRENT SITUATION: {situation}
BACKGROUND & HISTORY: {background}
IDENTIFIED RISK FACTORS: {risks}
LAST INTERVENTION: {intervention}
OUTCOME / STATUS: {outcome}"""

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**🌐 Language / Idioma / Hizkuntza**")
    lang_col1, lang_col2, lang_col3 = st.columns(3)
    with lang_col1:
        if st.button("🇪🇸 ES", use_container_width=True,
                     type="primary" if st.session_state.language == "es" else "secondary"):
            st.session_state.language = "es"; st.rerun()
    with lang_col2:
        if st.button("🇬🇧 EN", use_container_width=True,
                     type="primary" if st.session_state.language == "en" else "secondary"):
            st.session_state.language = "en"; st.rerun()
    with lang_col3:
        if st.button("🏴 EU", use_container_width=True,
                     type="primary" if st.session_state.language == "eu" else "secondary"):
            st.session_state.language = "eu"; st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    T = LANG[st.session_state.language]
    st.markdown(f'<div class="sidebar-title">{T["sidebar_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-sub">{T["sidebar_sub"]}</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(T["sidebar_how"], unsafe_allow_html=True)
    for strong, desc in T["sidebar_steps"]:
        st.markdown(f'<div class="sidebar-step">{strong}<br>{desc}</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(T["sidebar_about"], unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-tip">{T["sidebar_tip"]}</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(T["sidebar_footer"])

T = LANG[st.session_state.language]

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ede-hero">
    <div class="ede-badge">{T["hero_badge"]}</div>
    <h1>{T["hero_title"]}</h1>
    <p>{T["hero_desc"]}</p>
</div>
""", unsafe_allow_html=True)

# ── MODE TOGGLE ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="step-label">{T["mode_label"]}</div>', unsafe_allow_html=True)
mode_col1, mode_col2, mode_col3 = st.columns([1, 1, 2])
with mode_col1:
    if st.button(T["mode_case"], type="primary" if st.session_state.app_mode == "Case Advisory" else "secondary", use_container_width=True):
        st.session_state.app_mode = "Case Advisory"
        st.session_state.result_ready = False
        st.session_state.anon_ready = False
        st.rerun()
with mode_col2:
    if st.button(T["mode_prog"], type="primary" if st.session_state.app_mode == "Programme Research" else "secondary", use_container_width=True):
        st.session_state.app_mode = "Programme Research"
        st.session_state.result_ready = False
        st.session_state.anon_ready = False
        st.rerun()

app_mode = st.session_state.app_mode
if app_mode == "Case Advisory":
    st.info(T["mode_case_info"])
else:
    st.info(T["mode_prog_info"])

# ══════════════════════════════════════════════════════════════════════════════
# CASE ADVISORY MODE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "Case Advisory":

    st.markdown(f'<div class="step-label">{T["step1_case"]}</div>', unsafe_allow_html=True)

    tab_form, tab_upload = st.tabs([T["tab_form"], T["tab_upload"]])

    # ── STRUCTURED FORM TAB ────────────────────────────────────────────────────
    with tab_form:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        # Header bar (orange like the mockup)
        st.markdown(f'<div class="form-header"><span class="form-header-text">NEW CASE</span></div>', unsafe_allow_html=True)

        # Auto-generated reference
        st.markdown(f'<div class="form-ref-badge">🔖 {T["form_ref"]}: {st.session_state.case_ref}</div>', unsafe_allow_html=True)

        # Row 1: Name
        st.markdown(f'<div class="form-field-label">{T["form_name"]}</div>', unsafe_allow_html=True)
        f_name = st.text_input("name", placeholder=T["form_name_hint"], label_visibility="collapsed", key="f_name")

        # Row 2: Age + Nationality side by side
        col_age, col_nat = st.columns(2)
        with col_age:
            st.markdown(f'<div class="form-field-label">{T["form_age"]}</div>', unsafe_allow_html=True)
            f_age = st.text_input("age", placeholder="e.g. 27", label_visibility="collapsed", key="f_age")
        with col_nat:
            st.markdown(f'<div class="form-field-label">{T["form_nationality"]}</div>', unsafe_allow_html=True)
            f_nationality = st.text_input("nationality", placeholder="e.g. Malian", label_visibility="collapsed", key="f_nationality")

        # Current Situation
        st.markdown(f'<div class="form-field-label">{T["form_situation"]}</div>', unsafe_allow_html=True)
        st.caption(T["form_situation_hint"])
        f_situation = st.text_area("situation", height=100, label_visibility="collapsed", key="f_situation")

        # Background & History
        st.markdown(f'<div class="form-field-label">{T["form_background"]}</div>', unsafe_allow_html=True)
        st.caption(T["form_background_hint"])
        f_background = st.text_area("background", height=100, label_visibility="collapsed", key="f_background")

        # Risk Factors
        st.markdown(f'<div class="form-field-label">{T["form_risks"]}</div>', unsafe_allow_html=True)
        st.caption(T["form_risks_hint"])
        f_risks = st.text_area("risks", height=90, label_visibility="collapsed", key="f_risks")

        # Last Intervention
        st.markdown(f'<div class="form-field-label">{T["form_intervention"]}</div>', unsafe_allow_html=True)
        st.caption(T["form_intervention_hint"])
        f_intervention = st.text_area("intervention", height=80, label_visibility="collapsed", key="f_intervention")

        # Outcome / Status
        st.markdown(f'<div class="form-field-label">{T["form_outcome"]}</div>', unsafe_allow_html=True)
        st.caption(T["form_outcome_hint"])
        f_outcome = st.text_area("outcome", height=80, label_visibility="collapsed", key="f_outcome")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── ANONYMISE BUTTON ───────────────────────────────────────────────────
        if st.button(T["anonymise_btn"], type="primary", use_container_width=True):
            if not f_name.strip() or not f_age.strip() or not f_situation.strip():
                st.warning(T["warn_empty_form"])
            else:
                raw_text = build_case_text_from_form(
                    st.session_state.case_ref,
                    f_name, f_age, f_nationality,
                    f_situation, f_background,
                    f_risks, f_intervention, f_outcome
                )
                with st.spinner(T["anonymise_spinner"]):
                    try:
                        anon_response = claude.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1400,
                            system=T["anon_system"],
                            messages=[{"role": "user", "content": f"Please anonymise the following case:\n\n{raw_text}"}]
                        )
                        st.session_state.anonymised_text = anon_response.content[0].text.strip()
                        st.session_state._raw_text_for_highlight = raw_text
                        st.session_state.anon_ready = True
                        st.session_state.result_ready = False
                    except Exception as e:
                        st.markdown(f'<div class="error-box">⚠️ {str(e)}</div>', unsafe_allow_html=True)

    # ── UPLOAD TAB ─────────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown('<div class="upload-hint"><strong>Upload a case file</strong> — PDF or image (PNG, JPG, JPEG).<br>Text is extracted automatically and passed through the same AI anonymisation step.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("file", type=["pdf","png","jpg","jpeg"],
                                          label_visibility="collapsed", key="case_upload")
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            fname = uploaded_file.name.lower()
            file_id = f"{uploaded_file.name}_{len(file_bytes)}"
            if st.session_state.get("_last_upload_id") != file_id:
                with st.spinner("📄 Extracting text..."):
                    try:
                        if fname.endswith(".pdf"):
                            extracted = extract_text_from_pdf(file_bytes)
                            if not extracted or len(extracted) < 30:
                                extracted = extract_text_from_image_via_claude(file_bytes, "application/pdf")
                        else:
                            ext = fname.rsplit(".", 1)[-1]
                            media_type = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg"}.get(ext,"image/jpeg")
                            extracted = extract_text_from_image_via_claude(file_bytes, media_type)
                        st.session_state["_upload_extracted"] = extracted
                        st.session_state["_last_upload_id"] = file_id
                    except Exception as e:
                        st.markdown(f'<div class="error-box">⚠️ {str(e)}</div>', unsafe_allow_html=True)

            if st.session_state.get("_upload_extracted"):
                st.success("Text extracted. Ready to anonymise.")
                st.text_area("Extracted text", value=st.session_state["_upload_extracted"], height=200, key="upload_preview")
                if st.button(T["anonymise_btn"] + " (uploaded file)", type="primary", use_container_width=True):
                    with st.spinner(T["anonymise_spinner"]):
                        try:
                            anon_response = claude.messages.create(
                                model="claude-sonnet-4-6",
                                max_tokens=1400,
                                system=T["anon_system"],
                                messages=[{"role": "user", "content": f"Please anonymise the following case:\n\n{st.session_state['_upload_extracted']}"}]
                            )
                            st.session_state.anonymised_text = anon_response.content[0].text.strip()
                            st.session_state._raw_text_for_highlight = st.session_state["_upload_extracted"]
                            st.session_state.anon_ready = True
                            st.session_state.result_ready = False
                        except Exception as e:
                            st.markdown(f'<div class="error-box">⚠️ {str(e)}</div>', unsafe_allow_html=True)

    # ── ANONYMISATION PREVIEW ──────────────────────────────────────────────────
    if st.session_state.anon_ready and st.session_state.anonymised_text:

        st.markdown(f'<div class="step-label-blue">{T["anon_preview_title"]}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="anon-desc">{T["anon_preview_desc"]}</div>', unsafe_allow_html=True)
        st.markdown('<span class="gdpr-badge">🔒 GDPR — No identifying data below this line</span>', unsafe_allow_html=True)

        # Highlighted diff view (read-only display)
        highlighted = highlight_changes(
            st.session_state.get("_raw_text_for_highlight", ""),
            st.session_state.anonymised_text
        )
        st.markdown(f'<div style="background:#fff;border:1px solid #BFD7ED;border-radius:12px;padding:20px 24px;font-size:13.5px;line-height:1.85;color:#111827;white-space:pre-wrap;margin-bottom:16px;">{highlighted}</div>', unsafe_allow_html=True)

        # Editable version for corrections
        edited_anon = st.text_area(
            T["anon_edit_label"],
            value=st.session_state.anonymised_text,
            height=250,
            key="edited_anon_text"
        )

        # Confirm + submit button
        col_run, col_reset = st.columns([3, 1])
        with col_run:
            run = st.button(T["run_case"], type="primary", use_container_width=True)
        with col_reset:
            reset = st.button(T["reset"], type="secondary", use_container_width=True)

        if reset:
            for k in ["result_ready","advisory","client_case_snapshot","retrieved_chunks",
                      "anon_ready","anonymised_text","_raw_text_for_highlight",
                      "_upload_extracted","_last_upload_id","case_ref"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.case_ref = gen_ref()
            st.rerun()

        if run:
            final_case = edited_anon.strip()
            if not final_case:
                st.warning(T["warn_anon_first"])
                st.stop()

            with st.expander("⚙️ Technical log", expanded=False):
                log = st.empty()
                log.info(T["log_encoding"])
                query_vector = embedder.encode(final_case).tolist()
                log.success(T["log_encoded"])
                log.info(T["log_searching"])
                try:
                    search_result = qdrant.query_points(
                        collection_name=COLLECTION, query=query_vector,
                        limit=TOP_K, with_payload=True)
                    results = search_result.points
                    log.success(T["log_found"].format(len(results)))
                except Exception as e:
                    st.markdown(f'<div class="error-box"><strong>Qdrant error:</strong><br>{str(e)}</div>', unsafe_allow_html=True)
                    st.stop()

            if not results:
                st.info(T["no_records"])
                st.stop()

            chunks = []
            retrieved_context = ""
            for i, hit in enumerate(results, 1):
                score_pct = round(hit.score * 100, 1)
                chunk_text = (hit.payload.get("text") or hit.payload.get("content") or
                              hit.payload.get("chunk") or hit.payload.get("page_content") or str(hit.payload))
                source = hit.payload.get("source", f"Report {i}")
                chunks.append({"index": i, "score": score_pct, "source": source, "text": chunk_text})
                retrieved_context += f"\n\n--- Historical Record {i} (similarity: {score_pct}%) from {source} ---\n{chunk_text}"

            user_prompt = f"ANONYMISED CLIENT CASE:\n{final_case}\n\nRETRIEVED HISTORICAL RECORDS FROM DATABASE:\n{retrieved_context}\n\nPlease write a structured response based on the above."

            with st.spinner(T["spinner_case"]):
                try:
                    response = claude.messages.create(
                        model="claude-sonnet-4-6", max_tokens=1400,
                        system=T["system_case"],
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    advisory = response.content[0].text
                except Exception as e:
                    st.markdown(f'<div class="error-box"><strong>Claude API error:</strong><br>{str(e)}</div>', unsafe_allow_html=True)
                    st.stop()

            st.session_state.advisory = advisory
            st.session_state.client_case_snapshot = final_case
            st.session_state.retrieved_chunks = chunks
            st.session_state.result_ready = True
            st.rerun()

    # ── If anon not yet done, show reset only ─────────────────────────────────
    elif not st.session_state.anon_ready:
        if st.button(T["reset"], type="secondary"):
            for k in ["result_ready","advisory","anon_ready","anonymised_text","_upload_extracted","_last_upload_id"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.case_ref = gen_ref()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PROGRAMME RESEARCH MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f'<div class="step-label-blue">{T["step1_prog"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="programme-hint">{T["prog_hint"]}</div>', unsafe_allow_html=True)
    client_case = st.text_area("q", height=130, placeholder=T["prog_placeholder"],
                                label_visibility="collapsed", key="programme_query")

    col1, col2 = st.columns([3, 1])
    with col1:
        run = st.button(T["run_prog"], type="primary", use_container_width=True)
    with col2:
        reset = st.button(T["reset"], type="secondary", use_container_width=True)

    if reset:
        for k in ["result_ready","advisory","client_case_snapshot","retrieved_chunks"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    if run:
        if not client_case or not client_case.strip():
            st.warning(T["warn_empty_prog"])
            st.stop()

        with st.expander("⚙️ Technical log", expanded=False):
            log = st.empty()
            log.info(T["log_encoding"])
            query_vector = embedder.encode(client_case).tolist()
            log.success(T["log_encoded"])
            log.info(T["log_searching"])
            try:
                search_result = qdrant.query_points(
                    collection_name=COLLECTION, query=query_vector,
                    limit=TOP_K, with_payload=True)
                results = search_result.points
                log.success(T["log_found"].format(len(results)))
            except Exception as e:
                st.markdown(f'<div class="error-box"><strong>Qdrant error:</strong><br>{str(e)}</div>', unsafe_allow_html=True)
                st.stop()

        if not results:
            st.info(T["no_records"])
            st.stop()

        chunks = []
        retrieved_context = ""
        for i, hit in enumerate(results, 1):
            score_pct = round(hit.score * 100, 1)
            chunk_text = (hit.payload.get("text") or hit.payload.get("content") or
                          hit.payload.get("chunk") or hit.payload.get("page_content") or str(hit.payload))
            source = hit.payload.get("source", f"Report {i}")
            chunks.append({"index": i, "score": score_pct, "source": source, "text": chunk_text})
            retrieved_context += f"\n\n--- Historical Record {i} (similarity: {score_pct}%) from {source} ---\n{chunk_text}"

        user_prompt = f"RESEARCH INPUT:\n{client_case}\n\nRETRIEVED HISTORICAL RECORDS FROM DATABASE:\n{retrieved_context}\n\nPlease write a structured response based on the above."

        with st.spinner(T["spinner_prog"]):
            try:
                response = claude.messages.create(
                    model="claude-sonnet-4-6", max_tokens=1400,
                    system=T["system_prog"],
                    messages=[{"role": "user", "content": user_prompt}]
                )
                advisory = response.content[0].text
            except Exception as e:
                st.markdown(f'<div class="error-box"><strong>Claude API error:</strong><br>{str(e)}</div>', unsafe_allow_html=True)
                st.stop()

        st.session_state.advisory = advisory
        st.session_state.client_case_snapshot = client_case
        st.session_state.retrieved_chunks = chunks
        st.session_state.result_ready = True

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS (both modes)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.result_ready:

    if app_mode == "Case Advisory":
        st.markdown(f'<div class="step-label">{T["step2_results"]}</div>', unsafe_allow_html=True)
        for chunk in st.session_state.retrieved_chunks:
            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-meta">
                    Record {chunk["index"]} &nbsp;·&nbsp; {chunk["source"]}
                    <div class="sim-bar-wrap"><div class="sim-bar-fill" style="width:{chunk['score']}%"></div></div>
                    {chunk["score"]}% match
                </div>
                {chunk["text"][:600]}{"…" if len(chunk["text"]) > 600 else ""}
            </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="step-label">{T["step3_case"]}</div>', unsafe_allow_html=True)
        sections_found = False
        for raw_key, display_label in T["case_sections"]:
            content = extract_section(st.session_state.advisory, raw_key, T["case_sections"])
            if content:
                sections_found = True
                st.markdown(f"""
                <div class="advisory-section">
                    <div class="advisory-section-title">{display_label}</div>
                    <div class="advisory-body">{md_to_html(content)}</div>
                </div>""", unsafe_allow_html=True)
        if not sections_found:
            st.markdown(f"""
            <div class="advisory-section">
                <div class="advisory-section-title">{T["case_fallback_title"]}</div>
                <div class="advisory-body">{md_to_html(st.session_state.advisory)}</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f'<div class="step-label-blue">{T["step3_prog"]}</div>', unsafe_allow_html=True)
        sections_found = False
        for raw_key, display_label in T["prog_sections"]:
            content = extract_section(st.session_state.advisory, raw_key, T["prog_sections"])
            if content:
                sections_found = True
                st.markdown(f"""
                <div class="programme-section">
                    <div class="programme-section-title">{display_label}</div>
                    <div class="programme-body">{md_to_html(content)}</div>
                </div>""", unsafe_allow_html=True)
        if not sections_found:
            st.markdown(f"""
            <div class="programme-section">
                <div class="programme-section-title">{T["prog_fallback_title"]}</div>
                <div class="programme-body">{md_to_html(st.session_state.advisory)}</div>
            </div>""", unsafe_allow_html=True)

    # ── DOWNLOAD ───────────────────────────────────────────────────────────────
    dl_header = T["download_header_case"] if app_mode == "Case Advisory" else T["download_header_prog"]
    dl_label  = T["download_input_label"]
    dl_brief  = T["download_brief_label"]
    dl_btn    = T["download_case"] if app_mode == "Case Advisory" else T["download_prog"]
    dl_file   = T["file_case"] if app_mode == "Case Advisory" else T["file_prog"]

    download_content = f"""{dl_header}
{'='*60}

{dl_label}:
{st.session_state.client_case_snapshot}

{'='*60}

{dl_brief}:
{st.session_state.advisory}
"""
    st.download_button(
        label=dl_btn,
        data=download_content.encode("utf-8"),
        file_name=dl_file,
        mime="text/plain",
        use_container_width=True,
    )
