import streamlit as st
import anthropic
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re, base64

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
        "mode_case_info": "**Modo Asesoría de Casos** — Envía un perfil individual de cliente. Claude recuperará casos históricos similares y generará un informe de intervención estructurado.",
        "mode_prog_info": "**Modo Investigación de Programa** — Haz una pregunta amplia sobre los 13 registros históricos. Claude analizará patrones y presentará conclusiones a nivel de programa.",
        "step1_case": "✍️ Paso 1 — Enviar caso del cliente",
        "step1_prog": "🔵 Paso 1 — Escribe tu pregunta de investigación",
        "tab_type": "✏️  Escribir / pegar caso",
        "tab_upload": "📎  Subir archivo de caso",
        "case_placeholder": "Ejemplo: Amara, 28 años, refugiada recién llegada, español limitado, dos hijos pequeños, dificultades para acceder a vivienda y servicios de empleo...",
        "prog_placeholder": "Ejemplo: ¿Qué pasos de integración lingüística y marcos de apoyo psicológico hemos utilizado con éxito para mujeres solicitantes de asilo aisladas?",
        "upload_hint": "<strong>Sube un archivo de caso</strong> — PDF o imagen (PNG, JPG, JPEG).<br>El sistema extraerá el texto automáticamente y lo usará como entrada del caso.<br><em>Compatible: fichas de caso estructuradas, formularios escaneados, documentos fotografiados.</em>",
        "prog_hint": "<strong>Modo Investigación de Programa</strong> — Haz una pregunta amplia sobre patrones, marcos o resultados en los registros históricos de EDE.<br><br><em>Ejemplo: '¿Qué pasos de integración lingüística han funcionado para mujeres solicitantes de asilo aisladas?' o '¿Qué marcos de apoyo psicológico aparecen más en nuestros registros?'</em>",
        "extracted_label": "✅ Vista previa del texto extraído",
        "edit_btn": "✏️ Editar antes de enviar",
        "run_case": "🔍 Buscar Casos Similares y Generar Asesoría",
        "run_prog": "🔍 Buscar Registros y Generar Informe de Investigación",
        "reset": "🔄 Nuevo Caso",
        "warn_empty": "Por favor, introduce una descripción del caso o pregunta de investigación antes de continuar.",
        "log_encoding": "Codificando el perfil del caso en un vector de búsqueda...",
        "log_encoded": "✅ Perfil del caso codificado correctamente.",
        "log_searching": "Buscando registros similares en la colección Qdrant...",
        "log_found": "✅ Recuperados {} registros de Qdrant.",
        "spinner_case": "Claude está leyendo los registros y redactando tu informe de asesoría...",
        "spinner_prog": "Claude está analizando los registros y redactando tu informe de investigación...",
        "step2": "📂 Paso 2 — Registros históricos más similares",
        "step3_case": "📋 Paso 3 — Informe de Asesoría de Claude",
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
            ("<strong>② Describe o sube</strong>", "Escribe un perfil de caso o pregunta, o sube un PDF / imagen."),
            ("<strong>③ Ejecuta la búsqueda</strong>", "El sistema recupera los registros más similares de los 13 informes en Qdrant."),
            ("<strong>④ Lee la asesoría</strong>", "Claude genera un informe estructurado adaptado a tu entrada y modo."),
            ("<strong>⑤ Descarga o reinicia</strong>", "Guarda el informe como .txt o haz clic en Nuevo Caso."),
        ],
        "sidebar_about": "**ℹ️ Acerca de**",
        "sidebar_tip": "Esta herramienta usa <strong>Generación Aumentada por Recuperación (RAG)</strong>: tu consulta se compara con una base de datos vectorial Qdrant local, y el contexto recuperado se envía a Claude.<br><br>Desarrollado con: <strong>Streamlit · Qdrant · sentence-transformers · Claude API</strong>",
        "sidebar_footer": "EDE Fundazioa · BGT 4ª Edición · 2025",
        "no_records": "No se encontraron registros coincidentes. Tu colección Qdrant puede estar vacía.",
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
        "download_input_label": "ENTRADA",
        "download_brief_label": "INFORME",
        "system_case": """Eres un asesor experimentado de trabajo social que apoya a los trabajadores de casos en EDE Fundazioa, en el País Vasco, España.

Recibirás:
1. Un nuevo perfil de cliente escrito por un trabajador social
2. Extractos recuperados de informes de casos históricos similares al nuevo caso

Si la entrada NO es un perfil de cliente individual específico — por ejemplo, si es una pregunta de investigación general, una consulta amplia de programa, o una solicitud de resúmenes de varios casos — debes negarte a generar una asesoría de caso. En su lugar, explica claramente que este modo está diseñado solo para casos de clientes individuales y sugiere al usuario que cambie al Modo de Investigación de Programa.

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
1. Una pregunta de investigación a nivel de programa de un miembro del personal o responsable
2. Extractos recuperados de informes de casos históricos relevantes para la pregunta

Tu función es analizar patrones, marcos y resultados en estos registros y proporcionar una respuesta a nivel de programa basada en evidencias.

Estructura tu respuesta usando exactamente estas tres secciones etiquetadas:

PATRONES CLAVE IDENTIFICADOS
[Resume los principales patrones, tendencias o temas relevantes para la pregunta de investigación, extraídos de los registros recuperados]

MARCOS Y ENFOQUES EN USO
[Identifica marcos, métodos o enfoques específicos que aparecen en los registros — nómbralos claramente y señala cómo se aplicaron]

RECOMENDACIONES PARA EL DESARROLLO DEL PROGRAMA
[Ofrece 2-3 recomendaciones accionables para el diseño de programas o la prestación de servicios de EDE, basadas en la evidencia de los registros]

Sé analítico, basado en evidencias y escribe en español claro, adecuado para un responsable de programa o supervisor.
Señala claramente si los registros recuperados no contienen suficiente evidencia para responder plenamente a la pregunta — no inventes ni asumas detalles.""",
    },

    "en": {
        "hero_badge": "🌿 EDE Fundazioa · BGT Challenge",
        "hero_title": "Social Support Advisor",
        "hero_desc": "Select a mode below — then describe a client or ask a programme-level research question.<br>The system retrieves similar historical cases and generates a tailored brief via Claude.",
        "mode_label": "⚙️ Select Mode",
        "mode_case": "🟢 Case Advisory",
        "mode_prog": "🔵 Programme Research",
        "mode_case_info": "**Case Advisory Mode** — Submit an individual client profile. Claude will retrieve similar historical cases and generate a structured intervention brief.",
        "mode_prog_info": "**Programme Research Mode** — Ask a broad question across all 13 historical records. Claude will analyse patterns and surface insights at programme level.",
        "step1_case": "✍️ Step 1 — Submit client case",
        "step1_prog": "🔵 Step 1 — Enter your research question",
        "tab_type": "✏️  Type / paste case",
        "tab_upload": "📎  Upload case file",
        "case_placeholder": "Example: Amara, age 28, recently arrived refugee, limited language skills, two young children, struggling to access housing support and employment services...",
        "prog_placeholder": "Example: Based on our historical records, what specific language integration steps and psychological support frameworks have we successfully utilized for isolated female asylum seekers?",
        "upload_hint": "<strong>Upload a case file</strong> — PDF or image (PNG, JPG, JPEG).<br>The system will extract the text automatically and use it as the case input.<br><em>Supported: structured case sheets, scanned forms, photographed documents.</em>",
        "prog_hint": "<strong>Programme Research Mode</strong> — Ask a broad question about patterns, frameworks, or outcomes across EDE's historical case records.<br><br><em>Example: 'What language integration steps have worked for isolated female asylum seekers?' or 'What psychological support frameworks appear most in our records?'</em>",
        "extracted_label": "✅ Extracted text preview",
        "edit_btn": "✏️ Edit before submitting",
        "run_case": "🔍 Find Similar Cases & Generate Advisory",
        "run_prog": "🔍 Search Records & Generate Research Brief",
        "reset": "🔄 New Case",
        "warn_empty": "Please enter a case description or research question before running.",
        "log_encoding": "Encoding your case profile into a search vector...",
        "log_encoded": "✅ Case profile encoded successfully.",
        "log_searching": "Searching Qdrant collection for similar records...",
        "log_found": "✅ Retrieved {} records from Qdrant.",
        "spinner_case": "Claude is reading the records and writing your advisory brief...",
        "spinner_prog": "Claude is analysing the records and writing your research brief...",
        "step2": "📂 Step 2 — Most similar historical records",
        "step3_case": "📋 Step 3 — Claude Advisory Brief",
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
            ("<strong>② Describe or upload</strong>", "Type a case profile or research question, or upload a PDF / image."),
            ("<strong>③ Run the search</strong>", "The system retrieves the most similar records from 13 historical reports in Qdrant."),
            ("<strong>④ Read the advisory</strong>", "Claude generates a structured brief tailored to your input and mode."),
            ("<strong>⑤ Download or reset</strong>", "Save the brief as a .txt file, or click New Case to start fresh."),
        ],
        "sidebar_about": "**ℹ️ About**",
        "sidebar_tip": "This tool uses <strong>Retrieval-Augmented Generation (RAG)</strong>: your query is matched against a local Qdrant vector database, and retrieved context is sent to Claude to ground the advisory in real case history.<br><br>Built with: <strong>Streamlit · Qdrant · sentence-transformers · Claude API</strong>",
        "sidebar_footer": "EDE Fundazioa · BGT 4th Edition · 2025",
        "no_records": "No matching records found. Your Qdrant collection may be empty.",
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
        "download_input_label": "INPUT",
        "download_brief_label": "BRIEF",
        "system_case": """You are an experienced social work advisor supporting caseworkers at EDE Fundazioa in the Basque Country, Spain.

You will receive:
1. A new client profile written by a social worker
2. Retrieved excerpts from historical case reports similar to the new case

If the input is NOT a specific individual client profile — for example if it is a general research question, a broad programme query, or a request for summaries across multiple cases — you must refuse to generate a case advisory. Instead, clearly explain that this mode is designed for individual client cases only, and suggest the user switch to Programme Research Mode.

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

Your role is to analyse patterns, frameworks, and outcomes across these records and provide an evidence-based programme-level response.

Structure your response using exactly these three labelled sections:

KEY PATTERNS IDENTIFIED
[Summarise the main patterns, trends, or themes relevant to the research question, drawn from the retrieved records]

FRAMEWORKS & APPROACHES IN USE
[Identify specific frameworks, methods, or approaches that appear in the records — name them clearly and note how they were applied]

RECOMMENDATIONS FOR PROGRAMME DEVELOPMENT
[Offer 2–3 actionable recommendations for EDE's programme design or service delivery, grounded in the evidence from the records]

Be analytical, evidence-based, and write in plain English suitable for a programme manager or supervisor.
Clearly note if the retrieved records do not contain enough evidence to fully answer the question — do not invent or assume details.""",
    },

    "eu": {
        "hero_badge": "🌿 EDE Fundazioa · BGT Erronka",
        "hero_title": "Gizarte Laguntzaren Aholkularia",
        "hero_desc": "Hautatu modu bat — deskribatu bezero bat edo egin programa-mailako ikerketa galdera bat.<br>Sistemak antzeko kasu historikoak berreskuratzen ditu eta Claude bidez txosten pertsonalizatua sortzen du.",
        "mode_label": "⚙️ Hautatu Modua",
        "mode_case": "🟢 Kasuen Aholkularitza",
        "mode_prog": "🔵 Programaren Ikerketa",
        "mode_case_info": "**Kasuen Aholkularitza Modua** — Banakako bezeroaren profila bidali. Claudek antzeko kasu historikoak berreskuratuko ditu eta esku-hartze txostena sortuko du.",
        "mode_prog_info": "**Programaren Ikerketa Modua** — Egin galdera zabal bat 13 erregistro historikoen gainean. Claudek ereduak aztertuko ditu eta programa-mailako ondorioak aurkeztuko ditu.",
        "step1_case": "✍️ 1. urratsa — Bezeroaren kasua bidali",
        "step1_prog": "🔵 1. urratsa — Idatzi zure ikerketa galdera",
        "tab_type": "✏️  Idatzi / itsatsi kasua",
        "tab_upload": "📎  Kargatu kasuen fitxategia",
        "case_placeholder": "Adibidea: Amara, 28 urte, errefuxiatu berria, gaztelania mugatua, bi seme-alaba txiki, etxebizitza eta enplegu zerbitzuetara sarbidea izateko zailtasunak...",
        "prog_placeholder": "Adibidea: Gure erregistro historikoen arabera, zein hizkuntza-integrazio urrats eta laguntza psikologikorako esparru erabili ditugu arrakastaz isolatutako emakume asilo-eskatzaileekin?",
        "upload_hint": "<strong>Kargatu kasuen fitxategia</strong> — PDF edo irudia (PNG, JPG, JPEG).<br>Sistemak testua automatikoki aterako du eta kasuaren sarrera gisa erabiliko du.<br><em>Onartutakoak: egituratutako kasu-orriak, eskanatutako inprimakiak, argazkitatutako dokumentuak.</em>",
        "prog_hint": "<strong>Programaren Ikerketa Modua</strong> — Egin galdera zabal bat EDEren kasu-erregistro historikoen ereduei, esparruei edo emaitzei buruz.<br><br><em>Adibidea: 'Zein hizkuntza-integrazio urrats funtzionatu dute isolatutako emakume asilo-eskatzaileekin?' edo 'Zein laguntza psikologikorako esparru agertzen dira gehien gure erregistroetan?'</em>",
        "extracted_label": "✅ Ateratako testuaren aurrebista",
        "edit_btn": "✏️ Editatu bidali aurretik",
        "run_case": "🔍 Antzeko Kasuak Bilatu eta Aholkularitza Sortu",
        "run_prog": "🔍 Erregistroak Bilatu eta Ikerketa Txostena Sortu",
        "reset": "🔄 Kasu Berria",
        "warn_empty": "Mesedez, sartu kasuaren deskribapena edo ikerketa galdera aurrera jarraitu aurretik.",
        "log_encoding": "Kasuaren profila bilaketa-bektore batean kodetzen...",
        "log_encoded": "✅ Kasuaren profila ongi kodetu da.",
        "log_searching": "Qdrant bilduman antzeko erregistroak bilatzen...",
        "log_found": "✅ {} erregistro berreskuratu dira Qdrantetik.",
        "spinner_case": "Claude erregistroak irakurtzen eta zure aholkularitza txostena idazten ari da...",
        "spinner_prog": "Claude erregistroak aztertzen eta zure ikerketa txostena idazten ari da...",
        "step2": "📂 2. urratsa — Antzeko erregistro historikoak",
        "step3_case": "📋 3. urratsa — Clauderen Aholkularitza Txostena",
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
            ("<strong>② Deskribatu edo kargatu</strong>", "Idatzi kasuaren profila edo ikerketa galdera, edo kargatu PDF / irudia."),
            ("<strong>③ Bilaketa egin</strong>", "Sistemak Qdranteko 13 txostenetako erregistro antzekoenak berreskuratzen ditu."),
            ("<strong>④ Irakurri aholkularitza</strong>", "Claudek zure sarrerei eta moduari egokitutako txosten egituratu bat sortzen du."),
            ("<strong>⑤ Deskargatu edo berrezarri</strong>", "Gorde txostena .txt gisa edo egin klik Kasu Berria botoian."),
        ],
        "sidebar_about": "**ℹ️ Honi buruz**",
        "sidebar_tip": "Tresna honek <strong>Berreskuratze-Areagotutako Sorkuntza (RAG)</strong> erabiltzen du: zure kontsulta tokiko Qdrant datu-base bektorial baten aurka konparatzen da, eta berreskuratutako testuingurua Clauderi bidaltzen zaio.<br><br>Eraikita: <strong>Streamlit · Qdrant · sentence-transformers · Claude API</strong>",
        "sidebar_footer": "EDE Fundazioa · BGT 4. Edizioa · 2025",
        "no_records": "Ez da erregistro bat ere aurkitu. Zure Qdrant bilduma hutsik egon daiteke.",
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
        "download_input_label": "SARRERA",
        "download_brief_label": "TXOSTENA",
        "system_case": """Zu Euskal Herrian, Espainian, EDE Fundazioan lan egiten duten gizarte-langileei laguntzen dien gizarte-lanaren aholkulari esperientziadun bat zara.

Jasoko duzu:
1. Gizarte-langile batek idatzitako bezero berri baten profila
2. Kasu berriari antzeko erregistro historikoetatik ateratako zatiak

Sarrera EZ bada banakako bezeroaren profil espezifiko bat — adibidez, ikerketa galdera orokor bat, programa-kontsulta zabal bat, edo hainbat kasuren laburpenen eskaera bada — ezin duzu kasuen aholkularitza sortu. Horren ordez, argi azaldu modu hau banakako bezero-kasuetarako soilik diseinatu dela eta erabiltzaileari Programaren Ikerketa Modura aldatzea gomendatu.

Bezero-profil baliozkoa bidaltzen denean, estruktura ezazu zure erantzuna zehazki lau atal hauetan:

ARRISKU-FAKTOREAK ETA OZTOPOAK
[Identifikatu bezeroaren arrisku-faktore eta oztopo nagusiak bere profiletik]

ERREGISTRO HISTORIKOETATIK IKASGAIAK
[Atera ikasgai zehatzak berreskuratutako erregistro historikoetatik — erreferentziatu zenbakiaren arabera, adib. 1. Erregistroa, 2. Erregistroa]

GOMENDATUTAKO ESKU-HARTZEAK
[Gomendatu 2-3 esku-hartze aukera lehentasunezkoak arrazoi argiarekin bakoitzarentzat]

INFORMAZIO HUTSUNEAK
[Adierazi gizarte-langileak aurrera joan aurretik bildu beharreko xehetasun falta]

Izan zaitez zehatza, profesionala eta frogetan oinarritua. Idatzi euskara arruntean, lehen lerroko gizarte-langile batentzat egokia.
EZ asmatu emandako erregistroetan edo bezero-profilean ez dauden kasu-xehetasunik.""",
        "system_prog": """Zu EDE Fundazioa, Euskal Herriko gizarte-zerbitzu erakunde bati laguntzen dion programa-analistalari nagusi bat zara.

Jasoko duzu:
1. Langile edo arduradun batek egindako programa-mailako ikerketa galdera bat
2. Galderarekin zerikusia duten erregistro historikoetatik ateratako zatiak

Zure eginkizuna da erregistro hauetan ereduak, esparruak eta emaitzak aztertzea eta frogetan oinarritutako programa-mailako erantzuna ematea.

Estruktura ezazu zure erantzuna zehazki hiru atal hauetan:

IDENTIFIKATUTAKO FUNTSEZKO EREDUAK
[Laburtu ikerketa galderarekin zerikusia duten eredu, joera edo gai nagusiak, berreskuratutako erregistroetatik aterata]

ERABILERAN DAUDEN ESPARRUAK ETA IKUSPEGIAK
[Identifikatu erregistroetan agertzen diren esparru, metodo edo ikuspegi zehatzak — izendatu argi eta nola aplikatu ziren adierazi]

PROGRAMAREN GARAPENERAKO GOMENDIOAK
[Eskaini 2-3 gomendio ekintza-bideragarri EDEren programa-diseinurako edo zerbitzu-prestazioarentzat, erregistroen frogetan oinarrituta]

Izan analistikoa, frogetan oinarritua eta idatzi euskara arruntean, programa-arduradun edo gainbegiralearentzat egokia.
Argi adierazi berreskuratutako erregistroek galderari erantzuteko adina frogarik ez badute — ez asmatu edo suposatu xehetasunik.""",
    },
}

# ── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #F4F7F4; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 860px; }
.ede-hero { background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%); border-radius: 18px; padding: 40px 36px 32px; margin-bottom: 32px; text-align: center; box-shadow: 0 4px 24px rgba(27,67,50,0.18); }
.ede-hero h1 { font-family: 'DM Serif Display', serif; color: #D8F3DC; font-size: 34px; margin: 0 0 10px 0; letter-spacing: -0.5px; }
.ede-hero p { color: #95D5B2; font-size: 14.5px; margin: 0; line-height: 1.6; }
.ede-badge { display: inline-block; background: rgba(255,255,255,0.12); color: #B7E4C7; font-size: 11px; font-weight: 700; padding: 4px 14px; border-radius: 20px; margin-bottom: 16px; letter-spacing: 1px; text-transform: uppercase; border: 1px solid rgba(183,228,199,0.3); }
div[data-testid="stTabs"] button { font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 13.5px; color: #2D6A4F; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #1B4332; border-bottom-color: #40916C; }
.upload-hint { background: #EEF7EE; border: 2px dashed #74C69D; border-radius: 14px; padding: 22px 24px; margin-bottom: 16px; font-size: 13.5px; color: #2D6A4F; line-height: 1.7; }
.upload-hint strong { color: #1B4332; }
.extracted-preview { background: #ffffff; border: 1px solid #C8E6C9; border-left: 5px solid #52B788; border-radius: 12px; padding: 16px 20px; margin-top: 14px; font-size: 13px; line-height: 1.75; color: #1F2937; max-height: 220px; overflow-y: auto; white-space: pre-wrap; }
.extracted-label { font-size: 11px; font-weight: 700; color: #40916C; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.programme-hint { background: #EEF4FB; border: 2px dashed #74A9D5; border-radius: 14px; padding: 22px 24px; margin-bottom: 16px; font-size: 13.5px; color: #1B3A5C; line-height: 1.7; }
.programme-hint strong { color: #0F2840; }
.step-label { font-size: 11px; font-weight: 700; color: #2D6A4F; text-transform: uppercase; letter-spacing: 1.2px; margin: 32px 0 10px 0; display: flex; align-items: center; gap: 10px; }
.step-label::after { content: ''; flex: 1; height: 1px; background: #C8E6C9; }
.step-label-blue { font-size: 11px; font-weight: 700; color: #1B3A5C; text-transform: uppercase; letter-spacing: 1.2px; margin: 32px 0 10px 0; display: flex; align-items: center; gap: 10px; }
.step-label-blue::after { content: ''; flex: 1; height: 1px; background: #BFD7ED; }
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
section[data-testid="stSidebar"] { background-color: #1B4332; }
section[data-testid="stSidebar"] * { color: #D8F3DC !important; }
section[data-testid="stSidebar"] .sidebar-title { font-family: 'DM Serif Display', serif; font-size: 20px; color: #D8F3DC !important; margin-bottom: 4px; }
section[data-testid="stSidebar"] .sidebar-sub { font-size: 12px; color: #95D5B2 !important; margin-bottom: 20px; line-height: 1.5; }
section[data-testid="stSidebar"] hr { border-color: #2D6A4F !important; margin: 16px 0; }
section[data-testid="stSidebar"] .sidebar-step { background: rgba(255,255,255,0.07); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; font-size: 13px; line-height: 1.6; color: #B7E4C7 !important; }
section[data-testid="stSidebar"] .sidebar-step strong { color: #D8F3DC !important; display: block; margin-bottom: 3px; }
section[data-testid="stSidebar"] .sidebar-tip { background: rgba(64,145,108,0.25); border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #95D5B2 !important; line-height: 1.6; margin-top: 16px; }
div[data-testid="stButton"] > button[kind="primary"] { background: linear-gradient(135deg, #2D6A4F, #40916C); border: none; border-radius: 10px; color: white; font-weight: 600; font-size: 15px; padding: 0.65rem 1.2rem; box-shadow: 0 3px 12px rgba(45,106,79,0.35); transition: all 0.2s ease; }
div[data-testid="stButton"] > button[kind="primary"]:hover { background: linear-gradient(135deg, #1B4332, #2D6A4F); box-shadow: 0 5px 18px rgba(27,67,50,0.45); transform: translateY(-1px); }
div[data-testid="stButton"] > button[kind="secondary"] { border: 1.5px solid #40916C; color: #2D6A4F; border-radius: 10px; font-weight: 600; background: white; transition: all 0.2s ease; }
div[data-testid="stButton"] > button[kind="secondary"]:hover { background: #E8F5E9; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for key, default in [
    ("result_ready", False),
    ("advisory", ""),
    ("client_case_snapshot", ""),
    ("retrieved_chunks", []),
    ("extracted_text", ""),
    ("upload_processed", False),
    ("app_mode", "Case Advisory"),
    ("language", "es"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qdrant():
    return QdrantClient(host="localhost", port=6333)

@st.cache_resource
def load_claude():
    return anthropic.Anthropic()

try:
    embedder = load_embedder()
    qdrant   = load_qdrant()
    claude   = load_claude()
except Exception as e:
    st.markdown(f'<div class="error-box">⚠️ <strong>Error:</strong><br><br>{str(e)}</div>', unsafe_allow_html=True)
    st.stop()

COLLECTION = "social_work_reports"
TOP_K      = 3

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc).strip()

def extract_text_from_image_via_claude(file_bytes, media_type):
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
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

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Language selector
    st.markdown("**🌐 Language / Idioma / Hizkuntza**")
    lang_col1, lang_col2, lang_col3 = st.columns(3)
    with lang_col1:
        if st.button("🇪🇸 ES", use_container_width=True,
                     type="primary" if st.session_state.language == "es" else "secondary"):
            st.session_state.language = "es"
            st.rerun()
    with lang_col2:
        if st.button("🇬🇧 EN", use_container_width=True,
                     type="primary" if st.session_state.language == "en" else "secondary"):
            st.session_state.language = "en"
            st.rerun()
    with lang_col3:
        if st.button("🏴 EU", use_container_width=True,
                     type="primary" if st.session_state.language == "eu" else "secondary"):
            st.session_state.language = "eu"
            st.rerun()

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

# ── SHORTCUT TO TRANSLATIONS ───────────────────────────────────────────────────
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
        st.rerun()
with mode_col2:
    if st.button(T["mode_prog"], type="primary" if st.session_state.app_mode == "Programme Research" else "secondary", use_container_width=True):
        st.session_state.app_mode = "Programme Research"
        st.session_state.result_ready = False
        st.rerun()

app_mode = st.session_state.app_mode

if app_mode == "Case Advisory":
    st.info(T["mode_case_info"])
else:
    st.info(T["mode_prog_info"])

# ── INPUT ──────────────────────────────────────────────────────────────────────
client_case = ""

if app_mode == "Case Advisory":
    st.markdown(f'<div class="step-label">{T["step1_case"]}</div>', unsafe_allow_html=True)
    tab_type, tab_upload = st.tabs([T["tab_type"], T["tab_upload"]])

    with tab_type:
        client_case = st.text_area(
            label="case", height=160,
            placeholder=T["case_placeholder"],
            label_visibility="collapsed", key="typed_case"
        )

    with tab_upload:
        st.markdown(f'<div class="upload-hint">{T["upload_hint"]}</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("file", type=["pdf","png","jpg","jpeg"],
                                          label_visibility="collapsed", key="case_upload")
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            fname = uploaded_file.name.lower()
            file_id = f"{uploaded_file.name}_{len(file_bytes)}"
            if st.session_state.get("_last_upload_id") != file_id:
                with st.spinner("📄 ..."):
                    try:
                        if fname.endswith(".pdf"):
                            extracted = extract_text_from_pdf(file_bytes)
                            if not extracted or len(extracted) < 30:
                                extracted = extract_text_from_image_via_claude(file_bytes, "application/pdf")
                        else:
                            ext = fname.rsplit(".", 1)[-1]
                            media_type = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg"}.get(ext,"image/jpeg")
                            extracted = extract_text_from_image_via_claude(file_bytes, media_type)
                        st.session_state.extracted_text = extracted
                        st.session_state.upload_processed = True
                        st.session_state["_last_upload_id"] = file_id
                    except Exception as e:
                        st.markdown(f'<div class="error-box">⚠️ {str(e)}</div>', unsafe_allow_html=True)
                        st.session_state.extracted_text = ""

        if st.session_state.upload_processed and st.session_state.extracted_text:
            st.markdown(f'<div class="extracted-label">{T["extracted_label"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="extracted-preview">{st.session_state.extracted_text[:1200]}{"…" if len(st.session_state.extracted_text) > 1200 else ""}</div>', unsafe_allow_html=True)
            client_case = st.session_state.extracted_text
            if st.button(T["edit_btn"], type="secondary", use_container_width=True):
                st.session_state.extracted_editable = st.session_state.extracted_text
            if "extracted_editable" in st.session_state:
                client_case = st.text_area("edit", value=st.session_state.extracted_editable,
                                            height=200, key="edited_extracted")
else:
    st.markdown(f'<div class="step-label-blue">{T["step1_prog"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="programme-hint">{T["prog_hint"]}</div>', unsafe_allow_html=True)
    client_case = st.text_area("q", height=130, placeholder=T["prog_placeholder"],
                                label_visibility="collapsed", key="programme_query")

# ── ACTION BUTTONS ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    run = st.button(T["run_case"] if app_mode == "Case Advisory" else T["run_prog"],
                    type="primary", use_container_width=True)
with col2:
    reset = st.button(T["reset"], type="secondary", use_container_width=True)

if reset:
    for k in ["result_ready","advisory","client_case_snapshot","retrieved_chunks",
              "extracted_text","upload_processed","_last_upload_id","extracted_editable"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ── PIPELINE ───────────────────────────────────────────────────────────────────
if run:
    if not client_case or not client_case.strip():
        st.warning(T["warn_empty"])
        st.stop()

    st.session_state.result_ready = False

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

    system_prompt = T["system_case"] if app_mode == "Case Advisory" else T["system_prog"]
    user_prompt = f"RESEARCH INPUT:\n{client_case}\n\nRETRIEVED HISTORICAL RECORDS FROM DATABASE:\n{retrieved_context}\n\nPlease write a structured response based on the above."

    with st.spinner(T["spinner_case"] if app_mode == "Case Advisory" else T["spinner_prog"]):
        try:
            response = claude.messages.create(
                model="claude-sonnet-4-6", max_tokens=1400,
                system=system_prompt,
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

# ── DISPLAY RESULTS ────────────────────────────────────────────────────────────
if st.session_state.result_ready:

    if app_mode == "Case Advisory":
        st.markdown(f'<div class="step-label">{T["step2"]}</div>', unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
    dl_label = T["download_case"] if app_mode == "Case Advisory" else T["download_prog"]
    dl_file  = T["file_case"]     if app_mode == "Case Advisory" else T["file_prog"]
    dl_header = T["download_header_case"] if app_mode == "Case Advisory" else T["download_header_prog"]
    st.download_button(
        label=dl_label,
        data=f"{dl_header}\n{'='*50}\n\n{T['download_input_label']}:\n{st.session_state.client_case_snapshot}\n\n{'='*50}\n\n{T['download_brief_label']}:\n{st.session_state.advisory}",
        file_name=dl_file,
        mime="text/plain",
        use_container_width=True
    )