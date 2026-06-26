from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import uuid

client = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "social_work_reports"

synthetic_cases = [
    {
        "source": "EDE_SYNTHETIC_CASE_001.pdf",
        "text": "CLIENT NAME: Fatima S. AGE: 24. NATIONALITY: Moroccan (asylum seeker). CURRENT SITUATION: Single woman. Arrived 4 months ago. Living in emergency shelter in Bilbao. Asylum application submitted. Limited Spanish. No family in Spain. BACKGROUND & HISTORY: Secondary education in Morocco. Speaks Arabic and French. Fled due to domestic violence. No prior work experience in formal sector. IDENTIFIED RISK FACTORS: Trauma from domestic violence. Social isolation. Language barrier. No legal work authorization. Limited knowledge of available services. LAST INTERVENTION: Initial intake assessment completed. Referred to legal aid for asylum process. OUTCOME / STATUS: Awaiting asylum decision. Psychological support initiated."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_002.pdf",
        "text": "CLIENT NAME: Omar D. AGE: 31. NATIONALITY: Senegalese. CURRENT SITUATION: Single man. Arrived 8 months ago via irregular route. Residing in municipal shelter. Spanish language classes started. BACKGROUND & HISTORY: Completed primary education in Senegal. Worked as fisherman. Speaks Wolof and basic French. No family in Basque Country. IDENTIFIED RISK FACTORS: Irregular entry. Pending regularization. No formal qualifications recognized in Spain. Risk of homelessness if shelter placement ends. LAST INTERVENTION: Employment orientation session completed. Enrolled in Spanish A1 course. OUTCOME / STATUS: Language integration in progress. Employment prospects limited without documentation."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_003.pdf",
        "text": "CLIENT NAME: Miriam A. AGE: 35. NATIONALITY: Eritrean (refugee status granted). CURRENT SITUATION: Single mother with three children aged 2, 6, and 9. Renting shared flat in Bilbao. Refugee status granted 6 months ago. BACKGROUND & HISTORY: University educated. Former teacher. Speaks Tigrinya, Arabic, and some English. Children enrolled in school. IDENTIFIED RISK FACTORS: Precarious housing. Single parent stress. Limited Spanish affecting employment. Children showing signs of school adjustment difficulties. LAST INTERVENTION: Children referred to school support programme. Spanish B1 course enrolled. Job search support initiated. OUTCOME / STATUS: Stable housing. Language improving. Employment pending."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_004.pdf",
        "text": "CLIENT NAME: Yusuf H. AGE: 19. NATIONALITY: Somali (unaccompanied minor, now 19). CURRENT SITUATION: Recently aged out of youth care system. Transitioning to adult services. No stable housing. BACKGROUND & HISTORY: Arrived at age 15 as unaccompanied minor. Completed ESO in Spain. Speaks Spanish, Somali, and basic English. No family in Spain. IDENTIFIED RISK FACTORS: Transition from youth care. Risk of homelessness. No work experience. Emotional vulnerability due to loss of care placement. LAST INTERVENTION: Emergency housing arranged. Employment training referral made. OUTCOME / STATUS: In transitional housing. Vocational training enrolled."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_005.pdf",
        "text": "CLIENT NAME: Nadia K. AGE: 42. NATIONALITY: Ukrainian (temporary protection). CURRENT SITUATION: Married with two teenage children. Husband remained in Ukraine. Arrived 12 months ago. Renting flat with support. BACKGROUND & HISTORY: Accountant in Ukraine. University educated. Speaks Ukrainian, Russian, and basic English. Children in secondary school. IDENTIFIED RISK FACTORS: Separation from spouse. Mental health strain. Professional qualifications not recognized in Spain. Children adapting to new school system. LAST INTERVENTION: Mental health counselling started. Professional requalification options explored. OUTCOME / STATUS: Stable housing. Seeking employment in financial sector."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_006.pdf",
        "text": "CLIENT NAME: Amadou B. AGE: 27. NATIONALITY: Guinean (asylum seeker). CURRENT SITUATION: Single man. Arrived 6 months ago. Living in reception centre in Donostia. Asylum claim under review. BACKGROUND & HISTORY: Mechanic by trade. Speaks Pular, French, and basic Spanish. Fled political persecution. No criminal record. IDENTIFIED RISK FACTORS: Prolonged asylum process uncertainty. Limited Spanish. No income. Risk of depression due to inactivity. LAST INTERVENTION: Vocational skills assessment completed. Referred to mechanical workshop for informal skills practice. OUTCOME / STATUS: Asylum pending. Psychological wellbeing monitoring ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_007.pdf",
        "text": "CLIENT NAME: Hana T. AGE: 29. NATIONALITY: Ethiopian (humanitarian visa). CURRENT SITUATION: Single woman. Arrived 3 months ago. Staying with distant acquaintance temporarily. Precarious housing situation. BACKGROUND & HISTORY: Nursing background in Ethiopia. Speaks Amharic, Oromo, and English. Motivated to work in healthcare. IDENTIFIED RISK FACTORS: Precarious housing. Professional recognition process lengthy. Risk of exploitation in informal work. Social isolation. LAST INTERVENTION: Housing support application submitted. Healthcare qualification recognition process initiated. OUTCOME / STATUS: Housing pending. Professional pathway under assessment."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_008.pdf",
        "text": "CLIENT NAME: Ibrahim M. AGE: 45. NATIONALITY: Malian (asylum seeker). CURRENT SITUATION: Married. Wife and four children in Mali. Living in shared flat with other asylum seekers in Vitoria-Gasteiz. BACKGROUND & HISTORY: Farmer and trader. Primary education only. Speaks Bambara and basic French. Remitting money to family when possible. IDENTIFIED RISK FACTORS: Family separation. Low educational level. Limited transferable skills for urban labour market. Chronic stress due to family situation. LAST INTERVENTION: Basic literacy support referred. Family reunification options explored. OUTCOME / STATUS: Asylum pending. Family reunification process started."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_009.pdf",
        "text": "CLIENT NAME: Rosa L. AGE: 38. NATIONALITY: Honduran (asylum seeker). CURRENT SITUATION: Single mother with one child aged 7. Fled gang violence. Living in EDE-supported flat. BACKGROUND & HISTORY: Secondary education. Worked as market vendor. Speaks Spanish fluently. Child enrolled in primary school. IDENTIFIED RISK FACTORS: Trauma from gang violence. Child showing behavioural difficulties. Although Spanish-speaking, cultural isolation reported. Fear of deportation affecting mental health. LAST INTERVENTION: Child referred to school psychologist. Legal support for asylum claim strengthened. OUTCOME / STATUS: Asylum claim under review. Child support ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_010.pdf",
        "text": "CLIENT NAME: Kofi A. AGE: 33. NATIONALITY: Ghanaian (work permit). CURRENT SITUATION: Single man. Arrived 2 years ago on work permit. Lost job due to company closure. At risk of losing legal status. BACKGROUND & HISTORY: Electrical engineer. University educated. Speaks English, Twi, and intermediate Spanish. Previously employed in construction sector. IDENTIFIED RISK FACTORS: Risk of legal status lapse. Unemployment. Sector-specific skills may not transfer easily. Financial pressure. LAST INTERVENTION: Legal advice on permit renewal obtained. Job search support in engineering and construction sectors started. OUTCOME / STATUS: Permit renewal in process. Active job search ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_011.pdf",
        "text": "CLIENT NAME: Aisha F. AGE: 22. NATIONALITY: Nigerian (asylum seeker). CURRENT SITUATION: Single woman. Possible victim of trafficking. Referred by NGO. Living in safe house. BACKGROUND & HISTORY: Limited formal education. Speaks Igbo, Pidgin English. Arrived via irregular route through Libya. Trauma history suspected. IDENTIFIED RISK FACTORS: Suspected trafficking victim. Severe trauma. No documentation. High vulnerability. Distrust of authorities. LAST INTERVENTION: Specialist anti-trafficking support engaged. Psychological trauma assessment initiated. Legal protection measures applied. OUTCOME / STATUS: Under specialist protection. Asylum application being prepared."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_012.pdf",
        "text": "CLIENT NAME: Pita V. AGE: 51. NATIONALITY: Bolivian (irregular status). CURRENT SITUATION: Married couple, both irregular. Three adult children in Bolivia. Working informally in domestic service. BACKGROUND & HISTORY: Primary education. Worked as domestic worker for 8 years in Spain. Speaks Spanish. Health issues emerging. IDENTIFIED RISK FACTORS: Irregular status after long residence. Health problems without social security access. Exploitation risk in informal employment. Age-related employment barriers. LAST INTERVENTION: Legal regularization options explored. Health referral to community clinic. OUTCOME / STATUS: Regularization process initiated. Health support ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_013.pdf",
        "text": "CLIENT NAME: Tariq N. AGE: 26. NATIONALITY: Pakistani (student visa overstay). CURRENT SITUATION: Single man. Student visa expired. Working informally. Afraid to approach authorities. BACKGROUND & HISTORY: University degree in IT from Pakistan. Speaks Urdu, English, and basic Spanish. Highly motivated. No criminal record. IDENTIFIED RISK FACTORS: Irregular status. Fear of deportation. Informal work exploitation. Isolation from Pakistani community due to shame. LAST INTERVENTION: Legal options for regularization assessed. Psychological support for anxiety initiated. OUTCOME / STATUS: Exploring exceptional circumstances regularization route."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_014.pdf",
        "text": "CLIENT NAME: Blessing O. AGE: 30. NATIONALITY: Nigerian (humanitarian protection). CURRENT SITUATION: Single mother with infant (8 months). Protection status granted. Living in EDE transitional flat. BACKGROUND & HISTORY: Secondary education. Worked as hairdresser. Speaks Yoruba, English, and basic Spanish. Infant requires regular medical appointments. IDENTIFIED RISK FACTORS: Single parent with infant. Limited Spanish affecting integration. Childcare barriers to employment. Social isolation. LAST INTERVENTION: Childcare support arranged. Spanish language classes with creche provision enrolled. OUTCOME / STATUS: Language integration progressing. Employment to be explored once childcare stable."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_015.pdf",
        "text": "CLIENT NAME: Ahmed Z. AGE: 48. NATIONALITY: Algerian (long-term resident). CURRENT SITUATION: Married. Three children (14, 17, 20). Long-term resident for 15 years. Recently divorced. Housing instability following divorce. BACKGROUND & HISTORY: Worked in hospitality for 12 years. Speaks Arabic, French, and Spanish. Children integrated and in education. Recently lost job. IDENTIFIED RISK FACTORS: Post-divorce housing instability. Unemployment after long employment history. Mental health strain. Risk of social exclusion despite long residence. LAST INTERVENTION: Housing support application submitted. Mental health referral made. Job search support initiated. OUTCOME / STATUS: Temporary housing arranged. Mental health support ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_016.pdf",
        "text": "CLIENT NAME: Svetlana P. AGE: 39. NATIONALITY: Moldovan (temporary protection). CURRENT SITUATION: Single woman. Arrived 7 months ago. Working part-time in cleaning sector. Renting room. BACKGROUND & HISTORY: Secondary education. Worked as nurse in Moldova. Speaks Romanian, Russian, and basic Spanish. No family in Spain. IDENTIFIED RISK FACTORS: Healthcare qualification not recognized. Part-time informal work insufficient for stability. Social isolation. Risk of exploitation. LAST INTERVENTION: Professional recognition process started. Full-time employment search support initiated. OUTCOME / STATUS: Working part-time. Recognition process ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_017.pdf",
        "text": "CLIENT NAME: Moussa D. AGE: 23. NATIONALITY: Mauritanian (asylum seeker). CURRENT SITUATION: Single man. Arrived 2 months ago. Reception centre in Bilbao. Very limited Spanish. Presenting signs of depression. BACKGROUND & HISTORY: Secondary education incomplete. Speaks Arabic and Hassaniya. Fled forced labour situation. No prior migration experience. IDENTIFIED RISK FACTORS: Trauma from forced labour. Severe language barrier. Depression symptoms. Young and isolated. LAST INTERVENTION: Mental health assessment completed. Arabic-speaking counsellor engaged. Spanish A1 enrolment pending. OUTCOME / STATUS: Psychological support initiated. Asylum claim being prepared."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_018.pdf",
        "text": "CLIENT NAME: Lena B. AGE: 44. NATIONALITY: Belarusian (asylum seeker). CURRENT SITUATION: Single woman. Political activist. Fled following arrest threat. Living in EDE-supported flat in Donostia. BACKGROUND & HISTORY: University educated. Journalist by profession. Speaks Russian, Belarusian, and intermediate English. Highly articulate. IDENTIFIED RISK FACTORS: Political persecution trauma. Professional skills not easily transferable. Language barrier in Spanish. Risk of re-traumatisation. LAST INTERVENTION: Asylum application filed. Spanish language B1 course enrolled. Journalism network contacts explored. OUTCOME / STATUS: Asylum pending. Language integration progressing rapidly."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_019.pdf",
        "text": "CLIENT NAME: Grace N. AGE: 34. NATIONALITY: Cameroonian (asylum seeker). CURRENT SITUATION: Married. Husband in different European country. Two children aged 4 and 8 with her. Anglophone region origin. BACKGROUND & HISTORY: Teacher by profession. University educated. Speaks English, French, and basic Spanish. Children in school. IDENTIFIED RISK FACTORS: Family separation. Single parenting in practice. Children's school adjustment. Limited Spanish affecting employment. LAST INTERVENTION: Family reunification legal options explored. Children enrolled in school support. Spanish course started. OUTCOME / STATUS: Asylum pending. Family reunification application in progress."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_020.pdf",
        "text": "CLIENT NAME: Diego R. AGE: 29. NATIONALITY: Venezuelan (asylum seeker). CURRENT SITUATION: Single man. Arrived 5 months ago. Shares flat with other Venezuelans. Working informally in food delivery. BACKGROUND & HISTORY: University degree in economics. Speaks Spanish natively. Politically active in Venezuela. IDENTIFIED RISK FACTORS: Irregular work status despite language fluency. Professional qualification recognition needed. Risk of remaining in informal economy. Mild anxiety reported. LAST INTERVENTION: Asylum application supported. University qualification recognition process started. Formal employment search initiated. OUTCOME / STATUS: Asylum pending. Qualification recognition in process."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_021.pdf",
        "text": "CLIENT NAME: Amira H. AGE: 31. NATIONALITY: Sudanese (refugee status). CURRENT SITUATION: Single woman. Refugee status granted 3 months ago. Living in shared flat. Presents with significant trauma symptoms. BACKGROUND & HISTORY: Secondary education. Fled conflict and sexual violence. Speaks Arabic and basic English. No family in Spain. IDENTIFIED RISK FACTORS: Severe trauma including gender-based violence. Social isolation. Nightmares and anxiety reported. Limited Spanish. LAST INTERVENTION: Specialist trauma therapy initiated. Women's support group referral made. Spanish A2 course enrolled. OUTCOME / STATUS: Therapy ongoing. Gradual social integration beginning."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_022.pdf",
        "text": "CLIENT NAME: Samuel T. AGE: 55. NATIONALITY: Congolese (long-term resident). CURRENT SITUATION: Widower. Two adult children in Congo. Long-term resident 20 years. Lost housing after landlord sold property. BACKGROUND & HISTORY: Former driver. Primary education. Speaks Lingala, French, and Spanish. Health problems emerging. Pension not yet accessible. IDENTIFIED RISK FACTORS: Age-related employment barriers. Health problems. Housing instability. Social network reduced after wife's death. LAST INTERVENTION: Emergency housing support provided. Health referral completed. Social activation programme referral made. OUTCOME / STATUS: Temporary housing stable. Health treatment ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_023.pdf",
        "text": "CLIENT NAME: Zara M. AGE: 20. NATIONALITY: Afghan (refugee status). CURRENT SITUATION: Single woman. Arrived as unaccompanied minor at 17. Now adult. Completed ESO. Limited work experience. BACKGROUND & HISTORY: Speaks Dari, Pashto, and intermediate Spanish. Strong academic record in Spain. Motivated to study further. No family in Spain. IDENTIFIED RISK FACTORS: Transition to adulthood without family support. Financial pressures. Risk of dropping education for work. Emotional vulnerability. LAST INTERVENTION: University access support explored. Scholarship options identified. Mentoring programme enrolled. OUTCOME / STATUS: University application in progress. Mentoring ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_024.pdf",
        "text": "CLIENT NAME: Joseph K. AGE: 37. NATIONALITY: Kenyan (work permit). CURRENT SITUATION: Married. Wife and two children in Kenya. Working in agriculture sector in Álava. Seasonal contract ending. BACKGROUND & HISTORY: Secondary education. Experienced agricultural worker. Speaks Swahili, English, and basic Spanish. Sends remittances to family. IDENTIFIED RISK FACTORS: Seasonal employment insecurity. Family separation. Limited Spanish affecting year-round employment options. Housing tied to employer. LAST INTERVENTION: Year-round employment options explored. Spanish language support referred. Family reunification options assessed. OUTCOME / STATUS: Employment search ongoing. Family reunification application being considered."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_025.pdf",
        "text": "CLIENT NAME: Maricel S. AGE: 46. NATIONALITY: Filipino (long-term resident). CURRENT SITUATION: Single. Long-term resident 18 years. Works as live-in carer. Employer passed away. Suddenly homeless and unemployed. BACKGROUND & HISTORY: University educated nurse. Speaks Filipino, English, and Spanish. Extensive caregiving experience. Strong work record. IDENTIFIED RISK FACTORS: Sudden homelessness. Loss of income and housing simultaneously. Emotional grief over employer. Age-related employment barriers in competitive market. LAST INTERVENTION: Emergency housing provided. Employment search in healthcare and caregiving initiated. Grief support offered. OUTCOME / STATUS: Temporary housing stable. Job search active."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_026.pdf",
        "text": "CLIENT NAME: Abdi W. AGE: 28. NATIONALITY: Somali (refugee status). CURRENT SITUATION: Single man. Refugee status granted. Living independently in Bilbao. Struggling with isolation and purpose. BACKGROUND & HISTORY: Secondary education incomplete due to conflict. Speaks Somali, Arabic, and intermediate Spanish. Strong interest in IT and technology. IDENTIFIED RISK FACTORS: Social isolation. Lack of formal qualifications. Risk of radicalisation flagged by community worker — low level concern. Underemployment. LAST INTERVENTION: IT training programme enrolled. Community social activities referred. Mentor assigned from Somali diaspora. OUTCOME / STATUS: Training progressing. Social integration improving."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_027.pdf",
        "text": "CLIENT NAME: Fatou B. AGE: 25. NATIONALITY: Gambian (asylum seeker). CURRENT SITUATION: Single woman. Three months pregnant. Living in reception centre. Asylum pending. No partner present. BACKGROUND & HISTORY: Primary education only. Speaks Mandinka and basic French. Fled forced marriage. First time outside Gambia. IDENTIFIED RISK FACTORS: Pregnancy without social support. Forced marriage trauma. Very limited language skills. Uncertainty about asylum outcome affecting mental health. LAST INTERVENTION: Prenatal health referral completed. Specialist women's asylum support engaged. Language support via interpreter arranged. OUTCOME / STATUS: Health monitoring ongoing. Asylum claim being built with legal support."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_028.pdf",
        "text": "CLIENT NAME: Petro H. AGE: 41. NATIONALITY: Ukrainian (temporary protection). CURRENT SITUATION: Married. Wife and daughter (16) with him. Son (21) stayed in Ukraine. Working part-time in warehouse. BACKGROUND & HISTORY: Civil engineer. University educated. Speaks Ukrainian, Russian, and basic Spanish. Wife experiencing severe depression. IDENTIFIED RISK FACTORS: Wife's mental health. Professional qualification not recognized. Part-time work insufficient. Worry about son in Ukraine. Teenage daughter struggling in school. LAST INTERVENTION: Wife referred to mental health services. Professional requalification explored. Daughter enrolled in school integration support. OUTCOME / STATUS: Wife receiving treatment. Family partially stabilised."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_029.pdf",
        "text": "CLIENT NAME: Mariam D. AGE: 32. NATIONALITY: Syrian (refugee status). CURRENT SITUATION: Married with husband. No children. Both have refugee status. Living in EDE flat. Both seeking work. BACKGROUND & HISTORY: Both university educated — she is pharmacist, he is architect. Speaks Arabic, English, and intermediate Spanish. IDENTIFIED RISK FACTORS: Professional recognition process lengthy. Both unemployed causing financial and emotional strain. Risk of accepting far below qualification level work permanently. LAST INTERVENTION: Professional recognition applications filed for both. Intermediate employment options in related fields explored. OUTCOME / STATUS: Recognition pending. Both in Spanish B2 course."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_030.pdf",
        "text": "CLIENT NAME: Emmanuel O. AGE: 22. NATIONALITY: Nigerian (asylum seeker). CURRENT SITUATION: Single man. Arrived 1 month ago. Reception centre. No Spanish. Presenting with acute stress. BACKGROUND & HISTORY: Secondary education. Speaks Hausa, English. Fled religious persecution. First time outside Nigeria. IDENTIFIED RISK FACTORS: Very recent arrival — high vulnerability period. Acute stress symptoms. No Spanish. Cultural adjustment challenges. LAST INTERVENTION: Initial intake completed. Medical check done. Spanish A1 course enrolled. Peer support from Nigerian community arranged. OUTCOME / STATUS: Settling in. Asylum claim being prepared."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_031.pdf",
        "text": "CLIENT NAME: Ana C. AGE: 36. NATIONALITY: Colombian (asylum seeker). CURRENT SITUATION: Single mother with two children aged 5 and 11. Fled threats from armed group. Living in EDE-supported flat. BACKGROUND & HISTORY: Secondary education. Worked as hairdresser. Speaks Spanish natively. Children in school. IDENTIFIED RISK FACTORS: Trauma from threats and displacement. Children show anxiety symptoms. Despite Spanish fluency, cultural adjustment needed. Fear of being found. LAST INTERVENTION: Security protocol reviewed with legal team. Children referred to school counsellor. Psychological support for mother initiated. OUTCOME / STATUS: Asylum claim submitted. Psychological support ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_032.pdf",
        "text": "CLIENT NAME: Hassan A. AGE: 17. NATIONALITY: Moroccan (unaccompanied minor). CURRENT SITUATION: Unaccompanied minor. In youth care centre in Bilbao. Has been in Spain 6 months. BACKGROUND & HISTORY: Primary education incomplete. Speaks Arabic, Darija, and basic Spanish. Arrived via Strait of Gibraltar. Strong motivation to work and study. IDENTIFIED RISK FACTORS: Minor without parental protection. Irregular entry. Risk of exploitation when ageing out of system. Education gaps. LAST INTERVENTION: Legal guardian assigned. School enrolment completed. Vocational pathway assessment initiated. OUTCOME / STATUS: Stable in youth care. School progressing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_033.pdf",
        "text": "CLIENT NAME: Olga M. AGE: 52. NATIONALITY: Russian (asylum seeker). CURRENT SITUATION: Single woman. Political dissident. Arrived 4 months ago. EDE-supported flat. Limited Spanish. BACKGROUND & HISTORY: University educated academic. Speaks Russian, English, and basic Spanish. Published researcher. Fled following intimidation. IDENTIFIED RISK FACTORS: High-profile political case adding stress. Professional identity disrupted. Age-related employment barriers. Social isolation in new country. LAST INTERVENTION: Asylum application filed with specialist legal support. Academic network contacts explored. Spanish B1 enrolled. OUTCOME / STATUS: Asylum pending. Exploring remote academic work possibilities."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_034.pdf",
        "text": "CLIENT NAME: Cheikh D. AGE: 40. NATIONALITY: Senegalese (long-term resident). CURRENT SITUATION: Married with four children. Long-term resident 12 years. Small street vending business disrupted by new local regulations. Income severely reduced. BACKGROUND & HISTORY: Primary education. Speaks Wolof, French, and Spanish. Active in local Senegalese community association. IDENTIFIED RISK FACTORS: Income instability. Business regulatory barriers. Dependent family. Risk of debt. LAST INTERVENTION: Business regularisation advice provided. Alternative income options explored. Micro-credit application supported. OUTCOME / STATUS: Exploring formalising small business. Micro-credit application pending."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_035.pdf",
        "text": "CLIENT NAME: Linh N. AGE: 27. NATIONALITY: Vietnamese (irregular status). CURRENT SITUATION: Single woman. Arrived 2 years ago. Working informally in nail salon sector. Overstayed tourist visa. BACKGROUND & HISTORY: Secondary education. Speaks Vietnamese and basic Spanish. Sends remittances to family in Vietnam. IDENTIFIED RISK FACTORS: Irregular status. Exploitation risk in informal employment sector. Debt to travel agent. Social isolation within Vietnamese community due to fear. LAST INTERVENTION: Legal regularisation options assessed. Labour exploitation support referral made. OUTCOME / STATUS: Regularisation pathway identified. Labour rights support ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_036.pdf",
        "text": "CLIENT NAME: Kwame A. AGE: 35. NATIONALITY: Ghanaian (asylum seeker). CURRENT SITUATION: Single man. Asylum seeker. Living in reception centre in Vitoria-Gasteiz. Previously deported once, returned via different route. BACKGROUND & HISTORY: Secondary education. Speaks Twi, English, and basic Spanish. Skilled carpenter. IDENTIFIED RISK FACTORS: Previous deportation complicating asylum claim. Legal complexity. Skilled trade not easily recognised. Mild post-traumatic symptoms. LAST INTERVENTION: Specialist legal advice on complex asylum case obtained. Skills assessment for carpentry completed. OUTCOME / STATUS: Legal case being carefully built. Skills portfolio being prepared."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_037.pdf",
        "text": "CLIENT NAME: Valentina R. AGE: 23. NATIONALITY: Venezuelan (asylum seeker). CURRENT SITUATION: Single woman. University student before fleeing. Living in shared flat. Part-time informal work. BACKGROUND & HISTORY: Completed 2 years of medicine degree. Speaks Spanish natively and basic English. Highly motivated to continue studies. IDENTIFIED RISK FACTORS: Interrupted university education. Informal work taking time from studies. Emotional impact of abandoned professional dream. Financial instability. LAST INTERVENTION: University re-entry options explored with academic institutions. Scholarship search initiated. Part-time formal employment sought. OUTCOME / STATUS: University re-entry pathway identified. Scholarship application in progress."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_038.pdf",
        "text": "CLIENT NAME: Abdul R. AGE: 50. NATIONALITY: Afghan (refugee status). CURRENT SITUATION: Married with wife and two children (8 and 12). Refugee status. Living in EDE flat. Struggling with role reversal — children integrating faster than parents. BACKGROUND & HISTORY: Former government official. University educated. Speaks Dari, Pashto, and basic Spanish. Children fluent in Spanish. IDENTIFIED RISK FACTORS: Professional identity loss. Role reversal causing family tension. Depression symptoms in father. Wife very isolated. Language barrier for both parents. LAST INTERVENTION: Family counselling initiated. Father enrolled in Spanish B1. Mother enrolled in women's social group. OUTCOME / STATUS: Family counselling ongoing. Language integration progressing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_039.pdf",
        "text": "CLIENT NAME: Priya S. AGE: 28. NATIONALITY: Sri Lankan (asylum seeker). CURRENT SITUATION: Single woman. Tamil origin. Fled ethnic persecution. Living in EDE-supported flat in Bilbao. BACKGROUND & HISTORY: University educated IT professional. Speaks Tamil, English, and basic Spanish. Strong technical skills. IDENTIFIED RISK FACTORS: Ethnic persecution trauma. Professional skills not immediately deployable due to language. Social isolation. LAST INTERVENTION: Asylum application filed. IT sector employment explored for English-language roles. Spanish A2 course enrolled. OUTCOME / STATUS: Asylum pending. IT employment prospects being explored."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_040.pdf",
        "text": "CLIENT NAME: Mamadou C. AGE: 19. NATIONALITY: Guinean (unaccompanied minor, now 19). CURRENT SITUATION: Recently aged out of youth care. No stable housing. No work experience. BACKGROUND & HISTORY: Primary education incomplete. Speaks Pular, French, and intermediate Spanish. Artistic talent — draws and paints. IDENTIFIED RISK FACTORS: Transition from youth care without family support. No qualifications. Risk of homelessness and exploitation. LAST INTERVENTION: Transitional housing arranged. Creative skills assessed for vocational pathway. Basic employment skills training enrolled. OUTCOME / STATUS: Housing stable short-term. Vocational direction being established."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_041.pdf",
        "text": "CLIENT NAME: Irina V. AGE: 33. NATIONALITY: Ukrainian (temporary protection). CURRENT SITUATION: Single mother with daughter (6). Husband killed in conflict. Arrived 9 months ago. Working part-time in cleaning. BACKGROUND & HISTORY: Secondary education. Worked as shop manager. Speaks Ukrainian, Russian, and basic Spanish. Daughter in school. IDENTIFIED RISK FACTORS: Grief and trauma from husband's death. Single parent stress. Part-time income insufficient. Daughter showing emotional difficulties. LAST INTERVENTION: Grief counselling initiated. Daughter referred to school psychologist. Full-time employment search started. OUTCOME / STATUS: Grief support ongoing. Daughter improving. Employment search active."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_042.pdf",
        "text": "CLIENT NAME: Oumar S. AGE: 44. NATIONALITY: Malian (long-term resident). CURRENT SITUATION: Married with three children. Long-term resident 10 years. Recently diagnosed with chronic illness. Unable to continue manual labour work. BACKGROUND & HISTORY: Primary education. Worked in construction for 10 years. Speaks Bambara, French, and Spanish. Wife works part-time. IDENTIFIED RISK FACTORS: Health-related work incapacity. Family income reduced. Retraining needed. Psychological impact of illness on family. LAST INTERVENTION: Disability benefit application initiated. Retraining options for lighter work explored. Family support referral made. OUTCOME / STATUS: Benefit application pending. Retraining options under assessment."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_043.pdf",
        "text": "CLIENT NAME: Dina A. AGE: 26. NATIONALITY: Jordanian (student visa, overstay). CURRENT SITUATION: Single woman. Completed master's degree. Visa expired. Seeking regularisation. Working informally as tutor. BACKGROUND & HISTORY: University educated. Speaks Arabic, English, and advanced Spanish. Strong academic record. IDENTIFIED RISK FACTORS: Irregular status despite high qualifications. Risk of deportation. Underemployed relative to qualification level. LAST INTERVENTION: Exceptional circumstances regularisation explored. Formal employment in education sector sought. OUTCOME / STATUS: Legal pathway identified. Job applications in education sector submitted."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_044.pdf",
        "text": "CLIENT NAME: Femi A. AGE: 38. NATIONALITY: Nigerian (humanitarian protection). CURRENT SITUATION: Single man. Protection status. Living independently. Employed part-time in security sector. BACKGROUND & HISTORY: Secondary education. Worked as security guard in Nigeria. Speaks Yoruba, English, and intermediate Spanish. Stable but underemployed. IDENTIFIED RISK FACTORS: Underemployment relative to capacity. Social isolation. Risk of stagnation without further development. LAST INTERVENTION: Professional development options explored. Full-time security employment and career progression pathway assessed. Community engagement encouraged. OUTCOME / STATUS: Stable. Career development plan being built."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_045.pdf",
        "text": "CLIENT NAME: Catalina M. AGE: 41. NATIONALITY: Ecuadorian (long-term resident). CURRENT SITUATION: Married. Two teenage children. Long-term resident 16 years. Husband recently imprisoned. Sole income provider. BACKGROUND & HISTORY: Secondary education. Works as domestic worker. Speaks Spanish natively. Children in secondary school. IDENTIFIED RISK FACTORS: Sudden sole provider role. Income insufficient for family. Children's emotional impact of father's imprisonment. Risk of older child dropping out of school. LAST INTERVENTION: Financial support application submitted. Children referred to counselling. Employment hours increase explored. OUTCOME / STATUS: Financial support approved. Children in counselling."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_046.pdf",
        "text": "CLIENT NAME: Rashid A. AGE: 34. NATIONALITY: Iraqi (refugee status). CURRENT SITUATION: Married with wife and infant (4 months). Refugee status granted. Wife experiencing postpartum depression. BACKGROUND & HISTORY: Secondary education. Worked as mechanic. Speaks Arabic and basic Spanish. Wife speaks no Spanish. IDENTIFIED RISK FACTORS: Wife's postpartum depression. Infant care demands. Language barrier for wife. Limited income. Social isolation of wife. LAST INTERVENTION: Wife referred to specialist postpartum mental health support. Arabic-speaking support worker assigned. Infant health monitoring confirmed. OUTCOME / STATUS: Wife receiving treatment. Family support ongoing."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_047.pdf",
        "text": "CLIENT NAME: Bintou K. AGE: 29. NATIONALITY: Senegalese (asylum seeker). CURRENT SITUATION: Single woman. Victim of female genital mutilation. Fled risk of forced marriage. Living in women's shelter. BACKGROUND & HISTORY: Secondary education. Speaks Wolof, French, and basic Spanish. Courageous and determined personality noted by caseworker. IDENTIFIED RISK FACTORS: FGM trauma. Forced marriage risk. Cultural isolation. Health complications from FGM requiring medical attention. LAST INTERVENTION: Specialist FGM health referral completed. Asylum claim built around gender persecution. Psychological support initiated. OUTCOME / STATUS: Medical treatment ongoing. Asylum claim under review."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_048.pdf",
        "text": "CLIENT NAME: Marco T. AGE: 31. NATIONALITY: Brazilian (irregular status). CURRENT SITUATION: Single man. Arrived 3 years ago on tourist visa. Working informally in hospitality. Well integrated socially but legally vulnerable. BACKGROUND & HISTORY: University degree in tourism management. Speaks Portuguese, Spanish, and English. Strong social network in Bilbao. IDENTIFIED RISK FACTORS: Long-term irregular status. Risk of exploitation continuing. Despite integration, legal vulnerability. LAST INTERVENTION: Regularisation options including EU family link explored. Formal employment in tourism sector sought with employer willing to sponsor permit. OUTCOME / STATUS: Employer sponsorship pathway being explored."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_049.pdf",
        "text": "CLIENT NAME: Awa D. AGE: 21. NATIONALITY: Guinean (asylum seeker). CURRENT SITUATION: Single woman. Pregnant (5 months). No partner. Living in reception centre. Speaks no Spanish. BACKGROUND & HISTORY: Primary education incomplete. Speaks Pular only. Never left Guinea before. Extremely vulnerable first-time migrant. IDENTIFIED RISK FACTORS: Pregnancy. No Spanish. Extreme vulnerability. Complete absence of support network. Cultural shock. LAST INTERVENTION: Interpreter (Pular) assigned. Prenatal care referral completed. Specialist vulnerable women's asylum support engaged. OUTCOME / STATUS: Health stable. Asylum claim being prepared with full interpreter support."
    },
    {
        "source": "EDE_SYNTHETIC_CASE_050.pdf",
        "text": "CLIENT NAME: Nikolai B. AGE: 36. NATIONALITY: Belarusian (asylum seeker). CURRENT SITUATION: Married. Wife and two children (4 and 7) with him. Political activist. Fled after arrest. Living in EDE-supported flat. BACKGROUND & HISTORY: University educated software developer. Speaks Russian, Belarusian, English, and basic Spanish. Strong technical skills. IDENTIFIED RISK FACTORS: Political persecution trauma. Family all affected. Children's school adjustment. Professional skills deployable but language barrier remains. LAST INTERVENTION: Asylum application filed. Children in school. Remote software work explored while language develops. Spanish A2 enrolled. OUTCOME / STATUS: Asylum pending. Remote work options being explored. Family adjusting."
    },
]

print(f"Generating embeddings for {len(synthetic_cases)} synthetic cases...")
texts = [c["text"] for c in synthetic_cases]
embeddings = embedder.encode(texts, show_progress_bar=True)

points = []
for i, (case, embedding) in enumerate(zip(synthetic_cases, embeddings)):
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={"text": case["text"], "source": case["source"]}
    ))

print(f"\nUploading {len(points)} points to Qdrant...")
batch_size = 50
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    client.upsert(collection_name=COLLECTION, points=batch)
    print(f"  Uploaded {min(i+batch_size, len(points))}/{len(points)} points...")

collection_info = client.get_collection(COLLECTION)
print(f"\nDone! Collection now has {collection_info.points_count} total points.")