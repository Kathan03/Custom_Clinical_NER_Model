import pandas as pd
import numpy as np
import spacy
import en_core_med7_lg
from spacy import displacy
from IPython.display import display, HTML


print(spacy.__version__)


ner_model = en_core_med7_lg.load()
# Model Metrics:
# Precision = 0.865
# Recall = 0.889
# F-score = 0.877


def generate_annotation(texts):
    annotations = []
    for text in texts:
        doc = ner_model(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.start_char, ent.end_char, ent.label_))
        annotations.append((text, {'entities': entities}))
    return annotations


def color_labels():
    col_dict = {}
    s_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
    for label, colour in zip(ner_model.pipe_labels['ner'], s_colours):
        col_dict[label] = colour

    options = {'ents': ner_model.pipe_labels['ner'], 'colors': col_dict}

    return options


def named_ent_extractor(transcription):
    doc = ner_model(transcription)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities, doc


trans = """
Patient Progress Note

Patient Information:
Name: John Doe
Age: 65
Gender: Male
Medical Record Number: 123456789

Chief Complaint:
Severe cough and exacerbation of symptoms.

History of Present Illness:
Mr. John Doe, a 65-year-old male with a known history of COPD, presented to the clinic with a worsening cough over the past week. The cough is productive with greenish sputum, and he reports shortness of breath and fever. He denies any recent travel or contact with individuals known to have respiratory infections.

Medical History:
Chronic Obstructive Pulmonary Disease (COPD)
Hypertension
Type 2 Diabetes Mellitus

Medications:
Albuterol Inhaler
Lisinopril 10 mg daily
Metformin 500 mg twice daily

Allergies:
No known drug allergies

Physical Examination:
Vital Signs: BP 140/85 mmHg, HR 88 bpm, RR 24 breaths/min, Temp 101.2°F, SpO2 92% on room air
General: Appears in mild distress, alert and oriented
Respiratory: Bilateral crackles heard on auscultation, decreased breath sounds at the bases
Cardiovascular: Regular rate and rhythm, no murmurs
Abdomen: Soft, non-tender, no hepatosplenomegaly
Extremities: No edema, pulses 2+ bilaterally

Imaging:
Chest X-ray: Findings consistent with pneumonia, including patchy infiltrates in the right lower lobe

Diagnosis:
Community-acquired pneumonia
Acute exacerbation of COPD

Treatment Plan:
Medications:
Azithromycin 500 mg on day 1, followed by 250 mg once daily for the next 4 days
Prednisone 40 mg daily for 5 days
Continue Albuterol Inhaler as needed for shortness of breath
Acetaminophen 500 mg every 6 hours as needed for fever and discomfort

Supportive Care:
Encourage increased fluid intake
Advise rest and avoidance of strenuous activities
Educate on the importance of completing the full course of antibiotics
Follow-Up:
Follow-up appointment in one week to assess response to treatment and repeat chest X-ray
Instruct patient to return to the clinic or go to the emergency department if symptoms worsen or if there are signs of respiratory distress

Disposition:
Patient was advised to rest at home and contact the clinic with any concerns
Next appointment scheduled for 07/03/2024

Signature:
Dr. Jane Smith, MD

Provider Contact Information:
Clinic Address: 456 Oak Street, Anytown, USA
Phone: (555) 111-2222
Fax: (555) 333-4444

Date and Time of Note Completion:
06/26/202
"""

ent_type, extraction = named_ent_extractor(trans)
opts = color_labels()
print(ent_type)
# displacy.render(extraction, style="ent", options=opts, page=True, minify=True) #Cannot be use in pycharm, only works in jupyterntebook

