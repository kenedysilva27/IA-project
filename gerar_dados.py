# generate_pacientes_csv.py
import pandas as pd
import numpy as np
from faker import Faker
import random

# Inicializar Faker para gerar dados realistas
fake = Faker('pt_BR') # Usar localidade brasileira para nomes e endereços

def generate_patient_data(num_records=1000):
    """
    Gera dados sintéticos de pacientes para um arquivo CSV.
    """
    data = []
    
    # Listas de valores possíveis para as colunas
    diagnoses = [
        "Hipertensão Essencial", "Diabetes Mellitus Tipo 2", "Asma", "Doença Arterial Coronariana",
        "Insuficiência Cardíaca Congestiva", "Acidente Vascular Cerebral", "Doença Pulmonar Obstrutiva Crônica (DPOC)",
        "Infecção do Trato Urinário", "Pneumonia", "Gastroenterite Aguda", "Apneia do Sono", "Artrite Reumatoide",
        "Depressão", "Ansiedade", "Câncer de Mama", "Câncer de Próstata", "Fratura de Fêmur", "Cálculo Renal"
    ]
    
    comorbidities_list = [
        "Nenhuma", "Obesidade", "Doença Renal Crônica", "Doença Hepática", "Doença de Parkinson",
        "Demência", "Doença de Alzheimer", "HIV/AIDS", "Fibromialgia", "Síndrome do Intestino Irritável"
    ]
    
    procedures = [
        "Nenhum", "Cateterismo Cardíaco", "Cirurgia de Apendicectomia", "Angioplastia", "Biopsia",
        "Endoscopia", "Colonoscopia", "Cirurgia de Catarata", "Fisioterapia Respiratória", "Hemodiálise"
    ]
    
    medications = [
        "Nenhuma", "Losartana", "Metformina", "Salbutamol", "Aspirina", "Furosemida",
        "Sinvastatina", "Omeprazol", "Paracetamol", "Insulina", "Sertralina", "Fluoxetina"
    ]
    
    health_plans = [
        "Unimed Nacional", "Unimed Multi", "Unimed Essencial", "Unimed Pleno", "Unimed Total"
    ]
    
    exam_results = [
        "Estável", "Melhora", "Piora", "Sem Alteração Significativa"
    ]

    for i in range(1, num_records + 1):
        age = random.randint(18, 90)
        sex = random.choice(['Masculino', 'Feminino'])
        diagnosis = random.choice(diagnoses)
        
        # Gerar comorbidades, pode ser uma lista ou 'Nenhuma'
        num_comorbidities = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
        comorbidities = ", ".join(random.sample(comorbidities_list[1:], num_comorbidities)) if num_comorbidities > 0 else "Nenhuma"
        
        duration_stay = random.randint(1, 30) # Duração da internação em dias
        hospitalizations_last_year = random.randint(0, 5) # Quantidade de internações no último ano
        
        # Maior probabilidade de necessitar homecare para idosos ou com estadias longas
        necessity_homecare = False
        if age > 65 or duration_stay > 15:
            necessity_homecare = random.choices([True, False], weights=[0.6, 0.4], k=1)[0]
        else:
            necessity_homecare = random.choices([True, False], weights=[0.1, 0.9], k=1)[0]
            
        medication_post_discharge = ", ".join(random.sample(medications, random.randint(0, 3)))
        if medication_post_discharge == "":
            medication_post_discharge = "Nenhuma"
            
        # Maior probabilidade de readmissão para pacientes com múltiplas internações ou comorbidades
        readmission_30_days = False
        if hospitalizations_last_year > 1 or num_comorbidities > 1:
            readmission_30_days = random.choices([True, False], weights=[0.4, 0.6], k=1)[0]
        else:
            readmission_30_days = random.choices([True, False], weights=[0.1, 0.9], k=1)[0]

        health_plan = random.choice(health_plans)
        charlson_score = random.randint(0, 10) # Pontuação Charlson de comorbidade
        
        telemonitoring = False
        if charlson_score > 5 or necessity_homecare: # Maior probabilidade de telemonitoramento para escores altos ou homecare
            telemonitoring = random.choices([True, False], weights=[0.7, 0.3], k=1)[0]
        else:
            telemonitoring = random.choices([True, False], weights=[0.2, 0.8], k=1)[0]
            
        important_exam_result = random.choice(exam_results)
        
        num_procedures = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2], k=1)[0]
        performed_procedures = ", ".join(random.sample(procedures[1:], num_procedures)) if num_procedures > 0 else "Nenhum"


        data.append({
            "ID_Paciente": i,
            "Idade": age,
            "Sexo": sex,
            "Diagnostico_Principal": diagnosis,
            "Comorbidades": comorbidities,
            "Duracao_Internacao_Dias": duration_stay,
            "Qtd_Internacoes_Ult_Ano": hospitalizations_last_year,
            "Necessidade_Homecare": necessity_homecare,
            "Medicacao_Pos_Alta": medication_post_discharge,
            "Readmissao_30_Dias": readmission_30_days,
            "Plano_Saude": health_plan,
            "Score_Comorbidade_Charlson": charlson_score,
            "Telemonitoramento": telemonitoring,
            "Resultado_Exames_Importantes": important_exam_result,
            "Procedimentos_Realizados": performed_procedures
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df_patients = generate_patient_data(num_records=1000)
    df_patients.to_csv("pacientes.csv", index=False, encoding='utf-8')
    print("Arquivo 'pacientes.csv' com 1000 registros gerado com sucesso!")

