"""Fonte única de verdade dos protocolos internos (sintéticos/didáticos).

Usado tanto na geração de dados de fine-tuning (src/preprocessing/synthetic_data.py)
quanto na busca com citação garantida (src/rag/). Os dois nunca divergem porque
leem a mesma lista.
"""
from __future__ import annotations

VERSAO = "v1.0"
AVISO = "Apoio à decisão clínica: a conduta final deve ser validada pelo médico responsável."

PROTOCOLOS = [
    {
        "id": "HSF-PROT-001", "nome": "sepse",
        "indicacao": "Suspeita de infecção associada a disfunção orgânica (ex.: hipotensão, rebaixamento de consciência, oligúria).",
        "exames": ["Lactato", "Hemoculturas antes do antibiótico", "Hemograma", "Creatinina e ureia", "Gasometria"],
        "conduta": [
            "Reconhecer precocemente e classificar gravidade (qSOFA/SOFA).",
            "Coletar lactato e hemoculturas antes de iniciar o antibiótico.",
            "Iniciar antibiótico de amplo espectro na primeira hora, conforme protocolo de antimicrobianos vigente.",
            "Ressuscitação volêmica conforme protocolo e reavaliação frequente da perfusão.",
        ],
        "alertas": ["Hipotensão persistente após reposição volêmica", "Lactato elevado", "Rebaixamento do nível de consciência"],
    },
    {
        "id": "HSF-PROT-002", "nome": "dor torácica / síndrome coronariana aguda",
        "indicacao": "Paciente com dor torácica sugestiva de origem isquêmica.",
        "exames": ["ECG de 12 derivações em até 10 minutos", "Troponina seriada", "Radiografia de tórax", "Eletrólitos"],
        "conduta": [
            "Realizar ECG em até 10 minutos da chegada e monitorizar o paciente.",
            "Solicitar troponina seriada conforme protocolo.",
            "Avaliar antiagregação plaquetária conforme protocolo, checando contraindicações.",
            "Acionar a cardiologia para estratificação de risco.",
        ],
        "alertas": ["Supradesnivelamento do segmento ST", "Instabilidade hemodinâmica", "Arritmia sustentada"],
    },
    {
        "id": "HSF-PROT-003", "nome": "AVC agudo",
        "indicacao": "Déficit neurológico focal de início súbito.",
        "exames": ["Glicemia capilar", "TC de crânio sem contraste", "ECG", "Coagulograma"],
        "conduta": [
            "Registrar o horário de início dos sintomas (ou o último momento visto bem).",
            "Aplicar a escala NIHSS e medir a glicemia capilar.",
            "Realizar TC de crânio sem contraste imediatamente.",
            "Acionar a neurologia para avaliar elegibilidade a terapias de reperfusão dentro da janela.",
        ],
        "alertas": ["Janela terapêutica próxima do limite", "Piora neurológica", "Pressão arterial muito elevada"],
    },
    {
        "id": "HSF-PROT-004", "nome": "cetoacidose diabética",
        "indicacao": "Paciente diabético com hiperglicemia, cetonemia/cetonúria e acidose metabólica.",
        "exames": ["Glicemia", "Gasometria", "Cetonas", "Eletrólitos (incluindo potássio)", "Função renal"],
        "conduta": [
            "Confirmar o diagnóstico com glicemia, gasometria e cetonas.",
            "Dosar o potássio antes de iniciar insulina.",
            "Hidratação venosa conforme protocolo.",
            "Insulinoterapia conforme protocolo e monitorização horária de glicemia e eletrólitos.",
        ],
        "alertas": ["Potássio baixo antes da insulina", "Rebaixamento de consciência", "Acidose grave"],
    },
    {
        "id": "HSF-PROT-005", "nome": "pneumonia adquirida na comunidade",
        "indicacao": "Quadro respiratório agudo com infiltrado pulmonar novo em paciente da comunidade.",
        "exames": ["Radiografia de tórax", "Oximetria", "Hemograma", "Ureia", "Hemocultura se grave"],
        "conduta": [
            "Avaliar gravidade com CURB-65 e oximetria.",
            "Confirmar com radiografia de tórax.",
            "Iniciar antibiótico conforme protocolo de antimicrobianos vigente.",
            "Definir local de tratamento (ambulatorial, enfermaria ou UTI) pelos critérios de internação.",
        ],
        "alertas": ["Saturação de O2 baixa", "Confusão mental", "Frequência respiratória elevada"],
    },
    {
        "id": "HSF-PROT-006", "nome": "crise hipertensiva",
        "indicacao": "Pressão arterial muito elevada, com ou sem sintomas.",
        "exames": ["Creatinina", "Eletrólitos", "ECG", "Urina tipo 1", "Troponina se dor torácica"],
        "conduta": [
            "Diferenciar urgência de emergência hipertensiva pela presença de lesão de órgão-alvo.",
            "Investigar sinais de lesão aguda em coração, cérebro e rins.",
            "Reduzir a pressão de forma controlada, evitando queda abrupta.",
            "Na emergência hipertensiva, monitorizar e acionar a equipe de cuidados intensivos.",
        ],
        "alertas": ["Dor torácica", "Déficit neurológico", "Alteração visual ou confusão"],
    },
]
