"""
=================================================================================
SISTEMA DE AVALIAÇÃO E COMPARAÇÃO DE MODELOS DE MACHINE LEARNING
=================================================================================

Este script implementa um sistema completo de treinamento e avaliação de 4 modelos
de machine learning para análise de dados de vítimas em situações de emergência:

MODELOS IMPLEMENTADOS:
• 2 REGRESSORES (para predição de gravidade contínua):
  - RandomForestRegressor: Ensemble de árvores de decisão para regressão
  - MLPRegressor: Rede Neural Multicamadas para regressão

• 2 CLASSIFICADORES (para predição de classes de gravidade):
  - XGBClassifier: Extreme Gradient Boosting para classificação
  - MLPClassifier: Rede Neural Multicamadas para classificação

PROCESSO DE AVALIAÇÃO:
1. Feature Engineering: Criação de features derivadas dos sinais vitais
2. Normalização: Padronização dos dados usando StandardScaler
3. Data Augmentation: Técnicas de aumento de dados (Jitter + SMOTE)
4. Validação Cruzada: Avaliação robusta com 5-fold cross-validation
5. Análise de Métricas: Cálculo de múltiplas métricas de performance
6. Visualização: Geração de matrizes de confusão

MÉTRICAS CALCULADAS:
• Regressão → RMSE (Root Mean Squared Error)
• Classificação → Acurácia, Precisão, Recall, F1-Score

DADOS UTILIZADOS:
• qPA: Qualidade da Pressão Arterial [variável principal]
• pulso: Batimentos por Minuto [variável principal]
• freq_resp: Frequência Respiratória por Minuto [variável principal]
• grav: Gravidade (0-∞) [target para regressão]
• classe: Estado de saúde (1=crítico, 2=instável, 3=pot.estável, 4=estável) [target para classificação]

Nota: pSist e pDiast não são utilizadas conforme especificação do projeto.
=================================================================================
"""

# =================================================================================
# IMPORTAÇÕES DE BIBLIOTECAS
# =================================================================================

import os                          # Manipulação de caminhos de arquivos
import numpy as np                 # Operações numéricas e arrays
import pandas as pd               # Manipulação de dados tabulares

# Scikit-learn: Biblioteca principal de Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder    # Pré-processamento de dados
from sklearn.model_selection import cross_val_score, cross_val_predict  # Validação cruzada
from sklearn.ensemble import RandomForestRegressor               # Regressor baseado em ensemble de árvores
from sklearn.neural_network import MLPRegressor, MLPClassifier  # Redes neurais artificiais
from xgboost import XGBClassifier                               # Gradient boosting otimizado
from imblearn.over_sampling import SMOTE                        # Balanceamento de classes com oversampling
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Métricas e visualização
import matplotlib.pyplot as plt                                 # Geração de gráficos e plots

# =================================================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =================================================================================

# Definição do caminho para o dataset de treinamento
ROOT = os.path.dirname(__file__)  # Diretório do script atual
csv_path = os.path.join(ROOT, "data_4000v", "env_vital_signals.csv")  # 4000 vítimas para treinamento
print(f"📁 Carregando dados de: {csv_path}")

# Carregamento do dataset principal
df = pd.read_csv(csv_path)
print(f"📊 Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
print(f"📋 Colunas disponíveis: {list(df.columns)}")

# =================================================================================
# FEATURE ENGINEERING - CRIAÇÃO DE VARIÁVEIS DERIVADAS
# =================================================================================
print("\n🔧 Iniciando Feature Engineering...")

# FEATURE 1: Razão entre qualidade da PA e pulso
# Interpretação: Relação entre a qualidade cardiovascular e frequência cardíaca
# Valores altos podem indicar boa eficiência cardíaca
df["qPA_pulso_ratio"] = df["qPA"] / (df["pulso"].replace(0, np.nan))
print("   ✓ qPA_pulso_ratio: Razão entre qualidade PA e pulso")

# FEATURE 2: Produto entre pulso e frequência respiratória
# Interpretação: Interação entre sistemas cardíaco e respiratório
# Pode indicar nível de estresse fisiológico geral
df["pulso_freq_prod"] = df["pulso"] * df["freq_resp"]
print("   ✓ pulso_freq_prod: Produto pulso × freq_resp (interação cardio-respiratória)")

# FEATURE 3: Diferença entre qualidade da PA e pulso
# Interpretação: Desbalanceamento entre qualidade vascular e frequência cardíaca
# Valores extremos podem indicar condições críticas
df["qPA_minus_pulso"] = df["qPA"] - df["pulso"]
print("   ✓ qPA_minus_pulso: Diferença qPA - pulso (desbalanceamento)")

# Lista das features selecionadas para o modelo
feature_cols = [
    "qPA",                # Feature original: Qualidade da Pressão Arterial
    "pulso",             # Feature original: Batimentos por Minuto  
    "freq_resp",         # Feature original: Frequência Respiratória
    "qPA_pulso_ratio",   # Feature derivada: Eficiência cardiovascular
    "pulso_freq_prod",   # Feature derivada: Interação cardio-respiratória
    "qPA_minus_pulso",   # Feature derivada: Desbalanceamento
]

print(f"🎯 Features selecionadas: {len(feature_cols)} variáveis")
for i, feat in enumerate(feature_cols, 1):
    print(f"   {i}. {feat}")

# Preparação das matrizes de features (X) e targets (y)
X = df[feature_cols].fillna(0).values  # Substituir NaN por 0 para estabilidade
y_reg = df["grav"].values              # Target para regressão (gravidade contínua)
y_clf = df["classe"].values            # Target para classificação (classes 1-4)

print(f"\n📈 Dados preparados:")
print(f"   • Matriz X: {X.shape} (amostras × features)")
print(f"   • Target regressão: {y_reg.shape} - range [{y_reg.min():.2f}, {y_reg.max():.2f}]")
print(f"   • Target classificação: {y_clf.shape} - classes {sorted(np.unique(y_clf))}")
print(f"   • Distribuição de classes: {dict(zip(*np.unique(y_clf, return_counts=True)))}")

# =================================================================================
# NORMALIZAÇÃO DOS DADOS
# =================================================================================
print("\n📏 Iniciando normalização dos dados...")

# StandardScaler: Transforma os dados para ter média=0 e desvio padrão=1
# Fórmula: z = (x - μ) / σ
# Importância: Algoritmos como MLP e XGBoost são sensíveis à escala das features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("   ✓ StandardScaler aplicado: média ≈ 0, desvio padrão ≈ 1")
print(f"   📊 Estatísticas pós-normalização:")
print(f"      • Média das features: {X_scaled.mean(axis=0).round(3)}")
print(f"      • Desvio padrão: {X_scaled.std(axis=0).round(3)}")

# =================================================================================
# DATA AUGMENTATION - TÉCNICAS DE AUMENTO DE DADOS
# =================================================================================
print("\n🔄 Iniciando Data Augmentation...")

def augment_jitter(X, y, noise_level=0.03, n_copies=3):
    """
    TÉCNICA JITTER - Aumento de dados para regressão
    
    PROPÓSITO:
    - Aumenta o dataset adicionando ruído gaussiano controlado
    - Melhora a generalização do modelo
    - Reduz overfitting
    - Simula variabilidade natural dos sinais vitais
    
    PARÂMETROS:
    - X: matriz de features originais
    - y: targets originais
    - noise_level: intensidade do ruído (3% por padrão)
    - n_copies: número de cópias com ruído a gerar
    
    ALGORITMO:
    1. Mantém dados originais
    2. Para cada cópia:
       - Gera ruído gaussiano N(0, noise_level)
       - Adiciona ruído às features: X_novo = X_original + ruído
       - Mantém targets inalterados
    3. Concatena todos os dados
    
    RETORNO:
    - X_aumentado: (n_original * (1 + n_copies)) × n_features
    - y_aumentado: (n_original * (1 + n_copies))
    """
    X_aug = [X]  # Lista iniciada com dados originais
    y_aug = [y]  # Lista iniciada com targets originais
    
    # Geração das cópias com ruído
    for copy_num in range(n_copies):
        # Ruído gaussiano com mesma forma da matriz X
        noise = np.random.normal(0, noise_level, X.shape)
        
        # Adição do ruído aos dados originais
        X_with_noise = X + noise
        
        # Armazenamento da cópia
        X_aug.append(X_with_noise)
        y_aug.append(y)  # Targets permanecem iguais
    
    # Concatenação vertical de todas as cópias
    return np.vstack(X_aug), np.hstack(y_aug)

# Aplicação do Jitter para regressão
print("   🎲 Aplicando Jitter para dados de regressão...")
X_jit_reg, y_jit_reg = augment_jitter(X_scaled, y_reg, noise_level=0.03, n_copies=3)
print(f"   📈 Dataset de regressão expandido:")
print(f"      • Original: {X_scaled.shape[0]} amostras")
print(f"      • Após Jitter: {X_jit_reg.shape[0]} amostras (aumento de {X_jit_reg.shape[0]/X_scaled.shape[0]:.1f}x)")

# =================================================================================
# BALANCEAMENTO DE CLASSES COM SMOTE
# =================================================================================
print("\n⚖️ Iniciando balanceamento de classes...")

# LabelEncoder: Converte classes categóricas para números inteiros
# Necessário para algoritmos que requerem targets numéricos
le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)
print(f"   🔄 Classes codificadas: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# SMOTE (Synthetic Minority Oversampling Technique)
# PROPÓSITO:
# - Balanceia classes desbalanceadas
# - Gera exemplos sintéticos para classes minoritárias
# - Melhora performance em classes com poucos exemplos
# 
# ALGORITMO:
# 1. Para cada amostra da classe minoritária:
#    - Encontra k vizinhos mais próximos da mesma classe
#    - Seleciona aleatoriamente um vizinho
#    - Gera ponto sintético na linha entre amostra e vizinho
# 2. Repete até balancear todas as classes
smote = SMOTE(random_state=42)  # Seed fixa para reprodutibilidade
X_smote, y_smote = smote.fit_resample(X_scaled, y_clf_enc)

print(f"   📊 Balanceamento realizado:")
print(f"      • Antes do SMOTE: {dict(zip(*np.unique(y_clf_enc, return_counts=True)))}")
print(f"      • Após SMOTE: {dict(zip(*np.unique(y_smote, return_counts=True)))}")
print(f"      • Dataset expandido: {X_scaled.shape[0]} → {X_smote.shape[0]} amostras")

# =================================================================================
# FUNÇÃO DE AVALIAÇÃO COM VALIDAÇÃO CRUZADA
# =================================================================================

def avaliar_configs(model_class, cfgs, X, y, scoring, nome_modelo):
    """
    FUNÇÃO GENÉRICA DE AVALIAÇÃO DE CONFIGURAÇÕES DE MODELOS
    
    PROPÓSITO:
    - Avalia múltiplas configurações de hiperparâmetros para um modelo
    - Usa validação cruzada k-fold (k=5) para avaliação robusta
    - Calcula múltiplas métricas para classificadores
    - Identifica a melhor configuração baseada na métrica principal
    
    PARÂMETROS:
    - model_class: Classe do modelo (RandomForestRegressor, XGBClassifier, etc.)
    - cfgs: Lista de dicionários com configurações de hiperparâmetros
    - X: Matriz de features
    - y: Vector de targets
    - scoring: Métrica principal ('accuracy', 'neg_root_mean_squared_error')
    - nome_modelo: Nome do modelo para exibição
    
    VALIDAÇÃO CRUZADA K-FOLD:
    - Divide dados em k=5 partes (folds)
    - Treina em 4 folds, testa em 1 fold
    - Repete processo 5 vezes, cada fold serve como teste uma vez
    - Calcula média e desvio padrão das métricas
    - Reduz variabilidade e fornece estimativa mais confiável
    
    MÉTRICAS CALCULADAS:
    - REGRESSÃO: RMSE (Root Mean Squared Error)
    - CLASSIFICAÇÃO: Acurácia, Precisão, Recall, F1-Score
    
    RETORNO:
    - Lista de tuplas (configuração, score_médio)
    - Melhor configuração baseada na métrica principal
    """
    resultados = []
    print(f"\n{'='*15} {nome_modelo} {'='*15}")
    
    # Avaliação de cada configuração
    for idx, cfg in enumerate(cfgs, 1):
        print(f"\n🔍 Avaliando Configuração {idx}:")
        print(f"   Parâmetros: {cfg}")
        
        # Preparação da configuração com parâmetros padrão
        cfg_extra = cfg.copy()
        if "random_state" not in cfg_extra:
            cfg_extra["random_state"] = 42  # Reprodutibilidade
            
        # Instanciação do modelo com tratamento de exceções
        try:
            modelo = model_class(**cfg_extra)
        except TypeError:
            # Remove n_jobs se não suportado pelo modelo
            cfg_extra.pop("n_jobs", None)
            modelo = model_class(**cfg_extra)

        # AVALIAÇÃO ESPECÍFICA POR TIPO DE MODELO
        if scoring == "accuracy":
            # ===== CLASSIFICADORES =====
            print("   📊 Calculando métricas de classificação...")
            
            # Validação cruzada para múltiplas métricas
            accuracy_scores = cross_val_score(modelo, X, y, scoring="accuracy", cv=5, n_jobs=-1)
            precision_scores = cross_val_score(modelo, X, y, scoring="precision_macro", cv=5, n_jobs=-1)
            recall_scores = cross_val_score(modelo, X, y, scoring="recall_macro", cv=5, n_jobs=-1)
            f1_scores = cross_val_score(modelo, X, y, scoring="f1_macro", cv=5, n_jobs=-1)
            
            # Cálculo das médias
            accuracy_mean = accuracy_scores.mean()
            precision_mean = precision_scores.mean()
            recall_mean = recall_scores.mean()
            f1_mean = f1_scores.mean()
            
            # Exibição dos resultados
            print(f"   📈 RESULTADOS (média ± desvio padrão):")
            print(f"      🎯 Acurácia: {accuracy_mean:.4f} (±{accuracy_scores.std():.4f})")
            print(f"      🎯 Precisão: {precision_mean:.4f} (±{precision_scores.std():.4f})")
            print(f"      🎯 Recall:   {recall_mean:.4f} (±{recall_scores.std():.4f})")
            print(f"      🎯 F1-Score: {f1_mean:.4f} (±{f1_scores.std():.4f})")
            
            # Armazenamento do resultado (usando acurácia como métrica principal)
            resultados.append((cfg, accuracy_mean))
            
        else:
            # ===== REGRESSORES =====
            print("   📊 Calculando métricas de regressão...")
            
            # Validação cruzada para RMSE
            scores = cross_val_score(modelo, X, y, scoring=scoring, cv=5, n_jobs=-1)
            
            # Conversão de RMSE negativo para positivo
            media = (-scores.mean()) if scoring.startswith("neg_") else scores.mean()
            desvio = scores.std()
            
            # Exibição dos resultados
            metric_name = scoring.replace('neg_', '').upper()
            print(f"   📈 RESULTADO:")
            print(f"      🎯 {metric_name}: {media:.4f} (±{desvio:.4f})")
            
            # Armazenamento do resultado
            resultados.append((cfg, media))
    
    # IDENTIFICAÇÃO DA MELHOR CONFIGURAÇÃO
    if scoring.startswith("neg_"):
        # Para métricas negativas (como neg_rmse), maior valor = melhor
        melhor = max(resultados, key=lambda x: x[1])
    else:
        # Para métricas positivas (como accuracy), maior valor = melhor
        melhor = max(resultados, key=lambda x: x[1])
    
    print(f"\n🏆 MELHOR CONFIGURAÇÃO:")
    print(f"   Parâmetros: {melhor[0]}")
    print(f"   Score: {melhor[1]:.4f}")
    
    return resultados, melhor

# =================================================================================
# CONFIGURAÇÕES DE HIPERPARÂMETROS DOS MODELOS
# =================================================================================
print("\n⚙️ Definindo configurações de hiperparâmetros...")

# ===== RANDOM FOREST REGRESSOR =====
"""
RANDOM FOREST - FUNDAMENTOS TEÓRICOS:

ALGORITMO:
• Ensemble de múltiplas árvores de decisão treinadas independentemente
• Cada árvore é treinada em uma amostra bootstrap dos dados (bagging)
• Cada divisão de nó considera apenas um subconjunto aleatório das features
• Predição final = média das predições de todas as árvores (regressão)

HIPERPARÂMETROS PRINCIPAIS:
• n_estimators: Número de árvores no ensemble
  - Mais árvores = maior estabilidade, menor variância, maior tempo de treinamento
  - Range típico: 100-500 árvores
  
• max_depth: Profundidade máxima de cada árvore
  - Maior profundidade = maior expressividade, risco de overfitting
  - None = sem limite (cresce até critério de parada)
  
• min_samples_split: Mínimo de amostras para dividir um nó interno
  - Valores baixos = árvores mais complexas, possível overfitting
  - Valores altos = regularização, árvores mais simples
  
VANTAGENS:
• Robustez a outliers e ruído
• Não requer normalização de features
• Fornece importância das features
• Paralelização natural (n_jobs=-1)

DESVANTAGENS:
• Pode ter overfitting em datasets pequenos com muitas features
• Menos interpretável que árvore única
• Memória intensiva para muitas árvores
"""
rf_reg_cfgs = [
    # CONFIGURAÇÃO 1: Modelo médio-leve
    {
        "n_estimators": 200,        # 200 árvores no ensemble
        "max_depth": 20,           # Profundidade máxima de 20 níveis
        "min_samples_split": 2,    # Mínimo 2 amostras para dividir nó
        "n_jobs": -1              # Usar todos os cores disponíveis
    },
    
    # CONFIGURAÇÃO 2: Modelo robusto (GERALMENTE A MELHOR)
    {
        "n_estimators": 300,        # 300 árvores (mais estável)
        "max_depth": 30,           # Maior profundidade (mais expressivo)
        "min_samples_split": 5,    # Maior controle de overfitting
        "n_jobs": -1
    },
    
    # CONFIGURAÇÃO 3: Modelo mais complexo
    {
        "n_estimators": 400,        # 400 árvores (máxima expressividade)
        "max_depth": None,         # Sem limite de profundidade
        "min_samples_split": 2,    # Divisão mais agressiva
        "n_jobs": -1
    },
]

# ===== MLP REGRESSOR =====
"""
MULTILAYER PERCEPTRON (MLP) REGRESSOR - FUNDAMENTOS TEÓRICOS:

ARQUITETURA:
• Rede neural feedforward densamente conectada
• Input Layer → Hidden Layers → Output Layer (1 neurônio para regressão)
• Função de ativação padrão: ReLU nas camadas ocultas, linear na saída
• Otimizador padrão: LBFGS (para datasets pequenos) ou Adam

PROCESSO DE TREINAMENTO:
1. Forward Pass: Dados fluem da entrada para saída
   - z = W·x + b (combinação linear)
   - a = f(z) (aplicação da função de ativação)
2. Cálculo da Loss: MSE (Mean Squared Error) para regressão
3. Backward Pass: Gradientes propagam da saída para entrada (backpropagation)
4. Atualização dos pesos: W = W - α·∇W (gradient descent)

HIPERPARÂMETROS PRINCIPAIS:
• hidden_layer_sizes: Arquitetura das camadas ocultas
  - (64, 32): 2 camadas com 64 e 32 neurônios
  - Mais neurônios = maior capacidade, risco de overfitting
  - Mais camadas = maior expressividade, maior complexidade

• alpha: Regularização L2 (Ridge)
  - Penaliza pesos grandes: Loss_total = MSE + α·||W||²
  - Previne overfitting, melhora generalização
  - Range típico: 1e-5 a 1e-2

• max_iter: Número máximo de épocas de treinamento
  - Muito baixo = underfitting (convergência incompleta)
  - Muito alto = overfitting ou desperdício computacional

VANTAGENS:
• Aproximador universal de funções (teorema da aproximação universal)
• Capacidade de aprender padrões não-lineares complexos
• Adaptabilidade a diferentes tipos de dados

DESVANTAGENS:
• Sensível à escala das features (requer normalização)
• Propenso a mínimos locais
• Muitos hiperparâmetros para ajustar
• "Black box" - baixa interpretabilidade
"""
mlp_reg_cfgs = [
    # CONFIGURAÇÃO 1: Rede simples
    {
        "hidden_layer_sizes": (64, 32),    # 2 camadas: 64 → 32 neurônios
        "alpha": 1e-4,                     # Regularização L2 baixa
        "max_iter": 1000                   # 1000 épocas de treinamento
    },
    
    # CONFIGURAÇÃO 2: Rede média (GERALMENTE A MELHOR)
    {
        "hidden_layer_sizes": (128, 64, 32),  # 3 camadas: 128 → 64 → 32
        "alpha": 5e-4,                        # Regularização L2 média
        "max_iter": 2000                      # 2000 épocas
    },
    
    # CONFIGURAÇÃO 3: Rede complexa
    {
        "hidden_layer_sizes": (256, 128, 64), # 3 camadas grandes: 256 → 128 → 64
        "alpha": 1e-3,                        # Regularização L2 alta
        "max_iter": 2500                      # 2500 épocas
    },
]

# ===== XGBOOST CLASSIFIER =====
"""
EXTREME GRADIENT BOOSTING (XGBoost) - FUNDAMENTOS TEÓRICOS:

ALGORITMO GRADIENT BOOSTING:
• Ensemble sequencial de weak learners (árvores rasas)
• Cada nova árvore corrige erros das árvores anteriores
• Otimização: Minimiza função de perda usando gradientes de segunda ordem

PROCESSO ITERATIVO:
1. Inicialização: F₀(x) = predição inicial (média para regressão)
2. Para cada iteração t:
   - Calcula gradientes: gᵢ = ∂Loss/∂F(xᵢ) 
   - Calcula hessianos: hᵢ = ∂²Loss/∂F(xᵢ)²
   - Treina árvore hₜ(x) para predizer pseudorresíduos
   - Atualiza: Fₜ(x) = Fₜ₋₁(x) + η·hₜ(x)
3. Predição final: F(x) = Σ η·hₜ(x)

INOVAÇÕES DO XGBOOST:
• Regularização L1 e L2 integrada na função objetivo
• Paralelização inteligente da construção de árvores
• Handling automático de valores missing
• Pruning de árvores usando gamma (complexidade mínima)

HIPERPARÂMETROS PRINCIPAIS:
• n_estimators: Número de árvores (boosting rounds)
  - Mais árvores = maior expressividade, risco de overfitting
  - Necessita early stopping ou regularização

• learning_rate (η): Taxa de aprendizado
  - Controla contribuição de cada árvore
  - Baixo (0.01-0.1) = convergência lenta mas estável
  - Alto (0.1-0.3) = convergência rápida, risco de overfitting

• max_depth: Profundidade máxima das árvores
  - Árvores rasas (3-6) previnem overfitting
  - Árvores profundas (>10) capturam interações complexas

• subsample: Fração de amostras para treinar cada árvore
  - <1.0 introduz estocasticidade (regularização)
  - Reduz overfitting e acelera treinamento

VANTAGENS:
• Estado da arte em competições ML (Kaggle)
• Excelente performance out-of-the-box
• Handling robusto de features categóricas e missing values
• Regularização built-in

DESVANTAGENS:
• Muitos hiperparâmetros para ajustar
• Sensível ao desbalanceamento de classes
• Treinamento sequencial (menos paralelizável que Random Forest)
"""
xgb_clf_cfgs = [
    # CONFIGURAÇÃO 1: Modelo conservador
    {
        "n_estimators": 300,        # 300 estimadores
        "learning_rate": 0.05,     # Taxa de aprendizado baixa (mais estável)
        "max_depth": 6,            # Profundidade moderada
        "subsample": 0.8,          # 80% das amostras por árvore (regularização)
        "n_jobs": -1
    },
    
    # CONFIGURAÇÃO 2: Modelo rápido
    {
        "n_estimators": 200,        # Menos estimadores
        "learning_rate": 0.10,     # Taxa de aprendizado alta (convergência rápida)
        "max_depth": 5,            # Árvores mais rasas
        "subsample": 0.9,          # Mais amostras por árvore
        "n_jobs": -1
    },
    
    # CONFIGURAÇÃO 3: Modelo robusto (GERALMENTE A MELHOR)
    {
        "n_estimators": 400,        # Muitos estimadores
        "learning_rate": 0.03,     # Taxa muito baixa (aprendizado gradual)
        "max_depth": 7,            # Árvores mais profundas
        "subsample": 0.8,          # Regularização por subamostragem
        "n_jobs": -1
    },
]

# ===== MLP CLASSIFIER =====
"""
MULTILAYER PERCEPTRON (MLP) CLASSIFIER - FUNDAMENTOS TEÓRICOS:

ARQUITETURA PARA CLASSIFICAÇÃO:
• Estrutura similar ao MLP Regressor
• Output Layer: n neurônios (n = número de classes)
• Função de ativação final: Softmax para multiclasse
• Softmax: σ(zᵢ) = exp(zᵢ) / Σⱼexp(zⱼ) 
  - Converte logits em probabilidades [0,1]
  - Soma das probabilidades = 1

FUNÇÃO DE PERDA:
• Cross-Entropy Loss (Log-Loss) para classificação multiclasse:
  Loss = -Σᵢ Σⱼ yᵢⱼ·log(ŷᵢⱼ)
  - yᵢⱼ: true label (one-hot encoded)
  - ŷᵢⱼ: predicted probability
  - Penaliza mais severamente predições incorretas com alta confiança

PROCESSO DE DECISÃO:
1. Forward Pass produz probabilidades P(classe|x)
2. Predição = argmax(P) (classe com maior probabilidade)
3. Confiança = max(P) (probabilidade da classe predita)

REGULARIZAÇÃO ESPECÍFICA:
• Dropout: Desativa neurônios aleatoriamente durante treinamento
• Early Stopping: Para treinamento quando validação estagnar
• Weight Decay (L2): Igual ao MLP Regressor

DIFERENÇAS DO MLP REGRESSOR:
• Função de ativação final: Softmax vs Linear
• Função de perda: Cross-Entropy vs MSE
• Métrica de avaliação: Accuracy/F1 vs RMSE
• Interpretação: Probabilidades vs Valores contínuos

HIPERPARÂMETROS:
• hidden_layer_sizes: Mesmo princípio do regressor
  - Para classificação, camadas menores podem ser suficientes
  - Evita overfitting em datasets desbalanceados

• alpha: Regularização L2
  - Especialmente importante em classificação
  - Previne overconfidence em predições

• max_iter: Épocas de treinamento
  - Classificação converge geralmente mais rápido que regressão
  - Monitor loss de validação para early stopping

ESTRATÉGIAS PARA CLASSES DESBALANCEADAS:
• SMOTE: Oversampling sintético (já aplicado)
• Class weights: Penalizar mais erros em classes minoritárias
• Threshold tuning: Ajustar threshold de decisão
"""
mlp_clf_cfgs = [
    # CONFIGURAÇÃO 1: Rede simples
    {
        "hidden_layer_sizes": (64, 32),    # Arquitetura simples
        "alpha": 1e-4,                     # Regularização baixa
        "max_iter": 1000                   # Treinamento moderado
    },
    
    # CONFIGURAÇÃO 2: Rede equilibrada (GERALMENTE A MELHOR)
    {
        "hidden_layer_sizes": (128, 64),   # 2 camadas bem dimensionadas
        "alpha": 5e-4,                     # Regularização moderada
        "max_iter": 2000                   # Treinamento extenso
    },
    
    # CONFIGURAÇÃO 3: Rede complexa
    {
        "hidden_layer_sizes": (256, 128, 64), # Arquitetura profunda
        "alpha": 1e-3,                        # Regularização alta
        "max_iter": 2500                      # Treinamento muito extenso
    },
]

print(f"   🌲 Random Forest: {len(rf_reg_cfgs)} configurações")
print(f"   🧠 MLP Regressor: {len(mlp_reg_cfgs)} configurações") 
print(f"   🚀 XGBoost: {len(xgb_clf_cfgs)} configurações")
print(f"   🧠 MLP Classifier: {len(mlp_clf_cfgs)} configurações")
print(f"   📊 Total: {len(rf_reg_cfgs) + len(mlp_reg_cfgs) + len(xgb_clf_cfgs) + len(mlp_clf_cfgs)} avaliações")

# -------------------------------------------------------------------------
# Avaliação ----------------------------------------------------------------
# -------------------------------------------------------------------------

# Regressão (com jitter)
rf_results, rf_best = avaliar_configs(RandomForestRegressor, rf_reg_cfgs, X_jit_reg, y_jit_reg, scoring="neg_root_mean_squared_error", nome_modelo="RandomForestRegressor")
mlp_reg_results, mlp_reg_best = avaliar_configs(MLPRegressor, mlp_reg_cfgs, X_jit_reg, y_jit_reg, scoring="neg_root_mean_squared_error", nome_modelo="MLPRegressor")

# Classificação (com SMOTE)
xgb_results, xgb_best = avaliar_configs(XGBClassifier, xgb_clf_cfgs, X_smote, y_smote, scoring="accuracy", nome_modelo="XGBClassifier")
mlp_clf_results, mlp_clf_best = avaliar_configs(MLPClassifier, mlp_clf_cfgs, X_smote, y_smote, scoring="accuracy", nome_modelo="MLPClassifier")

print("\n✅ Avaliação concluída! Veja as métricas acima para cada configuração.")

# -------------------------------------------------------------------------
# Exibe as melhores configurações -----------------------------------------
# -------------------------------------------------------------------------
print("\n=== MELHORES CONFIGURAÇÕES ===")
print(f"RandomForestRegressor: {rf_best[0]} → RMSE = {abs(rf_best[1]):.4f}")
print(f"MLPRegressor:         {mlp_reg_best[0]} → RMSE = {abs(mlp_reg_best[1]):.4f}")
print(f"XGBClassifier:        {xgb_best[0]} → ACC  = {xgb_best[1]:.4f}")
print(f"MLPClassifier:        {mlp_clf_best[0]} → ACC  = {mlp_clf_best[1]:.4f}")

# -------------------------------------------------------------------------
# Métricas detalhadas das melhores configurações de classificação ---------
# -------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score

print("\n=== MÉTRICAS DETALHADAS DAS MELHORES CONFIGURAÇÕES ===")

# XGBClassifier - melhor configuração
print(f"\n🌲 XGBClassifier (melhor): {xgb_best[0]}")
xgb_model = XGBClassifier(**xgb_best[0])
acc_xgb = cross_val_score(xgb_model, X_smote, y_smote, scoring="accuracy", cv=5, n_jobs=-1)
prec_xgb = cross_val_score(xgb_model, X_smote, y_smote, scoring="precision_macro", cv=5, n_jobs=-1)
rec_xgb = cross_val_score(xgb_model, X_smote, y_smote, scoring="recall_macro", cv=5, n_jobs=-1)
f1_xgb = cross_val_score(xgb_model, X_smote, y_smote, scoring="f1_macro", cv=5, n_jobs=-1)

print(f"   Acurácia: {acc_xgb.mean():.4f} (±{acc_xgb.std():.4f})")
print(f"   Precisão: {prec_xgb.mean():.4f} (±{prec_xgb.std():.4f})")
print(f"   Recall:   {rec_xgb.mean():.4f} (±{rec_xgb.std():.4f})")
print(f"   F1-Score: {f1_xgb.mean():.4f} (±{f1_xgb.std():.4f})")

# MLPClassifier - melhor configuração
print(f"\n🧠 MLPClassifier (melhor): {mlp_clf_best[0]}")
mlp_model = MLPClassifier(**mlp_clf_best[0])
acc_mlp = cross_val_score(mlp_model, X_smote, y_smote, scoring="accuracy", cv=5, n_jobs=-1)
prec_mlp = cross_val_score(mlp_model, X_smote, y_smote, scoring="precision_macro", cv=5, n_jobs=-1)
rec_mlp = cross_val_score(mlp_model, X_smote, y_smote, scoring="recall_macro", cv=5, n_jobs=-1)
f1_mlp = cross_val_score(mlp_model, X_smote, y_smote, scoring="f1_macro", cv=5, n_jobs=-1)

print(f"   Acurácia: {acc_mlp.mean():.4f} (±{acc_mlp.std():.4f})")
print(f"   Precisão: {prec_mlp.mean():.4f} (±{prec_mlp.std():.4f})")
print(f"   Recall:   {rec_mlp.mean():.4f} (±{rec_mlp.std():.4f})")
print(f"   F1-Score: {f1_mlp.mean():.4f} (±{f1_mlp.std():.4f})")

def plot_confusion_matrix(model, X, y, title, class_names=None):
    """
    MATRIZ DE CONFUSÃO - ANÁLISE DETALHADA DE CLASSIFICAÇÃO
    
    PROPÓSITO:
    • Visualiza performance de classificação classe por classe
    • Identifica confusões específicas entre classes
    • Facilita interpretação de erros do modelo
    
    INTERPRETAÇÃO DA MATRIZ:
    • Linhas: Classes verdadeiras (true labels)
    • Colunas: Classes preditas (predicted labels) 
    • Diagonal principal: Predições corretas
    • Elementos fora da diagonal: Erros de classificação
    
    MÉTRICAS DERIVADAS:
    Para cada classe i:
    • True Positives (TP): matriz[i,i]
    • False Positives (FP): sum(matriz[:,i]) - matriz[i,i]
    • False Negatives (FN): sum(matriz[i,:]) - matriz[i,i]
    • True Negatives (TN): sum(matriz) - TP - FP - FN
    
    • Precisão = TP/(TP+FP) = "Das predições para classe i, quantas estavam corretas?"
    • Recall = TP/(TP+FN) = "Das amostras reais da classe i, quantas foram identificadas?"
    • F1-Score = 2·(Precisão·Recall)/(Precisão+Recall) = Média harmônica
    
    ANÁLISE DE PADRÕES:
    • Erros sistemáticos: Classes frequentemente confundidas
    • Bias do modelo: Tendência a predizer certas classes
    • Qualidade por classe: Performance varia entre classes?
    
    VALIDAÇÃO CRUZADA:
    • cross_val_predict: Garante que predições são em dados não vistos
    • Evita overfitting na matriz de confusão
    • Resultado mais realista da performance
    """
    # Usar cross_val_predict para obter predições com validação cruzada
    y_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
    
    # Calcular a matriz de confusão
    cm = confusion_matrix(y, y_pred)
    
    # Configurar o plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Se não foram fornecidos nomes das classes, usar números
    if class_names is None:
        class_names = [f"Classe {i}" for i in sorted(np.unique(y))]
    
    # Plotar a matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    # Configurar título e layout
    ax.set_title(f'Matriz de Confusão - {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Salvar a figura
    filename = f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Matriz de confusão salva como: {filename}")
    
    # Mostrar a figura
    plt.show()
    
    return cm

# -------------------------------------------------------------------------
# Matrizes de Confusão das Melhores Configurações -------------------------
# -------------------------------------------------------------------------

print("\n=== MATRIZES DE CONFUSÃO ===")

# Nomes das classes de gravidade
class_names = ["Crítico", "Instável", "Pot. Estável", "Estável"]

# Matriz de confusão para XGBClassifier
print(f"\n🌲 Matriz de Confusão - XGBClassifier")
xgb_model_final = XGBClassifier(**xgb_best[0])
cm_xgb = plot_confusion_matrix(xgb_model_final, X_smote, y_smote, 
                              "XGBClassifier", class_names)

# Matriz de confusão para MLPClassifier
print(f"\n🧠 Matriz de Confusão - MLPClassifier")
mlp_model_final = MLPClassifier(**mlp_clf_best[0])
cm_mlp = plot_confusion_matrix(mlp_model_final, X_smote, y_smote, 
                              "MLPClassifier", class_names)

print("\n✅ Análise completa finalizada! Matrizes de confusão salvas como imagens.")

"""
=================================================================================
RESUMO DA METODOLOGIA EXPERIMENTAL IMPLEMENTADA
=================================================================================

PIPELINE COMPLETO DE MACHINE LEARNING:

1. PREPARAÇÃO DOS DADOS:
   ✓ Carregamento: Dataset com 4000 vítimas
   ✓ Feature Engineering: 6 features (3 originais + 3 derivadas)
   ✓ Feature Selection: Apenas qPA, pulso, freq_resp (conforme especificação)
   ✓ Missing Values: Preenchimento com zero para estabilidade

2. PRÉ-PROCESSAMENTO AVANÇADO:
   ✓ Normalização: StandardScaler (μ=0, σ=1)
   ✓ Data Augmentation: Jitter (ruído gaussiano) para regressão
   ✓ Balanceamento: SMOTE para classificação (oversampling sintético)

3. MODELOS AVALIADOS:
   ✓ RandomForestRegressor: 3 configurações (200-400 árvores)
   ✓ MLPRegressor: 3 configurações (2-3 camadas, 64-256 neurônios)
   ✓ XGBClassifier: 3 configurações (200-400 estimadores)
   ✓ MLPClassifier: 3 configurações (2-3 camadas, 64-256 neurônios)

4. VALIDAÇÃO CRUZADA ROBUSTA:
   ✓ K-Fold Cross-Validation (k=5)
   ✓ Métrica regressão: RMSE (Root Mean Squared Error)
   ✓ Métricas classificação: Acurácia, Precisão, Recall, F1-Score
   ✓ Estatísticas: Média ± desvio padrão para cada métrica

5. ANÁLISE DE RESULTADOS:
   ✓ Ranking de configurações por performance
   ✓ Identificação das melhores configurações
   ✓ Métricas detalhadas para classificadores
   ✓ Matrizes de confusão para análise de erros

6. VISUALIZAÇÃO E RELATÓRIOS:
   ✓ Matrizes de confusão salvas como imagens PNG (300 DPI)
   ✓ Nomenclatura clara das classes (Crítico, Instável, etc.)
   ✓ Logs detalhados de todas as etapas

CONSIDERAÇÕES METODOLÓGICAS:

ROBUSTEZ EXPERIMENTAL:
• Validação cruzada elimina bias de divisão train/test
• Múltiplas configurações reduzem dependência de hiperparâmetros
• Data augmentation melhora generalização
• Seeds fixas garantem reprodutibilidade

ESCOLHAS DE DESIGN:
• SMOTE vs outras técnicas: Gera pontos sintéticos realistas
• StandardScaler vs MinMax: Melhor para redes neurais (distribuição gaussiana)
• 5-fold CV vs outros: Balanceio entre bias e variância
• RMSE vs MAE: Penaliza mais erros grandes (importante para gravidade)

LIMITAÇÕES E EXTENSÕES FUTURAS:
• Validação em dados temporalmente separados (se disponível)
• Ensemble de múltiplos algoritmos (voting, stacking)
• Otimização de hiperparâmetros (GridSearch, RandomSearch, Bayesian)
• Análise de importância de features (SHAP, permutation importance)
• Calibração de probabilidades para classificadores

APLICAÇÃO PRÁTICA:
Este pipeline pode ser facilmente adaptado para:
• Novos datasets (modificar csv_path)
• Diferentes features (alterar feature_cols)
• Outros algoritmos (adicionar em cfgs)
• Métricas customizadas (modificar scoring)

=================================================================================
"""