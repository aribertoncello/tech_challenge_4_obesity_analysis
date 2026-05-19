import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ObesityAI — Dashboard Analítico",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"]   = "none"

# ─────────────────────────────────────────
# Carregamento e preparação dos dados
# (replica exatamente o feature engineering do notebook)
# ─────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_csv("Obesity.csv")

    # Arredondamentos conforme notebook
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        df[col] = df[col].round().astype(int)

    # [FE-1] Faixa etária
    bins   = [0, 18, 25, 35, 50, 100]
    labels = ["Adolescente", "Jovem Adulto", "Adulto", "Meia-idade", "Idoso"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False).astype(str)

    # [FE-2] Score de hábitos saudáveis
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map  = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df["CAEC_num"] = df["CAEC"].map(caec_map)
    df["CALC_num"]  = df["CALC"].map(calc_map)
    positive = (df["CH2O"] / 3) + (df["FCVC"] / 3) + (df["FAF"] / 3)
    negative = (df["CAEC_num"] / 3) + (df["CALC_num"] / 3) + df["FAVC"].map({"yes": 1, "no": 0})
    df["healthy_score"] = ((positive - negative / 2) * 10 / 3).clip(0, 10).round(2)

    # [FE-3] Sedentarismo
    df["sedentary"] = ((df["FAF"] == 0) & (df["TUE"] >= 1)).astype(int)

    # [FE-4] Histórico familiar × FAVC
    df["fam_x_favc"] = (
        (df["family_history"] == "yes").astype(int) *
        (df["FAVC"] == "yes").astype(int)
    )

    # [FE-5] Razão refeições / atividade
    df["meal_activity_ratio"] = (df["NCP"] / (df["FAF"] + 1)).round(3)

    # [FE-6] Alto consumo de álcool
    df["high_alcohol"] = df["CALC_num"].apply(lambda x: 1 if x >= 2 else 0)

    # Ordem e tradução das classes
    ORDEM = [
        "Insufficient_Weight", "Normal_Weight",
        "Overweight_Level_I", "Overweight_Level_II",
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    ]
    TRADUCAO = {
        "Insufficient_Weight" : "Peso Insuficiente",
        "Normal_Weight"        : "Peso Normal",
        "Overweight_Level_I"   : "Sobrepeso I",
        "Overweight_Level_II"  : "Sobrepeso II",
        "Obesity_Type_I"       : "Obesidade I",
        "Obesity_Type_II"      : "Obesidade II",
        "Obesity_Type_III"     : "Obesidade III",
    }
    df["Obesity_PT"] = df["Obesity"].map(TRADUCAO)
    df["Obesity"] = pd.Categorical(df["Obesity"], categories=ORDEM, ordered=True)
    df["Obesity_PT"] = pd.Categorical(
        df["Obesity_PT"],
        categories=[TRADUCAO[c] for c in ORDEM],
        ordered=True
    )

    # Flag sintético (SMOTE)
    df["sintetico"] = (df["Age"] % 1 != 0)

    return df, ORDEM, TRADUCAO

df, ORDEM, TRADUCAO = carregar_dados()

CORES   = ["#43A047","#66BB6A","#FDD835","#FB8C00","#E53935","#B71C1C","#4A148C"]
PALETTE = dict(zip([TRADUCAO[c] for c in ORDEM], CORES))
ORDEM_PT = [TRADUCAO[c] for c in ORDEM]

# ─────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────
st.title("📊 ObesityAI — Dashboard Analítico")
st.markdown(
    f"Painel de insights para a **equipe médica**, baseado na análise de "
    f"**{len(df):,} pacientes** — hábitos comportamentais e estilo de vida."
)
st.divider()

# ─────────────────────────────────────────
# Filtros laterais
# ─────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Filtros")

    genero_sel = st.selectbox("Gênero",
        ["Todos", "Male", "Female"],
        format_func=lambda x: {"Todos":"Todos","Male":"Masculino","Female":"Feminino"}[x]
    )
    idade_min, idade_max = int(df["Age"].min()), int(df["Age"].max())
    idade_sel = st.slider("Faixa etária", idade_min, idade_max, (idade_min, idade_max))

    classes_sel = st.multiselect(
        "Níveis de Obesidade", options=ORDEM_PT, default=ORDEM_PT
    )

    mostrar_sintetico = st.radio(
        "Origem dos dados",
        ["Todos", "Apenas originais", "Apenas sintéticos (SMOTE)"],
        index=0
    )
    st.caption("65% dos dados são sintéticos (gerados via SMOTE para balancear as classes).")

# Aplicando filtros
dff = df.copy()
if genero_sel != "Todos":
    dff = dff[dff["Gender"] == genero_sel]
dff = dff[(dff["Age"] >= idade_sel[0]) & (dff["Age"] <= idade_sel[1])]
dff = dff[dff["Obesity_PT"].isin(classes_sel)]
if mostrar_sintetico == "Apenas originais":
    dff = dff[~dff["sintetico"]]
elif mostrar_sintetico == "Apenas sintéticos (SMOTE)":
    dff = dff[dff["sintetico"]]

if dff.empty:
    st.warning("Nenhum dado com os filtros selecionados. Ajuste os filtros.")
    st.stop()

# ─────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────
st.subheader("📌 Visão Geral")
k1, k2, k3, k4, k5, k6 = st.columns(6)

pct_obesos   = dff["Obesity"].isin(["Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]).mean() * 100
pct_normal   = (dff["Obesity"] == "Normal_Weight").mean() * 100
pct_sed      = dff["sedentary"].mean() * 100
hs_medio     = dff["healthy_score"].mean()
pct_hist_fam = (dff["family_history"] == "yes").mean() * 100
idade_media  = dff["Age"].mean()

k1.metric("👥 Pacientes",        f"{len(dff):,}")
k2.metric("⚠️ Com Obesidade",    f"{pct_obesos:.1f}%")
k3.metric("✅ Peso Normal",      f"{pct_normal:.1f}%")
k4.metric("🛋️ Sedentários",     f"{pct_sed:.1f}%")
k5.metric("💚 Score Hábitos",    f"{hs_medio:.1f}/10")
k6.metric("🧬 Hist. Familiar",   f"{pct_hist_fam:.1f}%")

st.divider()

# ─────────────────────────────────────────
# SEÇÃO 1 — Distribuição
# ─────────────────────────────────────────
st.subheader("📈 Distribuição dos Pacientes")
c1, c2 = st.columns([3, 2])

with c1:
    contagem = dff["Obesity_PT"].value_counts().reindex(ORDEM_PT).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(contagem.index, contagem.values,
                  color=[PALETTE[c] for c in contagem.index], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax.set_title("Pacientes por Nível de Obesidade", fontsize=11)
    ax.set_ylabel("Pacientes")
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with c2:
    gen_ob = dff.groupby(["Obesity_PT","Gender"]).size().unstack(fill_value=0)
    gen_ob.columns = ["Feminino" if c=="Female" else "Masculino" for c in gen_ob.columns]
    gen_ob = gen_ob.reindex(ORDEM_PT).dropna()
    fig, ax = plt.subplots(figsize=(5, 4))
    gen_ob.plot(kind="barh", ax=ax, color=["#e91e63","#1565c0"], edgecolor="white")
    ax.set_title("Distribuição por Gênero", fontsize=11)
    ax.set_xlabel("Pacientes"); ax.set_ylabel("")
    ax.legend(title="Gênero", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.divider()

# ─────────────────────────────────────────
# SEÇÃO 2 — Features de engenharia
# ─────────────────────────────────────────
st.subheader("🔬 Features Criadas no Modelo")
c3, c4, c5 = st.columns(3)

with c3:
    hs = dff.groupby("Obesity_PT")["healthy_score"].mean().reindex(ORDEM_PT).dropna()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(hs.index, hs.values, color=[PALETTE[c] for c in hs.index], alpha=0.9)
    ax.set_title("Score Médio de Hábitos Saudáveis\n(0 = pior | 10 = melhor)", fontsize=10)
    ax.set_xlabel("Score médio"); ax.set_xlim(0, 10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with c4:
    sed = dff.groupby("Obesity_PT")["sedentary"].mean().reindex(ORDEM_PT).dropna() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(sed.index, sed.values, color=[PALETTE[c] for c in sed.index], alpha=0.9)
    ax.set_title("% de Pacientes Sedentários\n(FAF=0 e TUE≥1)", fontsize=10)
    ax.set_xlabel("%"); ax.set_xlim(0, 100)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with c5:
    mar = dff.groupby("Obesity_PT")["meal_activity_ratio"].mean().reindex(ORDEM_PT).dropna()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(mar.index, mar.values, color=[PALETTE[c] for c in mar.index], alpha=0.9)
    ax.set_title("Razão Refeições / Atividade Física\n(quanto maior, mais desequilibrado)", fontsize=10)
    ax.set_xlabel("Razão média")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.divider()

# ─────────────────────────────────────────
# SEÇÃO 3 — Hábitos alimentares
# ─────────────────────────────────────────
st.subheader("🥗 Hábitos Alimentares")
c6, c7, c8 = st.columns(3)

with c6:
    fam = dff.groupby("Obesity_PT")["fam_x_favc"].mean().reindex(ORDEM_PT).dropna() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(fam.index, fam.values, color=[PALETTE[c] for c in fam.index], alpha=0.9)
    ax.set_title("% Histórico Familiar\n× Alimentos Calóricos", fontsize=10)
    ax.set_xlabel("%"); ax.set_xlim(0, 100)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with c7:
    veg = dff.groupby("Obesity_PT")["FCVC"].mean().reindex(ORDEM_PT).dropna()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(veg.index, veg.values, color=[PALETTE[c] for c in veg.index], alpha=0.9)
    ax.set_title("Consumo Médio de Vegetais\n(0–3)", fontsize=10)
    ax.set_xlabel("Frequência média"); ax.set_xlim(0, 3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with c8:
    alc = dff.groupby("Obesity_PT")["high_alcohol"].mean().reindex(ORDEM_PT).dropna() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(alc.index, alc.values, color=[PALETTE[c] for c in alc.index], alpha=0.9)
    ax.set_title("% Alto Consumo de Álcool\n(Frequentemente ou Sempre)", fontsize=10)
    ax.set_xlabel("%"); ax.set_xlim(0, 100)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.divider()

# ─────────────────────────────────────────
# SEÇÃO 4 — Faixa etária e transporte
# ─────────────────────────────────────────
st.subheader("🚗 Faixa Etária e Transporte")
c9, c10 = st.columns(2)

with c9:
    ordem_age = ["Adolescente","Jovem Adulto","Adulto","Meia-idade","Idoso"]
    ag = (
        dff.groupby(["age_group","Obesity_PT"]).size()
        .unstack(fill_value=0)
        .reindex(ordem_age)
        .dropna()
    )
    ag_pct = ag.div(ag.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ag_pct.plot(kind="bar", ax=ax,
                color=[PALETTE.get(c,"#999") for c in ag_pct.columns],
                edgecolor="white", stacked=True)
    ax.set_title("Distribuição de Obesidade por Faixa Etária", fontsize=11)
    ax.set_xlabel("Faixa Etária"); ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Nível", fontsize=7, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with c10:
    TRANSP_PT = {
        "Public_Transportation": "Transp. Público", "Walking": "Caminhando",
        "Automobile": "Carro", "Motorbike": "Moto", "Bike": "Bicicleta"
    }
    transp = dff["MTRANS"].value_counts()
    transp.index = [TRANSP_PT.get(i, i) for i in transp.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(transp.values, labels=transp.index, autopct="%1.1f%%",
           colors=sns.color_palette("Set2", len(transp)), startangle=140)
    ax.set_title("Meio de Transporte Habitual", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.divider()

# ─────────────────────────────────────────
# SEÇÃO 5 — Nota sobre o dataset
# ─────────────────────────────────────────
st.subheader("🔍 Composição do Dataset")
c11, c12 = st.columns(2)

with c11:
    n_orig = (~df["sintetico"]).sum()
    n_sint = df["sintetico"].sum()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie([n_orig, n_sint],
           labels=[f"Originais ({n_orig})", f"Sintéticos — SMOTE ({n_sint})"],
           colors=["#1565c0","#e53935"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Composição: Dados Originais vs Sintéticos", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with c12:
    st.info(
        "**Por que existem dados sintéticos?**\n\n"
        "O dataset original (UCI) tinha apenas ~498 registros reais e classes desbalanceadas. "
        "Os autores aplicaram a técnica **SMOTE** para gerar ~1.600 amostras adicionais, "
        "totalizando 2.111 registros.\n\n"
        "Isso explica as acurácias altas do modelo (~83–85% sem leakage). "
        "Os dados sintéticos seguem padrões matemáticos previsíveis — por isso o modelo aprende bem. "
        "Na sidebar do filtro você pode explorar apenas os dados originais."
    )

st.divider()

# ─────────────────────────────────────────
# SEÇÃO 6 — Insights
# ─────────────────────────────────────────
st.subheader("💡 Principais Insights para a Equipe Médica")
col_i1, col_i2 = st.columns(2)

with col_i1:
    st.success(
        "**💚 Score de Hábitos Saudáveis**\n\n"
        "Pacientes com peso normal apresentam score médio significativamente maior. "
        "O score combina hidratação, consumo de vegetais e atividade física — "
        "todos fatores modificáveis e de fácil intervenção clínica."
    )
    st.success(
        "**🥦 Consumo de Vegetais**\n\n"
        "Há uma tendência clara: quanto mais grave o nível de obesidade, "
        "menor o consumo de vegetais. Orientação nutricional deve ser prioridade."
    )
    st.info(
        "**👨‍👩‍👧 Histórico Familiar × Alimentação**\n\n"
        "A combinação de histórico familiar positivo com consumo frequente de "
        "alimentos calóricos (`fam_x_favc`) é especialmente prevalente nos níveis "
        "mais graves. Pacientes com esse perfil merecem acompanhamento preventivo redobrado."
    )

with col_i2:
    st.warning(
        "**🛋️ Sedentarismo**\n\n"
        "Pacientes com obesidade tipo II e III têm a maior proporção de perfil sedentário "
        "(sem atividade física e alto tempo em telas). "
        "Intervenções de incentivo ao movimento são altamente recomendadas."
    )
    st.warning(
        "**🍽️ Razão Refeições / Atividade**\n\n"
        "O desequilíbrio entre quantidade de refeições e nível de atividade física "
        "cresce progressivamente com a gravidade da obesidade. "
        "Reduzir refeições ou aumentar atividade pode ter impacto direto."
    )
    st.error(
        "**🍺 Álcool**\n\n"
        "O consumo frequente ou constante de álcool está associado aos níveis "
        "mais graves de obesidade. Triagem de consumo alcoólico deve integrar "
        "a avaliação de risco metabólico."
    )

st.divider()

# Tabela de dados filtrados
with st.expander("📄 Ver tabela de dados filtrados"):
    cols_exibir = ["Gender","Age","age_group","family_history","FAVC","FCVC",
                   "FAF","CH2O","TUE","healthy_score","sedentary","Obesity_PT"]
    rename = {
        "Gender":"Gênero","Age":"Idade","age_group":"Faixa Etária",
        "family_history":"Hist. Familiar","FAVC":"Alim. Calóricos",
        "FCVC":"Vegetais","FAF":"Ativ. Física","CH2O":"Água",
        "TUE":"Tempo Telas","healthy_score":"Score Hábitos",
        "sedentary":"Sedentário","Obesity_PT":"Nível de Obesidade"
    }
    st.dataframe(
        dff[cols_exibir].rename(columns=rename),
        use_container_width=True, hide_index=True
    )

st.caption(
    "📊 Dashboard analítico — POSTECH Tech Challenge Fase 04 — Data Analytics"
)
