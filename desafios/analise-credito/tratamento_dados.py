from pathlib import Path
from typing import List, Tuple

import pandas as pd

SOURCE = Path(__file__).with_name("dataset.csv")
OUTPUT = Path(__file__).with_name("dataset_tratado.csv")

# --- Constantes de domínio configuráveis ---
IDADE_MIN = 18
IDADE_MAX = 75
IDADE_CORRUPCAO_THRESHOLD = 100  # idades > 100 são consideradas corrompidas (x10)
TEMPO_EMPREGO_MIN = 0
TEMPO_EMPREGO_MAX = 30
COLUNAS_CATEGORICAS_TRIM = ["estado", "escolaridade", "finalidade"]
COLUNAS_ESPERADAS = [
    "idade", "estado", "escolaridade", "renda_mensal",
    "tempo_emprego_anos", "score_credito", "valor_solicitado",
    "prazo_meses", "finalidade", "dividas_ativas", "inadimplente",
]


def _validar_schema(df: pd.DataFrame) -> None:
    """Valida schema de entrada: colunas, tipos básicos, ranges esperados."""
    # Colunas
    colunas_faltando = set(COLUNAS_ESPERADAS) - set(df.columns)
    colunas_extra = set(df.columns) - set(COLUNAS_ESPERADAS)
    if colunas_faltando:
        raise ValueError(f"Colunas faltando: {colunas_faltando}")
    if colunas_extra:
        raise ValueError(f"Colunas inesperadas: {colunas_extra}")

    # Tipos numéricos básicos
    for col in ["idade", "tempo_emprego_anos", "score_credito", "prazo_meses", "dividas_ativas", "inadimplente"]:
        if not pd.api.types.is_integer_dtype(df[col]):
            raise TypeError(f"Coluna '{col}' deve ser inteira, encontrado: {df[col].dtype}")
    for col in ["renda_mensal", "valor_solicitado"]:
        if not pd.api.types.is_float_dtype(df[col]):
            raise TypeError(f"Coluna '{col}' deve ser float, encontrado: {df[col].dtype}")

    # Target binário
    if set(df["inadimplente"].unique()) != {0, 1}:
        raise ValueError("Target 'inadimplente' deve conter apenas 0 e 1")


def _log_duplicatas(df: pd.DataFrame) -> List[int]:
    """Identifica e loga índices de duplicatas exatas antes de remover."""
    duplicatas_mask = df.duplicated(keep="first")
    indices_dup = df.index[duplicatas_mask].tolist()
    if indices_dup:
        print(f"  [LOG] Duplicatas exatas encontradas nos índices: {indices_dup}")
    return indices_dup


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica correções determinísticas e marca valores inválidos como ausentes.

    Passos:
    1. Valida schema de entrada
    2. Normaliza categorias textuais (strip)
    3. Corrige idades corrompidas (x10) se resultarem em idade plausível
    4. Marca tempos de emprego fora do domínio como ausentes (imputação no pipeline)
    5. Remove duplicatas exatas (loga índices)
    6. Cria variável derivada: razao_emprestimo_renda
    """
    _validar_schema(df)
    tratado = df.copy()

    # 1. Normaliza categorias textuais que chegaram com espaços extras.
    for coluna in COLUNAS_CATEGORICAS_TRIM:
        tratado[coluna] = tratado[coluna].astype("string").str.strip()

    # 2. Corrige idades: valores > 100 são exatamente 10x a idade real.
    idade_corrompida = tratado["idade"] > IDADE_CORRUPCAO_THRESHOLD
    n_corrompidas = int(idade_corrompida.sum())
    if n_corrompidas:
        idade_corrigida = tratado.loc[idade_corrompida, "idade"] / 10
        if not idade_corrigida.between(IDADE_MIN, IDADE_MAX).all():
            raise ValueError(
                f"Idades corrompidas não resultam em idade plausível ({IDADE_MIN}-{IDADE_MAX}) após divisão por 10."
            )
        tratado.loc[idade_corrompida, "idade"] = idade_corrigida
        print(f"  [LOG] {n_corrompidas} idades corrigidas (divididas por 10)")

    tratado["idade"] = tratado["idade"].astype("int64")

    # 3. Marca tempos de emprego fora do domínio observado como ausentes.
    tempo_invalido = ~tratado["tempo_emprego_anos"].between(TEMPO_EMPREGO_MIN, TEMPO_EMPREGO_MAX)
    n_tempo_invalido = int(tempo_invalido.sum())
    if n_tempo_invalido:
        tratado.loc[tempo_invalido, "tempo_emprego_anos"] = pd.NA
        print(f"  [LOG] {n_tempo_invalido} tempos de emprego fora de [{TEMPO_EMPREGO_MIN}, {TEMPO_EMPREGO_MAX}] marcados como ausentes")
    tratado["tempo_emprego_anos"] = tratado["tempo_emprego_anos"].astype("Int64")

    # 4. Remove duplicatas exatas (loga índices para auditoria).
    indices_dup = _log_duplicatas(tratado)
    if indices_dup:
        print(f"  [LOG] Removendo {len(indices_dup)} duplicatas exatas (mantendo primeira ocorrência)")
    tratado = tratado.drop_duplicates().reset_index(drop=True)

    # 5. Cria variável derivada: razão empréstimo/renda.
    tratado["razao_emprestimo_renda"] = (
        tratado["valor_solicitado"] / tratado["renda_mensal"]
    )

    return tratado


def main() -> None:
    bruto = pd.read_csv(SOURCE)
    print(f"Carregado: {len(bruto)} linhas x {bruto.shape[1]} colunas")
    print("Iniciando tratamento...")

    tratado = tratar_dados(bruto)
    tratado.to_csv(OUTPUT, index=False)

    print(f"\nResultado: {len(bruto)} -> {len(tratado)} linhas")
    print(f"Duplicatas removidas: {len(_log_duplicatas(bruto))}")
    print(f"Idades corrigidas (>100): {(bruto['idade'] > IDADE_CORRUPCAO_THRESHOLD).sum()}")
    print(
        f"Tempos de emprego inválidos (fora [{TEMPO_EMPREGO_MIN},{TEMPO_EMPREGO_MAX}]): "
        f"{(~bruto['tempo_emprego_anos'].between(TEMPO_EMPREGO_MIN, TEMPO_EMPREGO_MAX)).sum()}"
    )
    print(f"Valores ausentes após tratamento: {int(tratado.isna().sum().sum())}")
    print(f"Arquivo gerado: {OUTPUT}")


if __name__ == "__main__":
    main()