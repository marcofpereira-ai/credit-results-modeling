from pathlib import Path

import pandas as pd


SOURCE = Path(__file__).with_name("dataset.csv")
OUTPUT = Path(__file__).with_name("dataset_tratado.csv")


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica correções determinísticas e marca valores inválidos como ausentes."""
    tratado = df.copy()

    # Normaliza categorias textuais que chegaram com espaços extras.
    for coluna in ["estado", "escolaridade", "finalidade"]:
        tratado[coluna] = tratado[coluna].astype("string").str.strip()

    # Os 15 valores fora do domínio são exatamente idades válidas multiplicadas por 10.
    idade_corrompida = tratado["idade"] > 100
    idade_corrigida = tratado.loc[idade_corrompida, "idade"] / 10
    if not idade_corrigida.between(18, 75).all():
        raise ValueError("Há idades inválidas que não podem ser corrigidas com segurança.")
    tratado.loc[idade_corrompida, "idade"] = idade_corrigida
    tratado["idade"] = tratado["idade"].astype("int64")

    # O domínio observado para tempo de emprego é 0--30 anos. Valores acima
    # disso são inconsistentes; ficam ausentes para imputação dentro do treino.
    tempo_invalido = ~tratado["tempo_emprego_anos"].between(0, 30)
    tratado.loc[tempo_invalido, "tempo_emprego_anos"] = pd.NA
    tratado["tempo_emprego_anos"] = tratado["tempo_emprego_anos"].astype("Int64")

    # Remove somente duplicatas exatas, preservando a primeira ocorrência.
    tratado = tratado.drop_duplicates().reset_index(drop=True)

    # Razão entre o valor solicitado e a renda mensal declarada.
    tratado["razao_emprestimo_renda"] = (
        tratado["valor_solicitado"] / tratado["renda_mensal"]
    )

    return tratado


def main() -> None:
    bruto = pd.read_csv(SOURCE)
    tratado = tratar_dados(bruto)
    tratado.to_csv(OUTPUT, index=False)

    print(f"Linhas: {len(bruto)} -> {len(tratado)}")
    print(f"Duplicatas removidas: {bruto.duplicated().sum()}")
    print(f"Idades corrigidas: {(bruto['idade'] > 100).sum()}")
    print(
        "Tempos de emprego inválidos marcados como ausentes: "
        f"{(~bruto['tempo_emprego_anos'].between(0, 30)).sum()}"
    )
    print(f"Valores ausentes após tratamento: {int(tratado.isna().sum().sum())}")
    print(f"Arquivo gerado: {OUTPUT}")


if __name__ == "__main__":
    main()
