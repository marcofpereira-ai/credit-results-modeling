from pathlib import Path
import re
import unicodedata
from typing import List

import pandas as pd

SOURCE = Path(__file__).with_name("dataset.csv")
OUTPUT = Path(__file__).with_name("dataset_tratado.csv")

# --- Constantes de domínio configuráveis ---
IDADE_MIN = 18
IDADE_MAX = 75
IDADE_CORRUPCAO_THRESHOLD = 100  # idades > 100 são consideradas corrompidas (x10)
TEMPO_EMPREGO_MIN = 0
TEMPO_EMPREGO_MAX = 30
FORMATOS_CAIXA_CATEGORIAS = {
    "estado": "upper",
    "escolaridade": "title",
    "finalidade": "title",
}
ENCODINGS_CSV = ("utf-8", "cp1252", "latin1")
COLUNAS_ESPERADAS = [
    "idade", "estado", "escolaridade", "renda_mensal",
    "tempo_emprego_anos", "score_credito", "valor_solicitado",
    "prazo_meses", "finalidade", "dividas_ativas", "inadimplente",
]


def _corrigir_mojibake(texto: str) -> str:
    """Tenta recuperar texto UTF-8 interpretado incorretamente como Windows-1252/Latin-1."""
    marcadores_mojibake = ("Ã", "Â", "â", "ð", "�")
    for _ in range(2):
        if not any(marcador in texto for marcador in marcadores_mojibake):
            break
        for encoding in ("cp1252", "latin1"):
            try:
                candidato = texto.encode(encoding).decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
            if candidato != texto:
                texto = candidato
                break
        else:
            break
    return texto


def sanitizar_texto(valor: object, caixa: str = "preserve") -> object:
    """Limpa e normaliza texto de forma determinística.

    A função corrige mojibake recuperável, aplica Unicode NFKD para remover
    acentos e cedilhas, troca pontuação e símbolos por espaços, elimina
    espaços excedentes e aplica a caixa solicitada. Valores ausentes são
    preservados para que o tratamento numérico continue compatível com
    pandas.

    Args:
        valor: Valor textual a ser sanitizado.
        caixa: Formato final: ``"preserve"``, ``"upper"``, ``"lower"`` ou
            ``"title"``.

    Returns:
        Texto sanitizado ou o próprio valor quando ele representa ausência.

    Raises:
        ValueError: Se ``caixa`` não for um formato suportado.
    """
    if valor is None or pd.isna(valor):
        return valor

    texto = _corrigir_mojibake(str(valor))
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(caractere for caractere in texto if not unicodedata.combining(caractere))
    texto = re.sub(r"[^A-Za-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    if caixa == "upper":
        return texto.upper()
    if caixa == "lower":
        return texto.lower()
    if caixa == "title":
        return texto.title()
    if caixa == "preserve":
        return texto
    raise ValueError(f"Formato de caixa inválido: {caixa}")


def _ler_csv_seguro(caminho: Path) -> pd.DataFrame:
    """Lê CSV tentando UTF-8 e usando CP1252/Latin-1 como fallback."""
    ultimo_erro = None
    for encoding in ENCODINGS_CSV:
        try:
            return pd.read_csv(caminho, encoding=encoding)
        except UnicodeDecodeError as erro:
            ultimo_erro = erro
    raise ValueError(f"Não foi possível decodificar o arquivo CSV: {caminho}") from ultimo_erro



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
    """Aplica normalização textual e correções determinísticas aos dados.

    Passos:
    1. Valida schema de entrada
    2. Sanitiza categorias textuais (mojibake, acentos, símbolos, espaços e caixa)
    3. Corrige idades corrompidas (x10) se resultarem em idade plausível
    4. Marca tempos de emprego fora do domínio como ausentes (imputação no pipeline)
    5. Remove duplicatas exatas (loga índices)
    6. Cria variável derivada: razao_emprestimo_renda
    """
    _validar_schema(df)
    tratado = df.copy()

    # 1. Sanitiza categorias textuais antes das regras dependentes de igualdade.
    for coluna, caixa in FORMATOS_CAIXA_CATEGORIAS.items():
        tratado[coluna] = tratado[coluna].map(
            lambda valor: sanitizar_texto(valor, caixa=caixa)
        )

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
    bruto = _ler_csv_seguro(SOURCE)
    print(f"Carregado: {len(bruto)} linhas x {bruto.shape[1]} colunas")
    print("Iniciando tratamento...")

    tratado = tratar_dados(bruto)
    tratado.to_csv(OUTPUT, index=False, encoding="utf-8")

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