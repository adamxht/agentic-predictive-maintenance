import pandas as pd


def safe_downcast_with_check(df, datatype):
    """
    Safely downcast selected columns and print min/max validation report.
    """

    df_original = df.copy()

    # select columns
    cols = df.select_dtypes(include=[datatype]).columns

    print("\n=== SAFE DOWNSCAST ===\n")

    for col in cols:
        col_data = df[col]

        before_min = col_data.min()
        before_max = col_data.max()
        before_dtype = col_data.dtype

        # downcast attempt
        if pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="integer")
        elif pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="float")

        after_min = df[col].min()
        after_max = df[col].max()
        after_dtype = df[col].dtype

        safe = (
            (pd.isna(before_min) and pd.isna(after_min)) or before_min == after_min
        ) and ((pd.isna(before_max) and pd.isna(after_max)) or before_max == after_max)

        if not safe:
            df[col] = df_original[col]
            status = "ROLLBACK (min/max changed)"
        else:
            status = f"DOWNCASTED → {after_dtype}"

        print(f"\nColumn: {col}")
        print(f"  Before dtype: {before_dtype} | After dtype: {df[col].dtype}")
        print(f"  Min: {before_min} → {df[col].min()}")
        print(f"  Max: {before_max} → {df[col].max()}")
        print(f"  Status: {status}")

    return df
