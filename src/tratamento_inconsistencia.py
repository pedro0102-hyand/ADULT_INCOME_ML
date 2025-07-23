import pandas as pd

train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print("\n--- Tratamento de Valores '?' ---\n")

def substituir_unknown(df):
    cols = ['workclass', 'occupation', 'native-country']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace('?', 'Unknown')
    return df

train_df = substituir_unknown(train_df)
validation_df = substituir_unknown(validation_df)
test_df = substituir_unknown(test_df)

# Verificar se restam '?' nos datasets
for name, df in zip(['Train', 'Validation', 'Test'], [train_df, validation_df, test_df]):
    print(f"\nValores '?' restantes em {name} Dataset:")
    for col in ['workclass', 'occupation', 'native-country']:
        print(f"  {col}: {(df[col] == '?').sum()}")


train_df.to_csv("data/train_clean.csv", index=False)
validation_df.to_csv("data/validation_clean.csv", index=False)
test_df.to_csv("data/test_clean.csv", index=False)

