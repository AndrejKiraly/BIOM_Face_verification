import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


BASE_PATH = "FaceVeriication"


def show_pair(img1_path, img2_path, label):
    """Zobrazí dvojicu obrázkov s popisom, či ide o rovnakú osobu."""
    full_img1_path = os.path.join(BASE_PATH, img1_path)
    full_img2_path = os.path.join(BASE_PATH, img2_path)

    img1 = Image.open(full_img1_path)
    img2 = Image.open(full_img2_path)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(f"Target: {'Zhodné' if label == 1 else 'Rozdielne'}", fontsize=14)

    axs[0].imshow(img1)
    axs[0].axis("off")
    axs[0].set_title("Obrázok 1")

    axs[1].imshow(img2)
    axs[1].axis("off")
    axs[1].set_title("Obrázok 2")

    plt.tight_layout()
    plt.show()


def main():
    # Načítanie CSV
    csv_path = os.path.join(BASE_PATH, "face_verification.csv")
    df = pd.read_csv(csv_path)

    # Štatistiky
    same_count = (df["target"] == 1).sum()
    different_count = (df["target"] == 0).sum()

    print(f"Zhodné páry (rovnaká osoba): {same_count}")
    print(f"Rozdielne páry (rôzne osoby): {different_count}")

    # Zobraz prvé 3 zhodné páry
    print("\n📸 Zhodné páry:")
    for _, row in df[df["target"] == 1].head(3).iterrows():
        show_pair(row["image_1"], row["image_2"], row["target"])

    # Zobraz prvé 3 rozdielne páry
    print("\n📸 Rozdielne páry:")
    for _, row in df[df["target"] == 0].head(3).iterrows():
        show_pair(row["image_1"], row["image_2"], row["target"])


if __name__ == "__main__":
    main()
