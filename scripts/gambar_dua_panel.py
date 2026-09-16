"""Susun Fig. 2 naskah dari dua gambar sumber, ROC dan kalibrasi.

Gambar ini semula dibuat lewat perintah sekali pakai yang tidak tersimpan,
sehingga naskah mengklaim seluruh gambar dapat direproduksi padahal satu di
antaranya tidak. Skrip ini menutup celah itu.

Kedua gambar sumber dihasilkan oleh gambar_ulang.py pada protokol yang sama
dengan eksperimen utama, yaitu regresi logistik pada seed 42. Di sini keduanya
hanya ditempel berdampingan, tidak ada nilai yang dihitung ulang.

Jalankan:  python kode/gambar_dua_panel.py
Keluaran:  gambar/fig2_roc_kalibrasi.png
"""

import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
GAMBAR = Path(__file__).resolve().parents[1] / "figures"
KIRI = GAMBAR / "gambar_roc.png"
KANAN = GAMBAR / "gambar_kalibrasi.png"
KELUAR = GAMBAR / "fig2_roc_kalibrasi.png"
SELA = 60
LATAR = (255, 255, 255)


def main():
    kiri = Image.open(KIRI).convert("RGB")
    kanan = Image.open(KANAN).convert("RGB")
    lebar = kiri.width + SELA + kanan.width
    tinggi = max(kiri.height, kanan.height)
    kanvas = Image.new("RGB", (lebar, tinggi), LATAR)
    kanvas.paste(kiri, (0, (tinggi - kiri.height) // 2))
    kanvas.paste(kanan, (kiri.width + SELA, (tinggi - kanan.height) // 2))
    kanvas.save(KELUAR, dpi=(600, 600))
    print(f"tersimpan di {KELUAR}, ukuran {kanvas.width} x {kanvas.height}")


if __name__ == "__main__":
    main()
