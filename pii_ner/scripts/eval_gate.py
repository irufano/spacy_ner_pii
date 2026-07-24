"""CI eval gate: run the hand-labeled gold examples against a freshly
trained model and fail (non-zero exit) if precision/recall drop below the
agreed thresholds.

    uv run python pii_ner/scripts/eval_gate.py --model pii_ner/models/model-best
    uv run python pii_ner/scripts/eval_gate.py --model ... --min-precision 0.90 --min-recall 0.97
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import spacy

DEFAULT_MIN_PRECISION = 0.90
DEFAULT_MIN_RECALL = 0.97
LABELS_TO_SCORE = {"PER", "ADR"}

# Copied verbatim from pii_ner_eval.ipynb cell 2a6bf20d - keep in sync manually.
eval_data = [
    # --- Nama tunggal & majemuk sederhana ---
    {"text": "Kemarin saya bertemu dengan Budi Santoso di Jakarta.", "gold": [("Budi Santoso", "PER")]},
    {"text": "Rina berangkat ke kantor pagi ini.", "gold": [("Rina", "PER")]},
    {"text": "Siti Aminah dan Jean-Luc Picard hadir dalam rapat itu.", "gold": [("Siti Aminah", "PER"), ("Jean-Luc Picard", "PER")]},
    {"text": "Elon Musk baru saja mengumumkan proyek barunya.", "gold": [("Elon Musk", "PER")]},
    {"text": "Megawati Soekarnoputri pernah menjabat sebagai Presiden RI.", "gold": [("Megawati Soekarnoputri", "PER")]},

    # --- Gelar akademis / profesi ---
    {"text": "Dr. Andi Wijaya, Sp.PD menerima pasien di RS Siloam Semarang.", "gold": [("Dr. Andi Wijaya, Sp.PD", "PER")]},
    {"text": "Rapat dipimpin oleh drg. Maya Kusuma dan dihadiri oleh Ir. Hartono, S.T., M.T.", "gold": [("drg. Maya Kusuma", "PER"), ("Ir. Hartono, S.T., M.T.", "PER")]},
    {"text": "Prof. Dr. Ing. B.J. Habibie dikenal sebagai Bapak Teknologi Indonesia.", "gold": [("Prof. Dr. Ing. B.J. Habibie", "PER")]},
    {"text": "Yusril Ihza Mahendra, S.H., M.Sc. menjadi kuasa hukum dalam sidang itu.", "gold": [("Yusril Ihza Mahendra, S.H., M.Sc.", "PER")]},

    # --- Gelar keagamaan / adat / militer ---
    {"text": "K.H. Ma'ruf Amin dan Hj. Siti Fatimah membuka acara bersama.", "gold": [("K.H. Ma'ruf Amin", "PER"), ("Hj. Siti Fatimah", "PER")]},
    {"text": "Ustadz Abdul Somad mengisi kajian di masjid kampus.", "gold": [("Ustadz Abdul Somad", "PER")]},
    {"text": "R.A. Kartini menulis banyak surat semasa hidupnya.", "gold": [("R.A. Kartini", "PER")]},
    {"text": "Sri Sultan Hamengkubuwono X menerima tamu kenegaraan.", "gold": [("Sri Sultan Hamengkubuwono X", "PER")]},
    {"text": "Jenderal Gatot Nurmantyo memberikan pidato pada upacara itu.", "gold": [("Jenderal Gatot Nurmantyo", "PER")]},

    # --- Nama internasional & partikel (van, de, bin/binti) ---
    {"text": "Ruud van Nistelrooy melatih tim nasional Belanda.", "gold": [("Ruud van Nistelrooy", "PER")]},
    {"text": "Fatima binti Muhammad tinggal di Kuala Lumpur.", "gold": [("Fatima binti Muhammad", "PER")]},
    {"text": "Van Gogh dan Leonardo da Vinci adalah pelukis legendaris.", "gold": [("Van Gogh", "PER"), ("Leonardo da Vinci", "PER")]},
    {"text": "Kim Jong-un bertemu dengan Xi Jinping di Beijing minggu lalu.", "gold": [("Kim Jong-un", "PER"), ("Xi Jinping", "PER")]},
    {"text": "Guillermo del Toro memenangkan piala Oscar tahun ini.", "gold": [("Guillermo del Toro", "PER")]},

    # --- Tanda hubung / apostrof / inisial ---
    {"text": "Di antara peserta terdapat O'Brien, McDonald, dan D'Angelo.", "gold": [("O'Brien", "PER"), ("McDonald", "PER"), ("D'Angelo", "PER")]},
    {"text": "J.K. Rowling menulis novel yang sangat terkenal.", "gold": [("J.K. Rowling", "PER")]},
    {"text": "Anne-Marie dan Jean-Pierre menikah tahun lalu.", "gold": [("Anne-Marie", "PER"), ("Jean-Pierre", "PER")]},

    # --- Posisi awal/akhir kalimat & enumerasi ---
    {"text": "Tim yang bertanding terdiri dari Cristiano Ronaldo, Lionel Messi, dan Neymar Jr.", "gold": [("Cristiano Ronaldo", "PER"), ("Lionel Messi", "PER"), ("Neymar Jr.", "PER")]},
    {"text": "Yang hadir dalam pertemuan itu adalah Budi, Rina, dan Ahmad.", "gold": [("Budi", "PER"), ("Rina", "PER"), ("Ahmad", "PER")]},

    # --- Dialog / kutipan ---
    {"text": '"Saya setuju," kata Budi, "asalkan Rina juga ikut."', "gold": [("Budi", "PER"), ("Rina", "PER")]},
    {"text": '"Kita harus bergerak cepat," ujar Sari sambil menutup laptopnya.', "gold": [("Sari", "PER")]},

    # --- Nama + jabatan (jabatan BUKAN bagian dari entity) ---
    {"text": "Menteri Keuangan RI, Sri Mulyani Indrawati, memaparkan APBN 2025.", "gold": [("Sri Mulyani Indrawati", "PER")]},
    {"text": "Gubernur DKI Jakarta sebelumnya adalah Anies Baswedan.", "gold": [("Anies Baswedan", "PER")]},
    {"text": "Wali Kota Surabaya, Eri Cahyadi, meresmikan taman kota baru.", "gold": [("Eri Cahyadi", "PER")]},
    {"text": "CEO Tesla, Elon Musk, bertemu dengan Mark Zuckerberg dari Meta.", "gold": [("Elon Musk", "PER"), ("Mark Zuckerberg", "PER")]},
    {"text": "Presiden AS ke-46, Joe Biden, berbicara dengan PM Inggris, Rishi Sunak.", "gold": [("Joe Biden", "PER"), ("Rishi Sunak", "PER")]},
    {"text": "Irfan Hidayat adalah seorang software engineer", "gold": [("Irfan Hidayat", "PER")]},

    # --- Hard negative: nama berdampingan dengan kontak/tanggal/uang ---
    {"text": "Nomor telepon Ani Yudhoyono dapat dihubungi di 0812-3456-7890.", "gold": [("Ani Yudhoyono", "PER")]},
    {"text": "Menurut catatan, alamat email John Doe adalah johndoe@example.com.", "gold": [("John Doe", "PER")]},
    {"text": "Surat itu ditandatangani oleh R.A. Kartini pada tahun 1902.", "gold": [("R.A. Kartini", "PER")]},
    {"text": "Faisal Rahman, S.E. mentransfer Rp1.000.000 ke rekening yayasan.", "gold": [("Faisal Rahman, S.E.", "PER")]},
    {"text": "Menurut Ustadz Abdul Somad, kajian akan dimulai pukul 19.00 WIB.", "gold": [("Ustadz Abdul Somad", "PER")]},

    # --- True negative: sama sekali tidak ada entity ---
    {"text": "Kantor pusatnya berlokasi di Jakarta dan cabangnya di Bandung.", "gold": []},
    {"text": "Menurut UU No. 8 Tahun 1999, konsumen memiliki hak-hak tertentu.", "gold": []},
    {"text": "Rapat membahas APBN 2025 akan dilaksanakan pukul 19.00 WIB.", "gold": []},
    {"text": "PT Astra International merilis laporan keuangan kuartal ini.", "gold": []},
    {"text": "Universitas Indonesia membuka pendaftaran mahasiswa baru bulan depan.", "gold": []},
    {"text": "COVID-19 masih menjadi perhatian utama Kementerian Kesehatan.", "gold": []},
    {"text": "Harga emas hari ini mencapai Rp1.200.000 per gram.", "gold": []},
    {"text": "Konferensi itu akan digelar di Bali pada 17 Agustus 2026.", "gold": []},

    # --- Kata ambigu: mirip nama tapi BUKAN nama orang dalam konteks ini ---
    {"text": "Bunga di taman itu mekar dengan indah setiap pagi.", "gold": []},
    {"text": "Mawar merah itu dijual seharga lima puluh ribu rupiah.", "gold": []},
    {"text": "Fajar mulai menyingsing di ufuk timur.", "gold": []},

    # --- Huruf kapital semua (headline berita) ---
    {"text": "BUDI SANTOSO RESMI DILANTIK SEBAGAI DIREKTUR BARU.", "gold": [("BUDI SANTOSO", "PER")]},

    # --- Multi-entity dalam satu kalimat panjang ---
    {"text": "Atlet bulu tangkis Indonesia, Kevin Sanjaya Sukamuljo dan Marcus Fernaldi Gideon, meraih medali emas.", "gold": [("Kevin Sanjaya Sukamuljo", "PER"), ("Marcus Fernaldi Gideon", "PER")]},
    {"text": "Albert Einstein dan Isaac Newton dikenal sebagai ilmuwan besar.", "gold": [("Albert Einstein", "PER"), ("Isaac Newton", "PER")]},

    # ============================================================
    # --- ADR (alamat domestik): pendek/sedang/lengkap ---
    {"text": "Alamat saya di Jl. Melati No. 12, Bandung.", "gold": [("Jl. Melati No. 12, Bandung", "ADR")]},
    {"text": "Silakan kirim paket ke Jl. Kenanga No. 7, RT 03/RW 05, Kel. Sukajadi, Kec. Sukasari, Bandung, Jawa Barat 40164.", "gold": [("Jl. Kenanga No. 7, RT 03/RW 05, Kel. Sukajadi, Kec. Sukasari, Bandung, Jawa Barat 40164", "ADR")]},
    {"text": "KTP menunjukkan alamat di Jalan Mawar No. 45A, Surabaya.", "gold": [("Jalan Mawar No. 45A, Surabaya", "ADR")]},
    {"text": "Rapat RT akan diadakan di Gg. Anggrek No. 3, Yogyakarta.", "gold": [("Gg. Anggrek No. 3, Yogyakarta", "ADR")]},

    # --- ADR negatif: bare kota/tempat, BUKAN alamat terstruktur ---
    {"text": "Saya tinggal di Jakarta sejak kecil.", "gold": []},
    {"text": "Kantor cabang baru akan dibuka di Surabaya.", "gold": []},
    {"text": "Kami sudah lama menetap di Bandung.", "gold": []},

    # --- Disambiguasi kunci: "Jl. + nama orang" (ADR) vs nama sendirian (PER) ---
    {"text": "Ahmad Yani adalah salah satu pahlawan revolusi Indonesia.", "gold": [("Ahmad Yani", "PER")]},
    {"text": "Kantor kami beralamat di Jl. Ahmad Yani No. 25, Surabaya.", "gold": [("Jl. Ahmad Yani No. 25, Surabaya", "ADR")]},
    {"text": "Diponegoro memimpin perlawanan pada Perang Jawa.", "gold": [("Diponegoro", "PER")]},
    {"text": "Toko itu berada di Jl. Diponegoro No. 88, Semarang.", "gold": [("Jl. Diponegoro No. 88, Semarang", "ADR")]},

    # --- Kombinasi PER + ADR dalam satu kalimat ---
    {"text": "Budi Santoso tinggal di Jl. Merdeka No. 10, Bandung.", "gold": [("Budi Santoso", "PER"), ("Jl. Merdeka No. 10, Bandung", "ADR")]},
    {"text": "Nama: Siti Aminah, Alamat: Jl. Sudirman No. 5, RT 01/RW 02, Kel. Menteng, Jakarta.", "gold": [("Siti Aminah", "PER"), ("Jl. Sudirman No. 5, RT 01/RW 02, Kel. Menteng, Jakarta", "ADR")]},
    {"text": "Paket untuk Elon Musk dikirim ke Jl. Cendrawasih No. 2, Denpasar.", "gold": [("Elon Musk", "PER"), ("Jl. Cendrawasih No. 2, Denpasar", "ADR")]},

    # ============================================================
    # --- ADR internasional: format nomor-di-depan + kode pos per negara ---
    {"text": "Kantor cabang kami berada di 123 Oxford Street, London, SW1A 1AA, Inggris.", "gold": [("123 Oxford Street, London, SW1A 1AA, Inggris", "ADR")]},
    {"text": "Silakan kirim dokumen ke 45 Park Avenue, New York, 10001, Amerika Serikat.", "gold": [("45 Park Avenue, New York, 10001, Amerika Serikat", "ADR")]},
    {"text": "Ia baru saja pindah ke 12 King Road, Sydney, Australia.", "gold": [("12 King Road, Sydney, Australia", "ADR")]},
    {"text": "Kantor pusat perusahaan itu berlokasi di 88 Orchard Boulevard, Singapura, Singapura.", "gold": [("88 Orchard Boulevard, Singapura, Singapura", "ADR")]},

    # --- ADR negatif internasional: bare kota luar negeri, BUKAN alamat ---
    {"text": "Ia sedang liburan di Tokyo sekarang.", "gold": []},
    {"text": "Perusahaan itu membuka kantor baru di Dubai.", "gold": []},

    # --- Disambiguasi internasional: "nama jalan dari tokoh" (ADR) vs nama sendirian (PER) ---
    {"text": "Washington adalah presiden pertama Amerika Serikat.", "gold": [("Washington", "PER")]},
    {"text": "Toko itu berlokasi di 200 Washington Street, New York, Amerika Serikat.", "gold": [("200 Washington Street, New York, Amerika Serikat", "ADR")]},
    {"text": "Lincoln dikenal karena menghapuskan perbudakan di Amerika Serikat.", "gold": [("Lincoln", "PER")]},
    {"text": "Kantor pengacara itu berada di 15 Lincoln Avenue, Los Angeles, Amerika Serikat.", "gold": [("15 Lincoln Avenue, Los Angeles, Amerika Serikat", "ADR")]},

    # --- Kombinasi PER + ADR internasional ---
    {"text": "John Doe tinggal di 77 Baker Street, London, Inggris.", "gold": [("John Doe", "PER"), ("77 Baker Street, London, Inggris", "ADR")]},

    # ============================================================
    # --- Enumerasi & koma (area lemah teridentifikasi): pastikan batas entity tidak melebar ke lokasi lain ---
    {"text": "Peserta pelatihan meliputi Dedi, Fitri, dan Yoga dari kantor cabang.", "gold": [("Dedi", "PER"), ("Fitri", "PER"), ("Yoga", "PER")]},
    {"text": "Rapat dihadiri oleh Bambang, Wati, Joko, dan Lestari sore ini.", "gold": [("Bambang", "PER"), ("Wati", "PER"), ("Joko", "PER"), ("Lestari", "PER")]},
    {"text": "Panitia terdiri dari Rian dan Dewi dari divisi humas.", "gold": [("Rian", "PER"), ("Dewi", "PER")]},
    {"text": "Ia berdiskusi dengan Hendra di kantin.", "gold": [("Hendra", "PER")]},
    {"text": "Surat itu dikirim oleh Maria di Semarang minggu lalu.", "gold": [("Maria", "PER")]},

    # --- Dialog / kutipan (tambahan) ---
    {"text": '"Tolong hubungi saya besok," pesan Doni lewat telepon.', "gold": [("Doni", "PER")]},
    {"text": 'Menurut Lisa, "acara akan dimulai lebih awal."', "gold": [("Lisa", "PER")]},
    {"text": '"Kami akan berangkat pagi," kata Wahyu dan Nita bersamaan.', "gold": [("Wahyu", "PER"), ("Nita", "PER")]},

    # --- Headline huruf kapital semua (tambahan) ---
    {"text": "RINA WIJAYA TERPILIH SEBAGAI KETUA UMUM ORGANISASI.", "gold": [("RINA WIJAYA", "PER")]},
    {"text": "GEMPA BUMI GUNCANG WILAYAH SELATAN JAWA.", "gold": []},
    {"text": "MENTERI PERDAGANGAN TINJAU PASAR INDUK.", "gold": []},

    # --- Disambiguasi nama tokoh vs nama tempat (area lemah teridentifikasi) ---
    {"text": "Bandung dikenal sebagai kota kembang di Jawa Barat.", "gold": []},
    {"text": "Sukarno adalah proklamator kemerdekaan Indonesia.", "gold": [("Sukarno", "PER")]},
    {"text": "Soeharto memimpin Indonesia selama lebih dari tiga dekade.", "gold": [("Soeharto", "PER")]},
    {"text": "Jenderal Sudirman memimpin perang gerilya melawan Belanda.", "gold": [("Jenderal Sudirman", "PER")]},
    {"text": "Jalan di depan kampus itu dinamai sesuai nama Sudirman.", "gold": []},

    # --- Media sosial / teks informal ---
    {"text": "@budisantoso baru saja mengunggah foto liburannya.", "gold": [("budisantoso", "PER")]},
    {"text": "min tolong follow back ya, makasih min admin.", "gold": []},
    {"text": "Halo semua, nama gue Tio, salam kenal ya!", "gold": [("Tio", "PER")]},

    # --- Form / tabel data (variasi tambahan) ---
    {"text": "Nama Lengkap: Rangga Pratama\nAlamat: Jl. Flamboyan No. 9, Malang", "gold": [("Rangga Pratama", "PER"), ("Jl. Flamboyan No. 9, Malang", "ADR")]},
    {"text": "Penerima: Wulan Sari, Tujuan: Jl. Kaliurang KM 5, Yogyakarta", "gold": [("Wulan Sari", "PER"), ("Jl. Kaliurang KM 5, Yogyakarta", "ADR")]},

    # --- Alamat format kost/dusun/perumahan (belum tercakup) ---
    {"text": "Kost berada di Dusun Krajan RT 02/RW 04, Desa Sukamaju, Kec. Cileunyi, Kab. Bandung.", "gold": [("Dusun Krajan RT 02/RW 04, Desa Sukamaju, Kec. Cileunyi, Kab. Bandung", "ADR")]},
    {"text": "Rumah dijual di Perumahan Griya Asri Blok C2 No. 14, Bekasi.", "gold": [("Perumahan Griya Asri Blok C2 No. 14, Bekasi", "ADR")]},
    {"text": "Silakan kirim ke Apartemen Kalibata City Tower D Lantai 12, Jakarta Selatan.", "gold": [("Apartemen Kalibata City Tower D Lantai 12, Jakarta Selatan", "ADR")]},

    # --- Alamat: PO Box ---
    {"text": "Kirim ke PO Box 1234, Jakarta 10110.", "gold": [("PO Box 1234, Jakarta 10110", "ADR")]},

    # --- True negative: organisasi yang mengandung kata mirip nama orang ---
    {"text": "Yayasan Ahmad Dahlan menggelar bakti sosial akhir pekan ini.", "gold": []},
    {"text": "Bank Mandiri membuka layanan baru di seluruh cabang.", "gold": []},
    {"text": "Rumah Sakit Bunda merawat pasien dengan fasilitas lengkap.", "gold": []},
    {"text": "Sekolah Dasar Kartini menerima murid baru tahun ini.", "gold": []},
]


def evaluate_model(nlp, data):
    """Identical scoring logic to pii_ner_eval.ipynb's evaluate_model:
    exact-match (substring, label) comparison against gold, greedy TP
    matching, per-label + overall precision/recall/F1, plus FP/FN listing.
    """
    tp = fp = fn = 0
    fp_list, fn_list = [], []
    per_label = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in LABELS_TO_SCORE}

    for item in data:
        doc = nlp(item["text"])
        predicted = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in LABELS_TO_SCORE]
        gold_remaining = item["gold"].copy()
        for p in predicted:
            if p in gold_remaining:
                tp += 1
                per_label[p[1]]["tp"] += 1
                gold_remaining.remove(p)
            else:
                fp += 1
                if p[1] in per_label:
                    per_label[p[1]]["fp"] += 1
                fp_list.append((item["text"], p))
        for g in gold_remaining:
            fn += 1
            per_label[g[1]]["fn"] += 1
            fn_list.append((item["text"], g))

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    precision, recall, f1 = prf(tp, fp, fn)
    print(f"Total kalimat uji: {len(data)}")
    print("=== Keseluruhan (semua label) ===")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1-score  : {f1:.2%}\n")

    per_label_result = {}
    print("=== Per Label ===")
    for lbl, c in per_label.items():
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        per_label_result[lbl] = {"precision": p, "recall": r, "f1": f}
        print(f"{lbl:8s} TP={c['tp']:<3d} FP={c['fp']:<3d} FN={c['fn']:<3d}  P={p:.2%}  R={r:.2%}  F1={f:.2%}")

    if fp_list:
        print("\nFalse Positive:")
        for text, p in fp_list:
            print(f"  - '{p[0]}' ({p[1]})  <-  \"{text}\"")
    if fn_list:
        print("\nFalse Negative:")
        for text, g in fn_list:
            print(f"  - '{g[0]}' ({g[1]})  <-  \"{text}\"")

    return {"precision": precision, "recall": recall, "f1": f1, "per_label": per_label_result}


@click.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--min-precision", type=float, default=DEFAULT_MIN_PRECISION, show_default=True)
@click.option("--min-recall", type=float, default=DEFAULT_MIN_RECALL, show_default=True)
def main(model_path: Path, min_precision: float, min_recall: float) -> None:
    nlp = spacy.load(model_path)
    result = evaluate_model(nlp, eval_data)

    print("\n=== Eval Gate ===")
    print(f"Thresholds -> min precision: {min_precision:.2%}, min recall: {min_recall:.2%}")
    print(f"Actual     -> precision: {result['precision']:.2%}, recall: {result['recall']:.2%}, f1: {result['f1']:.2%}")

    failures = []
    if result["precision"] < min_precision:
        failures.append(f"precision {result['precision']:.2%} < required {min_precision:.2%}")
    if result["recall"] < min_recall:
        failures.append(f"recall {result['recall']:.2%} < required {min_recall:.2%}")

    if failures:
        print("\nEVAL GATE FAILED:")
        for reason in failures:
            print(f"  - {reason}")
        sys.exit(1)

    print("\nEval gate passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
