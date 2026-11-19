# Demand Radar 📡

> **Autonomous Market Demand & Product Discovery Pipeline using Local LLMs (Ollama)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-black.svg)](https://ollama.ai/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![UMAP](https://img.shields.io/badge/UMAP-Clustering-orange.svg)](https://github.com/lmcinnes/umap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An automated data pipeline and intelligence engine that mines unmet software needs, consumer pain-points, and willingness-to-pay signals across **Hacker News**, **Stack Exchange**, **Bluesky**, **App Store reviews (1–3★)**, and **WordPress.org plugins**. Runs 100% locally on your machine via **Ollama** embeddings (`nomic-embed-text`) and a 2-stage LLM classification pipeline (`gemma4:12b` / `qwen2.5:7b`).

---

## Επισκόπηση / Overview

Βρίσκει ιδέες για προϊόντα/επιχειρήσεις ψάχνοντας ανάγκες που ο κόσμος λέει ότι **δεν καλύπτονται**:
«I wish there was…», «is there a tool that…», «I'd pay for…», «we still do this manually…», και στις κακές
κριτικές εφαρμογών που ο κόσμος ήδη πληρώνει: «δεν έχει export», «το έβαλαν στη συνδρομή», «το αφαίρεσαν».
Η συλλογή και η ανάλυση τρέχουν τοπικά στο Mac σου με Ollama. Το τοπικό μοντέλο **μαζεύει και οργανώνει
αποδείξεις** (ποιοι έχουν το πρόβλημα, πόσοι, τι ζητούν, τι χρησιμοποιούν σήμερα). Δεν κρίνει την αγορά: αυτό το
κάνει στο τέλος ένα session του Claude με αναζήτηση στο web, διαβάζοντας το `brief.md`.

```
collect            ->  extract  ->  classify (LLM, 2 στάδια)       ->  analyze                     ->  τελική κρίση
Hacker News (3 χρ.)    regex        1. είναι ανάγκη; (γρήγορο)          clustering ιδεών software/       Claude + web search
Stack Exchange (6 χρ.)              2. τι λύση; φτιάχνεται; ποιος;      υπηρεσίας -> score (μόνο         διαβάζει το brief.md
Bluesky (δείγμα)                       προϊόντα; (+ «θα πλήρωνε» με regex) αποδείξεις) -> περιγραφή      -> shortlist 5–10
App Store, WordPress.org                                                -> report.md + brief.md
(κριτικές 1–3★)
```

## Εγκατάσταση (μία φορά)

```bash
git clone https://github.com/georgepelal/demand-radar.git
cd demand-radar
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
ollama pull gemma4:12b
ollama pull nomic-embed-text
```

## Τρέξιμο

Κάθε φορά που ανοίγεις νέο terminal: `cd demand-radar && source .venv/bin/activate`

Όλα μαζί (για ώρες την πρώτη φορά, μετά μόνο τα καινούρια):

```bash
python -m ideafinder collect && python -m ideafinder extract && python -m ideafinder classify --model gemma4:12b && python -m ideafinder analyze --summary-model gemma4:12b
```

| Εντολή | Τι κάνει |
|---|---|
| `collect` | κατεβάζει από Hacker News, Stack Exchange, Bluesky, App Store και WordPress.org μόνο ό,τι λείπει από τη βάση (`--source hn` για μία πηγή, `--source as,wp` μόνο τις κριτικές, ~1 ώρα: ~45 λεπτά το App Store, ~5 λεπτά το WordPress.org) |
| `extract` | κρατά από κάθε κείμενο τις προτάσεις με ανάγκη/παράπονο |
| `classify` | το αργό βήμα, σε 2 στάδια. Ctrl+C είναι ασφαλές: συνεχίζει από εκεί που έμεινε. Με `--dry-run` δείχνει μόνο πόσα θα ελέγξει και πόση ώρα θα πάρει |
| `analyze` | clustering + report.md + brief.md (~20–30 λεπτά· ξανατρέχει μόνο του, χωρίς νέο classify) |
| `compete` | προαιρετικό: για τις top 15 ιδέες του τελευταίου report: τι υπάρχει ήδη (Show HN, GitHub) και αν μένει κενό → `compete.md` |
| `stats` | τι υπάρχει στη βάση |
| `evaluate` | σύγκριση μοντέλων (δες παρακάτω) |
| `doctor` | διάγνωση ενός μοντέλου του Ollama που κολλάει |

Το `classify`:
- **στάδιο 1:** ρωτά για κάθε απόσπασμα μόνο «είναι ανάγκη;». Είναι γρήγορο και προτιμά να κρατήσει κάτι αμφίβολο.
- **στάδιο 2:** μόνο για όσα πέρασαν: τύπος λύσης, αν φτιάχνεται από έναν developer, ποιος μιλά, προϊόντα,
  κατηγορία. Μπορεί ακόμα να απορρίψει. Το «θα πλήρωνε» βγαίνει κυρίως από regex («I'd pay for», «take my money»),
  που στη δοκιμή ήταν πολύ πιο αξιόπιστο από το LLM.
- Τα μοντέλα των δύο σταδίων μπορούν να διαφέρουν: `--model qwen2.5:7b --detail-model gemma4:12b`.
- Ξεκινά από τα πιο υποσχόμενα, οπότε και με `--limit 1000` βγαίνει χρήσιμο report. Παραλείπει όσα μόνο ξεσπάνε
  («so frustrating»), εκτός αν βάλεις `--include-weak`. Με `--recheck-rejected` ξαναελέγχει όσα απέρριψε παλιότερη
  έκδοση.
- Στην αρχή γράφει πόση ώρα θα πάρει. Μετά από κάθε τρέξιμο κρατά στη βάση την πραγματική ταχύτητα κάθε μοντέλου
  και πόσα περνούν το στάδιο 1, ώστε η επόμενη εκτίμηση να είναι ακριβέστερη. Πριν από κάτι μεγάλο:
  `python -m ideafinder classify --model gemma4:12b --dry-run`.

Το report γράφεται στο `reports/<ημερομηνία>/`:

- `brief.md`: **αυτό δίνεις στο τελικό session του Claude.** Έχει οδηγίες για τον έλεγχο και τις top 40 ανάγκες
  με περιγραφή, αριθμούς και 4 αποσπάσματα η καθεμία (από διαφορετικές πηγές όπου γίνεται), και στο τέλος τις
  πιο δυνατές ανάγκες των μικρότερων πηγών (Stack Exchange, Bluesky, App Store, WordPress.org), για να μη
  χάνονται κάτω από το Hacker News
- `report.md`: όλες οι ιδέες software/υπηρεσίας, οι top με αναλυτική περιγραφή, και στο τέλος οι υπόλοιπες
  ανάγκες (hardware, content, πολιτική)
- `clusters.csv`: ένα cluster ανά γραμμή (για Excel/Sheets)
- `items.csv`: κάθε απόσπασμα με την πηγή, το cluster, τα πεδία του LLM και link
- `clusters.json`: όλα τα δεδομένα για επόμενη επεξεργασία
- `compete.md` (μετά το `compete`): για κάθε top ιδέα τι βρέθηκε, ετυμηγορία (open / some competition / crowded),
  πόσο κενό μένει (0–2) και πώς θα μπορούσε να ξεχωρίσει κάποιος νέος. Ένα «open» θέλει πάντα και δικό σου
  έλεγχο (Google, Product Hunt, app stores): η αναζήτηση βλέπει μόνο Show HN και GitHub.

## Πηγές

| Πηγή | Τι φέρνει | Όρια |
|---|---|---|
| Hacker News | πολλές ανάγκες, κυρίως developers | κανένα (δωρεάν API) |
| Stack Exchange | ερωτήσεις «Is there a website/tool…» από 25 sites εκτός προγραμματισμού (Personal Finance, Travel, DIY, Parenting, Workplace…) | ~300 αιτήματα τη μέρα χωρίς κλειδί. Αν τελειώσουν, ξανατρέχεις `collect` την επόμενη μέρα. Με δωρεάν κλειδί (https://stackapps.com/apps/oauth/register): `export IDEAFINDER_SE_KEY=...` για 10.000 |
| Bluesky | καταναλωτές, creators | χωρίς λογαριασμό μόνο ένα δείγμα: έως 15 posts ανά φράση και μήνα (`--bsky-per-month`). Περίπου 1 στα 4 είναι πραγματική ανάγκη |
| App Store (`as`) | κριτικές 1–3★ των top-grossing εφαρμογών σε 10 κατηγορίες software (Productivity, Business, Finance, Health & Fitness, Education, Photo & Video, Utilities, Lifestyle, Graphics & Design, Developer Tools), 15 ανά κατηγορία. Εκτός: Google, Apple, Meta, Snap, Amazon, OpenAI, Anthropic, X, ByteDance και εφαρμογές γνωριμιών | δημόσιο RSS της Apple, χωρίς κλειδί· έως 4 σελίδες των 50 «most helpful» κριτικών ανά εφαρμογή (`--review-pages`). Το feed συχνά απαντά άδειο, οπότε κάθε σελίδα ξαναζητιέται· παρ' όλα αυτά, στο τρέξιμο του Νοεμβρίου 2026 οι 68 από τις 149 εφαρμογές δεν έφεραν καμία κριτική. Ξανατρέχοντας `collect --source as` αργότερα, ζητούνται ξανά μόνο όσες έχουν λιγότερες από 8 |
| WordPress.org (`wp`) | κριτικές 1–3★ των 100 πιο δημοφιλών plugins (εκτός όσων γράφει η Google ή το WordPress.org) | δημόσιο API και RSS· οι 30 πιο πρόσφατες κριτικές ανά αριθμό αστεριών |

Από κάθε εφαρμογή ή plugin κρατιούνται μόνο οι **8 πιο χρήσιμες κριτικές** (`--per-app`): όσες λένε ότι κάτι λείπει,
ζητούν κάτι, ότι κάτι αφαιρέθηκε ή μπήκε σε συνδρομή, ή ότι φεύγουν για άλλο προϊόν· προτιμώνται όσες έχουν πολλές
ψήφους «helpful» και 2–3 αστέρια (πιο συγκεκριμένες από τα 1★). Έτσι ένα πλήρες τρέξιμο φέρνει έως ~2.000 κριτικές.
Στο τρέξιμο του Νοεμβρίου 2026 ήρθαν 1.198 (608 App Store, 590 WordPress.org), το ~60% πέρασε το στάδιο 1, το
classify με `gemma4:12b` κράτησε ~1 ώρα 50 λεπτά, και 659 κρίθηκαν πραγματικές ανάγκες. Για το LLM, ανάγκη σε κριτική = κάτι που το προϊόν δεν έχει, κλειδώνει πίσω από πληρωμή,
χρεώνει ακριβά, αφαίρεσε ή περιορίζει (sync, export, συσκευές, integrations, privacy). Δεν είναι ανάγκη: bugs,
crashes, αργή λειτουργία, προβλήματα λογαριασμού/χρέωσης/επιστροφής χρημάτων/υποστήριξης, σκέτες διαφημίσεις.
Σε αυτές τις πηγές κάθε εφαρμογή ή plugin μετρά ως ένα «thread», οπότε **threads = σε πόσα διαφορετικά προϊόντα
εμφανίζεται το ίδιο παράπονο** (κενό σε όλη την κατηγορία, όχι ιδιοτροπία ενός προϊόντος).

Το Reddit δεν μπαίνει: από τον Νοέμβριο 2025 θέλει έγκριση για κάθε νέο κλειδί API και τα προσωπικά scripts
συνήθως απορρίπτονται.

## Ποιο μοντέλο;

```bash
python -m ideafinder evaluate --models qwen2.5:7b,gemma4:12b,qwen2.5:7b+gemma4:12b
```

Τρέχει κάθε μοντέλο στα αποσπάσματα του `eval/labels.jsonl` που έχει χαρακτηρίσει ο Claude (236 από συζητήσεις
και 50 κριτικές) και δείχνει:

- **need P/R/F1**: αν βρίσκει τις πραγματικές ανάγκες (P = πόσα από όσα κρατά είναι σωστά, R = πόσα από τα σωστά
  βρίσκει)
- **stage1 recall**: πόσες πραγματικές ανάγκες αφήνει να περάσουν ο γρήγορος πρώτος έλεγχος
- **software P/R/F1**: αν ξεχωρίζει σωστά τις ιδέες που θα κατατάξει το report
- **pays**: αν πιάνει σωστά το «θα πλήρωνε»
- **s/item**: ταχύτητα
- κάτω από τον πίνακα: need P/R/F1 **χωριστά για τις κριτικές** και για τις συζητήσεις (το δεύτερο συγκρίνεται με
  τα τρεξίματα πριν μπουν οι κριτικές)

`a+b` σημαίνει: μοντέλο a για το στάδιο 1, μοντέλο b για το στάδιο 2. Με `--show-errors` βλέπεις τα λάθη και πού
μπερδεύεται. Κάθε τρέξιμο αποθηκεύεται στο `eval/runs/` και μια γραμμή στο `eval/history.csv`, για να συγκρίνεις
εκδόσεις. Οι ετικέτες είναι προσεκτικές αλλά όχι αλάνθαστες: διαφορές 2–3 μονάδων είναι θόρυβος.

Όταν διορθώνεται μια ετικέτα (το παλιό label και ο λόγος μένουν στο πεδίο `revised`), όλα τα παλιά τρεξίματα
ξαναβαθμολογούνται σε δευτερόλεπτα, χωρίς να ξανατρέξει κανένα μοντέλο:

```bash
python -m ideafinder evaluate --rescore
```

Για τις περιγραφές των clusters (μόνο ~50 κλήσεις) αξίζει μεγαλύτερο μοντέλο: `analyze --summary-model gemma4:12b`.

Αν ένα μοντέλο κολλάει: `python -m ideafinder doctor --model gemma4:12b`. Αν το `ollama ps` δείχνει μοντέλο
κολλημένο σε «Stopping...», κάνε επανεκκίνηση του Ollama.

## Πώς διαβάζεις το report

- **People / Threads**: πόσοι διαφορετικοί άνθρωποι, σε πόσες διαφορετικές συζητήσεις, έχουν το ίδιο πρόβλημα.
  Για τις κριτικές, threads = διαφορετικές εφαρμογές/plugins, και το «Low-star reviews of» λέει ποιες.
- **Would pay**: ποσοστό που λέει ότι θα πλήρωνε ή ήδη πληρώνει (σε παρένθεση: όσοι περιγράφουν κόστος).
- **Growth**: αν το θέμα εμφανίζεται συχνότερα στο πιο πρόσφατο τρίτο της περιόδου κάθε πηγής
  (×1.5 = 50% πάνω από το αναμενόμενο). Οι κριτικές δεν μετρούν εδώ.
- **Sources**: από ποιες κοινότητες έρχεται η ανάγκη. Το ίδιο πρόβλημα σε δύο ή τρεις διαφορετικές πηγές είναι πιο
  δυνατό σήμα από το ίδιο νούμερο σε μία.
- **Score** (μόνο αποδείξεις) = log(1+people) × log(1+threads) × (1 + 1.5 × would pay + 0.5 × implied) × growth ×
  (1 + 0.3 για κάθε επιπλέον πηγή). ⚠ = το cluster ανακατεύει διαφορετικά προβλήματα, οπότε τα νούμερα δεν αφορούν
  μία ανάγκη (×0.6). Το LLM δεν βαθμολογεί αγορά ή ανταγωνισμό: αυτό γίνεται στο τελικό session.
- Κάθε περιγραφή έχει: **Problem**, **Who has it**, **What they ask for**, **What they use today** και
  **Check first** (τι πρέπει να ελεγχθεί πρώτο στο web).
- **Exists**: πόσα Show HN launches και πόσοι «το φτιάχνω» βρέθηκαν μέσα στο cluster.
- **Related**: clusters με πολύ κοντινό θέμα (συχνά η ίδια αγορά από άλλη γωνία).

Μετά το clustering, αποσπάσματα που έμειναν εκτός ομάδας μπαίνουν στο κοντινότερο cluster αν είναι τόσο κοντά όσο
ένα τυπικό μέλος του, και τα top clusters που το LLM κρίνει ανάμεικτα (coherence 1–2) σπάνε στα δύο και
περιγράφονται ξανά.

Το report δίνει **υποψήφιες ιδέες, όχι επικυρωμένες**. Επόμενο βήμα: άνοιξε ένα session του Claude με web search,
δώσε του το `brief.md` (ή πες του πού είναι) και ζήτα του να ακολουθήσει τις οδηγίες στην αρχή του. Για όποια ιδέα
περάσει: άνοιξε τα links και διάβασε τις συζητήσεις, και μίλα με 5–10 ανθρώπους που έχουν το πρόβλημα.

## Γνωστοί περιορισμοί

- **Τα clusters ανακατεύουν συγγενικά αλλά διαφορετικά προβλήματα, και τα νούμερα φουσκώνουν ακόμα και χωρίς ⚠.**
  Παράδειγμα από το τρέξιμο του Νοεμβρίου 2026: το cluster «payment and invoicing tools» είχε 41 άτομα και «would pay
  24%», αλλά μόνο 12 αποσπάσματα αφορούσαν πραγματικά τιμολόγια. Τα υπόλοιπα ήταν εφαρμογές για χρέη, crypto, κάρτες
  κ.λπ. Πριν εμπιστευτείς τα νούμερα μιας ιδέας, διάβασε τα αποσπάσματά της στο `items.csv`.
- **Το όριο των 8 κριτικών ανά προϊόν υποτιμά τα προϊόντα με πολλά παράπονα.** Για το ίδιο cluster, το Joist
  εμφανιζόταν με 4 κριτικές του 2019, ενώ στο App Store έχει 79 κριτικές 1–3★ μόνο στο 2025–26. Για μια ιδέα που
  περνά τον πρώτο έλεγχο, αξίζει να μαζέψεις όλες τις πρόσφατες κριτικές των σχετικών προϊόντων.
- **Το growth βγαίνει από λίγα αποσπάσματα** στα μικρά clusters, οπότε π.χ. ×1.4 από 13 αποσπάσματα είναι κυρίως θόρυβος.
- **Το evaluate δεν έχει τρέξει ακόμα στις 50 ετικέτες κριτικών**: τα σκορ στο `eval/history.csv` είναι μόνο από
  συζητήσεις (HN, Stack Exchange, Bluesky).

## Ρυθμίσεις

| Τι | Πώς |
|---|---|
| Άλλο LLM | `--model ...` ή `export IDEAFINDER_LLM=gemma4:12b` (το default είναι `qwen2.5:7b`) |
| Reasoning μοντέλα (gemma4, qwen3, deepseek-r1) | Το «thinking» κλείνει αυτόματα, αλλιώς κάθε απάντηση θα έπαιρνε ~7× περισσότερο |
| Χωρίς τα παραδείγματα στα prompts | `classify --no-examples` / `evaluate --no-examples` |
| Άλλο embedding μοντέλο | `--embed-model mxbai-embed-large` |
| Ollama σε άλλη διεύθυνση | `export IDEAFINDER_OLLAMA_URL=http://host:11434` |
| Περισσότερα/λιγότερα clusters | `analyze --min-cluster-size 10` (μικρότερο = περισσότερα, μικρότερα clusters) |
| Δικές σου φράσεις για το HN | `collect --phrases-file phrases.txt` (μία φράση ανά γραμμή) |
| Άλλη περίοδος | `collect --days 365 --se-days 1095` |
| Άλλα Stack Exchange sites | `collect --se-sites money,travel,diy` |
| Κριτικές: λιγότερες/περισσότερες | `collect --source as,wp --apps-per-genre 10 --wp-plugins 50 --per-app 5` |
| Κριτικές: άλλες κατηγορίες App Store | `collect --source as --app-genres 6007,6000` (τα ids στο `collect_as.py`) |
| Πόσα clusters αναλυτικά (και στο brief) | `analyze --top 60` (default 40) |

## Δομή

```
ideafinder/
  collect_hn.py    Hacker News (Algolia API) + λίστα φράσεων
  collect_se.py    Stack Exchange API
  collect_bsky.py  Bluesky public search API
  collect_as.py    App Store: top-grossing εφαρμογές + κριτικές 1–3★ (RSS)
  collect_wp.py    WordPress.org: δημοφιλή plugins + κριτικές 1–3★ (API + RSS)
  reviews.py       κοινά για τις κριτικές: ποιες 8 ανά προϊόν κρατάμε, τίτλοι
  patterns.py      regex για ανάγκες, παράπονα, σήματα πληρωμής
  extract.py       κρατά τις σχετικές προτάσεις από κάθε κείμενο
  classify.py      LLM σε 2 στάδια + παραδείγματα στα prompts
  embed.py         embeddings (Ollama) ή TF-IDF χωρίς LLM
  analyze.py       UMAP + HDBSCAN, βαθμολόγηση, «υπάρχει ήδη», LLM περιγραφή ανά cluster
  report.py        report.md / brief.md / CSV / JSON
  compete.py       έλεγχος ανταγωνισμού για τις top ιδέες
  evaluate.py      σύγκριση μοντέλων με το eval/labels.jsonl
  doctor.py        διάγνωση ενός μοντέλου του Ollama
  llm.py           μικρός client για το Ollama
eval/labels.jsonl  302 αποσπάσματα με ετικέτες (286 για μέτρηση, 16 παραδείγματα για τα prompts)
eval/runs/         οι απαντήσεις κάθε evaluate
eval/history.csv   ένα σκορ ανά evaluate
data/ideafinder.db SQLite με ό,τι έχει μαζευτεί
reports/           τα αποτελέσματα
```

## Επόμενα βήματα

- Shopify App Store ως πηγή κριτικών (πρώτα έλεγχος των όρων χρήσης)
- Αυτόματος γρήγορος έλεγχος ανταγωνισμού πριν από το brief
- Σύγκριση embedding μοντέλων σε ζευγάρια «ίδια ανάγκη ή όχι»
- Έλεγχος της κατάταξης: «τυφλή» βαθμολόγηση των top ιδεών και ρύθμιση των βαρών του score
- Πιο «καθαρά» clusters: έλεγχος του LLM ανά απόσπασμα αν ανήκει στο θέμα του cluster, πριν μετρηθούν άτομα και
  «would pay»
- `evaluate` στις ετικέτες κριτικών (App Store / WordPress.org)
