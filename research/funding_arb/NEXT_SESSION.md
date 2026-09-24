# Brief per la prossima sessione — carry su commodity via IBKR/TWS

Questo documento è scritto per una sessione Claude Code che gira **sulla macchina
dell'utente, con TWS o IB Gateway in esecuzione** e un account MEXEM/IBKR.

Prima di iniziare leggi `MEMORY.md`. Contiene tutto ciò che è già stato misurato,
e rimisurarlo è tempo buttato.

---

## Contesto in cinque righe

Su cripto è stato costruito e validato un carry delta-neutral (long spot / short
perp) con selezione trasversale. L'edge esiste — la classifica del funding
persiste con ρ≈0.65 su 3 anni, mai negativa in 155 periodi — ma rende **+0.96%
APR sul capitale**, sotto il risk-free, perché l'edge lordo sfiora appena la
soglia dei costi (34bp di round trip).

Sui futures quella soglia è **4-20 volte più bassa** (`MEMORY.md` §8). È l'unica
ragione per cui vale la pena estendere l'analisi, e va verificata prima di tutto
il resto.

---

## L'obiettivo di questa sessione

**Una sola domanda: la classifica del carry fra commodity persiste come persiste
quella del funding?**

Non costruire un backtest. Non scrivere un esecutore. Rispondere a quella
domanda, che costa poco e può chiudere l'intero filone.

### Criterio go/stop, deciso ora e non dopo aver visto i numeri

Misura la ρ di Spearman fra carry trailing e carry realizzato nel periodo
successivo, su finestre non sovrapposte, come fa `persistence.py` sul funding.

- **ρ medio > 0.30 e positiva in almeno l'80% dei periodi** → prosegui: ha senso
  costruire il backtest.
- **ρ medio < 0.15, oppure segno instabile** → fermati e riferiscilo. Il filone si
  chiude qui, ed è un risultato utile quanto l'altro.
- **In mezzo** → riporta i numeri e chiedi, non decidere da solo.

Fissare la soglia prima di guardare i dati è il punto. Su cripto la ρ è ~0.65; una
commodity carry a 0.35 sarebbe comunque tradabile *perché i costi sono 10× più
bassi*, ed è per questo che la soglia qui è più permissiva.

---

## Setup

```bash
pip install ib_async          # fork mantenuto di ib_insync; l'originale è fermo
```

TWS o IB Gateway in esecuzione, API abilitata:
`Configure → API → Settings → Enable ActiveX and Socket Clients`, porta 7497
(paper) o 7496 (live). **Usa la porta paper.** Questa sessione non deve toccare
il conto reale.

Serve la sottoscrizione ai dati storici per gli exchange interessati (NYMEX,
COMEX, CBOT per le commodity). Senza, `reqHistoricalData` restituisce un errore
di permessi invece dei dati — se succede, è quello, non un bug del codice.

**Pacing:** IBKR limita a circa 60 richieste storiche ogni 10 minuti e ti
penalizza se sforI. Lo script fornito rispetta già un intervallo; non toglierlo.
È la stessa lezione del bug #5 in `MEMORY.md`: un throttle non gestito non dà
errore, scarta silenziosamente dei contratti e rende il campione irriproducibile.

---

## Passo 1 — scaricare le curve

`fetch_ib_curves.py` (in questa cartella) enumera le scadenze disponibili per un
paniere di future e ne scarica le barre giornaliere. **Non è mai stato eseguito
contro un TWS reale** — è stato scritto da un container senza accesso a IBKR.
Trattalo come una bozza da far funzionare, non come codice collaudato.

```bash
python research/funding_arb/fetch_ib_curves.py \
    --port 7497 --years 5 --out research/funding_arb/data/ib_curves.csv
```

Paniere iniziale, scelto per coprire settori decorrelati e avere curve liquide:

| settore | simboli | exchange |
|---|---|---|
| energia | CL, NG, HO, RB | NYMEX |
| metalli | GC, SI, HG | COMEX |
| agricoli | ZC, ZS, ZW | CBOT |
| bestiame | LE, HE | CME |

Schema CSV atteso:

```
date,symbol,exchange,expiry,close,volume
2026-09-19,CL,NYMEX,202611,107.02,412331
2026-09-19,CL,NYMEX,202612,106.41,201044
```

Una riga per (giorno, contratto). Servono **almeno due scadenze per simbolo per
ogni giorno**, altrimenti la curva non è definita e il carry non è calcolabile.

Verifica subito, prima di andare avanti:
- quanti anni di storia sono arrivati davvero (IBKR taglia a seconda del prodotto);
- se ci sono buchi (il bug #5 insegna: conta le righe per simbolo per mese);
- se i prezzi sono in punti o centesimi (CL è in dollari/barile, ZC in
  centesimi/bushel — se lo sbagli il carry esce di un fattore 100).

---

## Passo 2 — costruire la serie di carry

Il carry di una commodity è il **roll yield**: la pendenza della curva, annualizzata.

```
carry_apr = (F_vicino / F_lontano - 1) * (365 / giorni_fra_le_due_scadenze)
```

Positivo = backwardation = ti pagano per essere lungo. È l'analogo esatto del
funding positivo che paga chi è corto sul perp.

Attenzione a tre cose:
- **Usa sempre la stessa coppia di scadenze relative** (prima e seconda attiva),
  non scadenze fisse, altrimenti misuri il passare del tempo invece della curva.
- **Alla rollata la coppia cambia**: assicurati che non produca un salto
  artificiale nella serie.
- **Stagionalità**: NG e gli agricoli hanno curve stagionali fortissime. Un carry
  alto a novembre su NG non è un premio, è l'inverno. Se la ρ regge solo grazie
  alla stagionalità, non è un edge tradabile in modo trasversale — va verificato
  de-stagionalizzando (confronta ogni contratto con lo stesso mese degli anni
  precedenti).

Quest'ultimo punto è il rischio più serio di tutto il passo: la stagionalità
produce **persistenza spuria**, che somiglia esattamente al risultato cercato.

### La stagionalità come segnale, non come contaminante

È stato chiesto se si possa invertire il problema e usare la stagionalità come
edge (cfr. SeasonAlgo e lo spread trading stagionale). Vedi `MEMORY.md` §11-bis
per l'indagine completa. In sintesi, per questa sessione:

- **Non fare scanning.** Lo spazio di ricerca è di ~120 milioni di combinazioni e
  con 30 anni di dati ci si attendono ~509 spread con 27/30 anni vincenti **per
  puro caso**. Qualunque pattern trovato scansionando è indistinguibile dal
  rumore, e l'evidenza out-of-sample pubblicata (arXiv 2609.12227) è negativa.
- **Il test giusto è una decomposizione, non una ricerca.** Il carry di una
  commodity e la sua stagionalità sono in larga parte **lo stesso numero**: la
  pendenza della curva a novembre sul gas *è* la stagionalità invernale. Quindi la
  domanda ben posta non è "quale dei due funziona" ma:

  > quanta parte della ρ del carry sopravvive dopo aver rimosso la componente
  > stagionale media di ogni contratto?

  Si ottiene con una sola misura aggiuntiva: calcolare il carry medio di ogni
  (simbolo, mese-di-calendario) sulla storia **esclusa l'osservazione corrente**,
  sottrarlo, e rifare il test di persistenza sul residuo.

  - Se ρ crolla → il carry commodity **è** stagionalità, quindi è già prezzato in
    curva e non è un premio. Filone chiuso.
  - Se ρ regge → esiste un premio strutturale oltre la stagionalità, che è
    esattamente l'analogo del funding. Si prosegue.

  Una misura, due domande. Falla nel passo 3, non in un passo separato.
  **`seasonality.py` la implementa già** (`decompose`, `deseasonalize`,
  `carry_apr`): passagli una serie mensile per commodity e leggi le colonne.
- **Già misurato su 4 commodity energetiche con dati EIA gratuiti** (`MEMORY.md`
  §11-bis): a 12 mesi la persistenza è quasi tutta stagionale, ma **a 1-3 mesi
  destagionalizzare la rafforza** — su NATGAS da −0.102 a +0.338. La stagionalità
  nasconde il carry invece di fornirlo. Serve replicarlo su un universo più ampio,
  ed è questo che i dati IBKR devono permettere.
- **Se qualcosa merita un test dedicato dopo**, è il vincolo di full carry (vedi
  `MEMORY.md` §11-bis): in contango lo spread è limitato dal costo di stoccaggio,
  in backwardation no. È un'asimmetria fisica, non statistica.

---

## Passo 3 — il test di persistenza

`persistence.py` è **già riutilizzabile così com'è**. Vuole un dizionario
`{simbolo: [FundingPoint(time_ms, rate), ...]}` e la `measure()` fa il resto.

Costruisci `FundingPoint` con un `rate` giornaliero equivalente
(`carry_apr / 365`) e passa `interval_hours=24`, così `annualize()` restituisce il
carry annualizzato corretto.

```python
from research.funding_arb.persistence import measure, format_persistence

panel = {sym: [FundingPoint(ts, carry_apr/365.0) for ts, carry_apr in serie]
         for sym, serie in curve.items()}
risultati = [measure(panel, start_ms, end_ms, w, interval_hours=24.0)
             for w in (7, 14, 30, 90)]
print(format_persistence(risultati, len(panel)))
```

Se il numero di simboli scende sotto 20, abbassa `min_coins` — ma sappi che con
12 commodity l'IC trasversale è molto più rumoroso che con 39 coin, e gli
intervalli vanno letti di conseguenza.

---

## Passo 3-bis — gli spread stagionali sui futures (priorità alta)

`MEMORY.md` §11-quater: sugli spot EIA il crack benzina *long dal 21-26 gennaio
per 90 giorni* vince 23/26 anni fuori campione (CI 71-96%), scelto identico in
25 anni su 26. **Ma è spot.** Il test decisivo è sui futures, perché il futures di
maggio a gennaio incorpora già l'attesa primaverile.

Con i dati IBKR costruisci, per ogni anno, lo spread fra contratti dello **stesso
mese di consegna** (es. `42 * RB_maggio − CL_maggio`, in $/bbl) e passalo a
`walk_forward`:

```python
from research.funding_arb.seasonal_walkforward import walk_forward, format_results
r = walk_forward("RB-CL maggio", serie_giornaliera, lookback=15, min_wins=12, cost=0.05)
print(format_results([r]))
for anno, finestra, pnl in r.best_picks: print(anno, finestra, pnl)
```

Attenzione: una serie per contratto vive ~1 anno, quindi serve concatenare gli
anni con il contratto dello stesso mese (maggio 2010 per il 2010, maggio 2011 per
il 2011...), non una serie continua che rolla.

Criteri, fissati ora:
- **OOS win del best pick con limite inferiore di Wilson > 55%** e
- **la stessa finestra scelta in almeno 2/3 degli anni** (stabilità) →
  strategia candidata al paper trading.
- Se l'OOS scende verso il 50%, la stagionalità era prezzata: filone chiuso.

## Passo 4 — riferire

Riporta la tabella di `format_persistence`, il confronto con ρ≈0.65 del cripto, e
la decisione secondo il criterio go/stop fissato sopra. Aggiorna `MEMORY.md` con
quello che hai misurato, che il risultato sia positivo o negativo.

**Un risultato negativo va scritto con la stessa cura di uno positivo.** Metà del
valore di `MEMORY.md` sono i vicoli ciechi documentati.

---

## Cosa riusare e cosa riscrivere

| riutilizzabile così com'è | da riscrivere |
|---|---|
| `persistence.py` (ρ, quintili, finestre disgiunte) | adapter dei venue (`venues.py`) |
| `costs.py` (modello di costo, breakeven) | contabilità: il funding si accumula, il roll yield si realizza alla rollata |
| `metrics.py` (Wilson, Sharpe, drawdown, attribuzione) | gestione scadenze, date di first notice, calendari di roll |
| `basis.py` (qualità della copertura) | — |
| la logica del gate di volatilità | — |

La **metodologia è agnostica rispetto all'asset**; l'infrastruttura dati no.

---

## Limiti da rispettare

- **Nessun ordine, né reale né simulato**, in questa fase. L'obiettivo è misurare,
  e il conto serve solo come fonte dati.
- Porta paper (7497). Se ti trovi connesso alla 7496, fermati.
- Non scrivere codice che *potrebbe* piazzare ordini: niente `placeOrder`, nemmeno
  commentato o dietro un flag.

---

## Cose che non so e che scoprirai tu

Onestà su cosa è ipotesi e cosa è misurato:

- **Quanta storia dà IBKR sui futures.** Non l'ho potuto verificare. Potrebbe
  essere molto meno di 5 anni su alcuni prodotti, il che ridurrebbe il campione.
- **Se `fetch_ib_curves.py` funziona.** Scritto alla cieca contro la
  documentazione, mai eseguito.
- **Se i costi calcolati in `MEMORY.md` §8 reggono.** Sono basati sui listini
  pubblici IBKR e su un tick di spread attraversato. I fill reali possono essere
  peggiori, specie sui mesi differiti che sono molto meno liquidi del front —
  **e lo spread calendario si esegue sul book dello spread, non sui due outright**,
  il che può andare in entrambe le direzioni.
- **Se la ρ del carry commodity regge de-stagionalizzata.** È la domanda che può
  far crollare tutto il passo 2.
