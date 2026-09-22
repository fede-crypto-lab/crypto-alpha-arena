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
